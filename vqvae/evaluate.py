"""
Evaluation and visualization for trained VQ-VAE.

Usage:
    python -m vqvae.evaluate --checkpoint checkpoints/vqvae_best.pt
    python -m vqvae.evaluate --checkpoint checkpoints/vqvae_best.pt --full_res --num_images 16

Features:
    1. Reconstruction grid: side-by-side original vs reconstructed at 256x512
    2. Full-resolution output: decode and resize to native 840x420
    3. Codebook statistics: usage histogram, utilization rate, perplexity
    4. Single-frame metrics: PSNR, SSIM, LPIPS, MSE, MAE
    5. FID / KID (when --heavy_metrics is set)
    6. All results logged to W&B
"""

import argparse
import os
import math

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae.model import VQVAE
from vqvae.dataset import create_dataloaders

from evaluation.config import load_eval_config
from evaluation.single_frame_metrics import (
    evaluate_single_frame,
    compute_codebook_stats,
    compute_fid,
    compute_kid,
)
from logger.wandb_logger import WandbLogger
from logger.metric_names import M
from logger.architecture import serialize_model_architecture


def _repeat_to_min_samples(items: list, min_samples: int) -> list:
    """Repeat elements cyclically so the returned list has at least min_samples."""
    if min_samples <= 0 or len(items) >= min_samples or not items:
        return items
    return [items[i % len(items)] for i in range(min_samples)]


# ---------------------------------------------------------------------------
# Reconstruction grid (256x512)
# ---------------------------------------------------------------------------


@torch.no_grad()
def make_reconstruction_grid(model, val_loader, device, num_images=8):
    """
    Generate a grid of [original | reconstruction] image pairs.

    Returns:
        grid: Tensor image grid, ready to save.
        originals: [N, 3, 256, 512] original images in [-1, 1].
        reconstructions: [N, 3, 256, 512] reconstructed images in [-1, 1].
    """
    model.eval()

    originals = []
    for batch in val_loader:
        originals.append(batch)
        if sum(b.shape[0] for b in originals) >= num_images:
            break
    originals = torch.cat(originals, dim=0)[:num_images].to(device)

    x_recon, _, _, _ = model(originals)

    grid_images = []
    for i in range(num_images):
        grid_images.append(originals[i])
        grid_images.append(x_recon[i])

    grid = vutils.make_grid(
        torch.stack(grid_images),
        nrow=2,
        normalize=True,
        value_range=(-1, 1),
        padding=4,
        pad_value=0.5,
    )

    return grid, originals, x_recon


# ---------------------------------------------------------------------------
# Full-resolution (840x420) output
# ---------------------------------------------------------------------------


@torch.no_grad()
def to_full_resolution(images_256x512):
    """
    Resize images from 256x512 to native 420x840 using bicubic interpolation.
    """
    images_full = F.interpolate(
        images_256x512,
        size=(420, 840),
        mode="bicubic",
        align_corners=False,
    ).clamp(-1, 1)

    images_uint8 = ((images_full + 1.0) * 127.5).clamp(0, 255).byte()
    return images_uint8


@torch.no_grad()
def save_full_resolution_samples(model, val_loader, device, save_dir, num_images=8):
    """
    Encode -> decode -> resize to 840x420 and save individual PNGs.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    originals = []
    for batch in val_loader:
        originals.append(batch)
        if sum(b.shape[0] for b in originals) >= num_images:
            break
    originals = torch.cat(originals, dim=0)[:num_images].to(device)

    x_recon, _, _, _ = model(originals)

    orig_full = to_full_resolution(originals)
    recon_full = to_full_resolution(x_recon)

    for i in range(num_images):
        orig_img = orig_full[i].permute(1, 2, 0).cpu().numpy()
        from PIL import Image
        Image.fromarray(orig_img).save(
            os.path.join(save_dir, f"original_{i:03d}.png")
        )

        recon_img = recon_full[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(recon_img).save(
            os.path.join(save_dir, f"recon_{i:03d}.png")
        )

        comparison = np.concatenate([orig_img, recon_img], axis=1)
        Image.fromarray(comparison).save(
            os.path.join(save_dir, f"compare_{i:03d}.png")
        )

    print(f"Saved {num_images} full-resolution samples to {save_dir}")


# ---------------------------------------------------------------------------
# Codebook statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_codebook_stats_full(model, val_loader, device):
    """
    Compute codebook utilization statistics over the full validation set.
    """
    model.eval()
    num_embeddings = model.quantizer.num_embeddings
    usage_counts = torch.zeros(num_embeddings, dtype=torch.long, device=device)

    total_tokens = 0
    for batch in tqdm(val_loader, desc="Computing codebook stats"):
        x = batch.to(device)
        z_e = model.encoder(x)
        _, _, indices, _ = model.quantizer(z_e)
        flat = indices.reshape(-1)
        usage_counts.scatter_add_(
            0, flat, torch.ones_like(flat, dtype=torch.long)
        )
        total_tokens += flat.numel()

    usage_counts = usage_counts.cpu()

    used_codes = (usage_counts > 0).sum().item()
    utilization = used_codes / num_embeddings

    probs = usage_counts.float() / usage_counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * probs.log()).sum().item()
    perplexity = math.exp(entropy)

    stats = {
        "usage_counts": usage_counts.numpy(),
        "utilization": utilization,
        "used_codes": used_codes,
        "total_codes": num_embeddings,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }

    return stats


def print_codebook_stats(stats):
    print(f"\n{'='*50}")
    print(f"Codebook Statistics")
    print(f"{'='*50}")
    print(f"  Total codebook entries: {stats['total_codes']}")
    print(f"  Entries used (>0):      {stats['used_codes']}")
    print(f"  Utilization:            {stats['utilization']:.2%}")
    print(f"  Perplexity:             {stats['perplexity']:.1f} / {stats['total_codes']}")
    print(f"  Total tokens processed: {stats['total_tokens']:,}")

    counts = stats["usage_counts"]
    nonzero = counts[counts > 0]
    if len(nonzero) > 0:
        print(f"\n  Usage distribution (non-zero codes):")
        print(f"    Min:    {nonzero.min()}")
        print(f"    Max:    {nonzero.max()}")
        print(f"    Mean:   {nonzero.mean():.1f}")
        print(f"    Median: {np.median(nonzero):.1f}")
    print(f"{'='*50}\n")


def save_usage_histogram(stats, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = stats["usage_counts"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(range(len(counts)), counts, width=1.0)
        axes[0].set_xlabel("Codebook Index")
        axes[0].set_ylabel("Usage Count")
        axes[0].set_title(
            f"Codebook Usage (utilization: {stats['utilization']:.1%}, "
            f"perplexity: {stats['perplexity']:.0f})"
        )

        sorted_counts = np.sort(counts)[::-1]
        axes[1].bar(range(len(sorted_counts)), sorted_counts, width=1.0)
        axes[1].set_xlabel("Rank")
        axes[1].set_ylabel("Usage Count")
        axes[1].set_title("Codebook Usage (sorted by frequency)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved usage histogram to {save_path}")

    except ImportError:
        print("matplotlib not installed -- skipping histogram plot")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    # Create model with same config
    model = VQVAE(
        latent_dim=config.get("latent_dim", 16),
        num_embeddings=config.get("num_embeddings", 1024),
        commitment_cost=config.get("commitment_cost", 0.25),
        ema_decay=config.get("ema_decay", 0.99),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    print(f"Loaded model from epoch {epoch}, val_loss={val_loss}")

    # Data (validation only)
    _, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- W&B Logger ---
    eval_config = load_eval_config(args.wandb_config)
    wb = WandbLogger(
        config={
            "model": "VQ-VAE",
            "mode": "evaluation",
            "checkpoint": args.checkpoint,
            "epoch": epoch,
            **config,
        },
        run_name=args.run_name,
        tags=["vqvae", "evaluation"],
        enabled=not args.no_wandb,
        metric_enabled_fn=eval_config.is_metric_enabled,
    )
    wb.log_architecture(model, "VQ-VAE", extra_metadata={
        "latent_dim": config.get("latent_dim", 16),
        "num_embeddings": config.get("num_embeddings", 1024),
    })

    # 1. Reconstruction grid
    print("\nGenerating reconstruction grid...")
    grid, originals, reconstructions = make_reconstruction_grid(
        model, val_loader, device, num_images=args.num_images
    )
    grid_path = os.path.join(output_dir, "reconstruction_grid.png")
    save_image(grid, grid_path)
    print(f"Saved reconstruction grid to {grid_path}")
    wb.log_image("Evaluation/Reconstruction Grid", grid_path)

    # 2. Full-resolution output
    if args.full_res:
        print("\nGenerating full-resolution (840x420) samples...")
        full_res_dir = os.path.join(output_dir, "full_resolution")
        save_full_resolution_samples(
            model, val_loader, device, full_res_dir, num_images=args.num_images
        )

    # 3. Single-frame metrics (averaged over validation set)
    print("\nComputing single-frame metrics...")
    all_frame_metrics = {}
    num_batches = 0
    real_images_np = []
    fake_images_np = []

    for batch in tqdm(val_loader, desc="Single-frame metrics"):
        x = batch.to(device)
        with torch.no_grad():
            x_recon, _, _, _ = model(x)

        frame_metrics = evaluate_single_frame(x_recon, x, compute_components=True)
        for k, v in frame_metrics.items():
            all_frame_metrics[k] = all_frame_metrics.get(k, 0.0) + v
        num_batches += 1

        if args.heavy_metrics:
            real_np = ((x + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            fake_np = ((x_recon + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
            for i in range(real_np.shape[0]):
                real_images_np.append(real_np[i])
                fake_images_np.append(fake_np[i])

    avg_frame_metrics = {k: v / num_batches for k, v in all_frame_metrics.items()}
    print(f"\nSingle-frame metrics (avg over val set):")
    for k, v in avg_frame_metrics.items():
        print(f"  {k}: {v:.4f}")
    wandb_frame = {f"Evaluation/{k}": v for k, v in avg_frame_metrics.items()}
    wb.log(wandb_frame)

    # 4. Codebook statistics
    print("\nComputing codebook statistics...")
    stats = compute_codebook_stats_full(model, val_loader, device)
    print_codebook_stats(stats)
    wb.log({
        M.codebook_perplexity(): stats["perplexity"],
        M.codebook_utilization(): stats["utilization"],
    })

    hist_path = os.path.join(output_dir, "codebook_usage.png")
    save_usage_histogram(stats, hist_path)
    wb.log_image("Evaluation/Codebook Usage Histogram", hist_path)

    # 5. FID / KID
    if args.heavy_metrics and len(real_images_np) > 0:
        print("\nComputing FID / KID...")
        num_samples = min(len(real_images_np), eval_config.metric_num_samples("fid"))
        fid_min_samples = eval_config.metric_min_samples("fid", default=1)
        kid_min_samples = eval_config.metric_min_samples("kid", default=1)

        real_fid = _repeat_to_min_samples(real_images_np[:num_samples], fid_min_samples)
        fake_fid = _repeat_to_min_samples(fake_images_np[:num_samples], fid_min_samples)
        try:
            fid_val = compute_fid(real_fid, fake_fid)
            print(f"  FID: {fid_val:.2f}")
            wb.log({M.fid(): fid_val})
        except ImportError as e:
            print(f"  FID skipped: {e}")
        except Exception as e:
            print(f"  FID skipped (tiny-sample backend error): {e}")

        try:
            real_kid = _repeat_to_min_samples(real_images_np[:num_samples], kid_min_samples)
            fake_kid = _repeat_to_min_samples(fake_images_np[:num_samples], kid_min_samples)
            kid_val = compute_kid(real_kid, fake_kid)
            print(f"  KID: {kid_val:.4f}")
            wb.log({M.kid(): kid_val})
        except ImportError as e:
            print(f"  KID skipped: {e}")
        except Exception as e:
            print(f"  KID skipped (tiny-sample backend error): {e}")

    wb.log_summary(avg_frame_metrics)
    wb.finish()
    print("\nEvaluation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained VQ-VAE")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/vqvae_best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="eval_output",
                        help="Output directory for visualizations")
    parser.add_argument("--num_images", type=int, default=8,
                        help="Number of images to visualize")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--full_res", action="store_true",
                        help="Also generate full-resolution 840x420 outputs")
    parser.add_argument("--heavy_metrics", action="store_true",
                        help="Compute heavy metrics (FID, KID)")

    # W&B
    parser.add_argument("--wandb_config", type=str, default="wandb_config.json",
                        help="Path to wandb/eval config JSON")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name override")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
