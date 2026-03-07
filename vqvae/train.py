"""
Training script for the VQ-VAE.

Usage:
    python -m vqvae.train                         # defaults
    python -m vqvae.train --epochs 100 --batch_size 32
    python -m vqvae.train --data_dir dataset --resume checkpoints/vqvae_latest.pt

Hyperparameters (plan defaults):
    Optimizer:  AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-6
    Scheduler:  Linear warmup 5 epochs, then cosine decay to 1e-5
    Batch size: 16
    Epochs:     50 (early stopping, patience=10)
    Mixed prec: FP16 autocast + GradScaler (VQ internals stay float32)
    Grad clip:  max_norm=1.0
"""

import argparse
import os
import time
import warnings

# Suppress TensorFlow oneDNN warnings from lpips
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from tqdm import tqdm

from vqvae.model import VQVAE
from vqvae.dataset import create_dataloaders
from vqvae.losses import PerceptualLoss, compute_loss

from evaluation.config import load_eval_config
from evaluation.orchestrator import EvalOrchestrator
from logger.wandb_logger import WandbLogger
from logger.metric_names import M
from logger.gradient_stats import compute_gradient_stats
from logger.architecture import serialize_model_architecture


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch, min_lr):
    """
    Build a built-in scheduler chain: Linear warmup -> Cosine annealing.
    """
    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_steps = max(int(total_epochs) * steps_per_epoch, 1)
    warmup_steps = max(0, int(warmup_epochs) * steps_per_epoch)

    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )

    if warmup_steps >= total_steps:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=total_steps,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(model, val_loader, perceptual_loss_fn, device):
    """
    Run validation and return average loss dict.
    """
    model.eval()
    total_losses = {}
    num_batches = 0

    for batch in val_loader:
        x = batch.to(device)

        # No autocast for validation -- simpler and more accurate metrics
        x_recon, commit_loss, indices, usage = model(x)
        _, loss_dict = compute_loss(x_recon, x, commit_loss, perceptual_loss_fn)
        loss_dict["codebook_usage"] = torch.tensor(usage)

        for k, v in loss_dict.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        num_batches += 1

    model.train()

    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    return avg_losses


# ---------------------------------------------------------------------------
# Save reconstruction grid
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_reconstruction_grid(model, val_loader, device, save_path, num_images=8):
    """
    Save a side-by-side grid of original vs reconstructed images.
    """
    model.eval()

    # Gather images from validation set
    originals = []
    for batch in val_loader:
        originals.append(batch)
        if sum(b.shape[0] for b in originals) >= num_images:
            break
    if not originals:
        model.train()
        return None

    originals = torch.cat(originals, dim=0)[:num_images].to(device)
    actual_num_images = originals.shape[0]

    x_recon, _, _, _ = model(originals)

    # Interleave: orig1, recon1, orig2, recon2, ...
    grid_images = []
    for i in range(actual_num_images):
        grid_images.append(originals[i])
        grid_images.append(x_recon[i])

    grid = vutils.make_grid(
        torch.stack(grid_images),
        nrow=2,  # 2 columns: original | reconstruction
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )

    # Save as PNG
    from torchvision.utils import save_image
    save_image(grid, save_path)

    model.train()
    return grid


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Model
    model = VQVAE(
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        commitment_cost=args.commitment_cost,
        ema_decay=args.ema_decay,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # Perceptual loss (frozen VGG)
    perceptual_loss_fn = PerceptualLoss(net="vgg").to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    if args.warmup_epochs > args.epochs:
        raise ValueError(
            f"warmup_epochs ({args.warmup_epochs}) cannot be greater than epochs ({args.epochs})"
        )

    # Scheduler
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr=args.min_lr,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    if device.type == "cuda":
        print(f"Mixed precision (FP16) enabled - Tensor Cores will be utilized")

    # --- W&B Logger ---
    eval_config = load_eval_config(args.wandb_config)
    run_config = {
        "model": "VQ-VAE",
        "latent_dim": args.latent_dim,
        "num_embeddings": args.num_embeddings,
        "commitment_cost": args.commitment_cost,
        "ema_decay": args.ema_decay,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "eval_config": eval_config.raw,
    }
    wb = WandbLogger(
        config=run_config,
        run_name=args.run_name,
        group="VQ-VAE",
        tags=["vqvae", "training"],
        enabled=not args.no_wandb,
        metric_enabled_fn=eval_config.is_metric_enabled,
    )

    wb.log_architecture(model, "VQ-VAE", extra_metadata={
        "input_shape": [3, 256, 512],
        "latent_shape": [args.latent_dim, 32, 32],
        "num_embeddings": args.num_embeddings,
    })

    orchestrator = EvalOrchestrator(eval_config, wb, device)

    # Checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.recon_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    # Gradient stats config (epoch-based logging)
    grad_enabled = eval_config.gradient_stats_enabled
    grad_modules = eval_config.gradient_modules("vqvae")

    # --------------- Training Loop ---------------
    print(f"\n{'='*60}")
    print(f"Training VQ-VAE for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {}
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            x = batch.to(device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                x_recon, commit_loss, indices, usage = model(x)
                total_loss, loss_dict = compute_loss(
                    x_recon, x, commit_loss, perceptual_loss_fn
                )

            loss_dict["codebook_usage"] = torch.tensor(usage)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1

            # Accumulate epoch losses
            for k, v in loss_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0.0
                epoch_losses[k] += v.item()

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "recon": f"{loss_dict['recon_l1'].item():.4f}",
                "cb_use": f"{usage:.2%}",
            })

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch = {k: v / max(len(train_loader), 1) for k, v in epoch_losses.items()}
        epoch_step = epoch + 1
        print(f"\nEpoch {epoch+1} train (avg) | "
              f"loss: {avg_epoch['total']:.4f} | "
              f"recon: {avg_epoch['recon_l1']:.4f} | "
              f"percep: {avg_epoch['perceptual']:.4f} | "
              f"commit: {avg_epoch['commitment']:.4f} | "
              f"cb_use: {avg_epoch['codebook_usage']:.2%} | "
              f"time: {epoch_time:.1f}s")

        # W&B train metrics (epoch-based step)
        current_lr = optimizer.param_groups[0]["lr"]
        wb.log({
            M.train_loss("Total"): avg_epoch["total"],
            M.train_loss("L1 Reconstruction"): avg_epoch["recon_l1"],
            M.train_loss("LPIPS"): avg_epoch["perceptual"],
            M.train_loss("Commitment"): avg_epoch["commitment"],
            M.codebook_usage("Train"): avg_epoch["codebook_usage"],
            M.learning_rate(): current_lr,
        }, step=epoch_step, commit=False)

        # Gradient stats are epoch-based to keep all metrics on the same axis.
        if grad_enabled:
            grad_stats = compute_gradient_stats(model, grad_modules, model_name="VQ-VAE")
            wb.log(grad_stats, step=epoch_step, commit=False)

        # --------------- Validation ---------------
        val_losses = validate(model, val_loader, perceptual_loss_fn, device)
        print(f"Epoch {epoch+1} val   (avg) | "
              f"loss: {val_losses['total']:.4f} | "
              f"recon: {val_losses['recon_l1']:.4f} | "
              f"percep: {val_losses['perceptual']:.4f} | "
              f"commit: {val_losses['commitment']:.4f} | "
              f"cb_use: {val_losses['codebook_usage']:.2%}")

        # W&B val metrics
        wb.log({
            M.val_loss("Total"): val_losses["total"],
            M.val_loss("L1 Reconstruction"): val_losses["recon_l1"],
            M.val_loss("LPIPS"): val_losses["perceptual"],
            M.val_loss("Commitment"): val_losses["commitment"],
            M.codebook_usage("Val"): val_losses["codebook_usage"],
        }, step=epoch_step, commit=False)

        # --------------- Evaluation Orchestrator ---------------
        eval_metrics = orchestrator.run_vqvae_epoch_eval(
            model, val_loader, epoch_step, epoch_step, commit=False
        )

        # --------------- Reconstruction Grid ---------------
        grid_path = os.path.join(args.recon_dir, f"recon_epoch_{epoch+1:03d}.png")
        grid = save_reconstruction_grid(model, val_loader, device, grid_path, num_images=8)
        if grid is not None:
            print(f"Saved reconstruction grid to {grid_path}")
            wb.log_image("Evaluation/Reconstruction Grid", grid_path, step=epoch_step, commit=True)
        else:
            print("Skipped reconstruction grid (validation loader is empty)")
            # Always finalize the W&B step once per epoch.
            wb.log({M.learning_rate(): current_lr}, step=epoch_step, commit=True)

        # --------------- Checkpointing ---------------
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_losses["total"],
            "best_val_loss": best_val_loss,
            "config": {
                "latent_dim": args.latent_dim,
                "num_embeddings": args.num_embeddings,
                "commitment_cost": args.commitment_cost,
                "ema_decay": args.ema_decay,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
            },
        }

        # Save latest
        latest_path = os.path.join(args.checkpoint_dir, "vqvae_latest.pt")
        torch.save(checkpoint, latest_path)
        if eval_config.log_latest_checkpoint:
            wb.log_checkpoint(latest_path, "vqvae-latest", metadata={"epoch": epoch, "val_loss": val_losses["total"]})

        # Save best
        val_total = val_losses["total"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            best_path = os.path.join(args.checkpoint_dir, "vqvae_best.pt")
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)
            print(f"New best model! val_loss={val_total:.4f}")
            if eval_config.log_best_checkpoint:
                wb.log_checkpoint(best_path, "vqvae-best", metadata={"epoch": epoch, "val_loss": val_total})
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {args.patience} epochs)")
            break

        print()

    wb.log_summary({"best_val_loss": best_val_loss})
    wb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'vqvae_best.pt')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on game frames")

    # Data
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing session_*/pair_*.npz files")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader num_workers")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use from dataset (None = use all)")

    # Model
    parser.add_argument("--latent_dim", type=int, default=16,
                        help="Latent channel dimension")
    parser.add_argument("--num_embeddings", type=int, default=1024,
                        help="VQ codebook size")
    parser.add_argument("--commitment_cost", type=float, default=0.25,
                        help="VQ commitment loss weight")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA decay for codebook updates")

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate (end of cosine decay)")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs")
    parser.add_argument("--patience", type=int, default=1000,
                        help="Early stopping patience (epochs)")

    # Logging
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log to W&B every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--recon_dir", type=str, default="reconstructions",
                        help="Reconstruction grid save directory")

    # W&B
    parser.add_argument("--wandb_config", type=str, default="wandb_config.json",
                        help="Path to wandb/eval config JSON")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name override")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
