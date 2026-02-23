"""
Training script for the VQ-VAE.

Usage:
    python -m vqvae.train                         # defaults
    python -m vqvae.train --epochs 100 --batch_size 32
    python -m vqvae.train --data_dir dataset --resume checkpoints/vqvae_latest.pt

Hyperparameters (plan defaults):
    Optimizer:  AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=1e-6
    Scheduler:  Linear warmup 500 steps, then cosine decay to 1e-5
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.utils as vutils

from vqvae.model import VQVAE
from vqvae.dataset import create_dataloaders
from vqvae.losses import PerceptualLoss, compute_loss


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def get_lr_lambda(warmup_steps, total_steps, min_lr, max_lr):
    """
    Returns a lambda for torch.optim.lr_scheduler.LambdaLR.

    - Linear warmup from 0 to max_lr over `warmup_steps`.
    - Cosine decay from max_lr to min_lr over remaining steps.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / max(warmup_steps, 1)
        else:
            # Cosine decay
            import math
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / max_lr) + (1.0 - min_lr / max_lr) * cosine
    return lr_lambda


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
    originals = torch.cat(originals, dim=0)[:num_images].to(device)

    x_recon, _, _, _ = model(originals)

    # Interleave: orig1, recon1, orig2, recon2, ...
    grid_images = []
    for i in range(num_images):
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

    # Scheduler
    lr_lambda = get_lr_lambda(
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr=args.min_lr,
        max_lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    if device.type == "cuda":
        print(f"Mixed precision (FP16) enabled - Tensor Cores will be utilized")

    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Checkpointing
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
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

            # Log every N steps
            if global_step % args.log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/total_loss", loss_dict["total"].item(), global_step)
                writer.add_scalar("train/recon_l1", loss_dict["recon_l1"].item(), global_step)
                writer.add_scalar("train/perceptual", loss_dict["perceptual"].item(), global_step)
                writer.add_scalar("train/commitment", loss_dict["commitment"].item(), global_step)
                writer.add_scalar("train/codebook_usage", usage, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total'].item():.4f}",
                "recon": f"{loss_dict['recon_l1'].item():.4f}",
                "cb_use": f"{usage:.2%}",
            })

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch = {k: v / max(len(train_loader), 1) for k, v in epoch_losses.items()}
        print(f"\nEpoch {epoch+1} train (avg) | "
              f"loss: {avg_epoch['total']:.4f} | "
              f"recon: {avg_epoch['recon_l1']:.4f} | "
              f"percep: {avg_epoch['perceptual']:.4f} | "
              f"commit: {avg_epoch['commitment']:.4f} | "
              f"cb_use: {avg_epoch['codebook_usage']:.2%} | "
              f"time: {epoch_time:.1f}s")

        # --------------- Validation ---------------
        val_losses = validate(model, val_loader, perceptual_loss_fn, device)
        print(f"Epoch {epoch+1} val   (avg) | "
              f"loss: {val_losses['total']:.4f} | "
              f"recon: {val_losses['recon_l1']:.4f} | "
              f"percep: {val_losses['perceptual']:.4f} | "
              f"commit: {val_losses['commitment']:.4f} | "
              f"cb_use: {val_losses['codebook_usage']:.2%}")

        # Tensorboard val metrics
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, global_step)

        # --------------- Reconstruction Grid ---------------
        grid_path = os.path.join(args.recon_dir, f"recon_epoch_{epoch+1:03d}.png")
        save_reconstruction_grid(model, val_loader, device, grid_path, num_images=8)
        print(f"Saved reconstruction grid to {grid_path}")

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

        # Save best
        val_total = val_losses["total"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            best_path = os.path.join(args.checkpoint_dir, "vqvae_best.pt")
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)
            print(f"New best model! val_loss={val_total:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {args.patience} epochs)")
            break

        print()

    writer.close()
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
    parser.add_argument("--num_workers", type=int, default=4,
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
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Linear warmup steps")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")

    # Logging
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log to tensorboard every N steps")
    parser.add_argument("--log_dir", type=str, default="runs/vqvae",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--recon_dir", type=str, default="reconstructions",
                        help="Reconstruction grid save directory")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
