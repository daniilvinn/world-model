#!/usr/bin/env python3
"""
Quick Start Script - World Model Project

This script guides you through the complete workflow:
1. Data collection
2. VQ-VAE training
3. Latent precomputation
4. Dynamics model training
5. Playing the AI-driven game

Run this to see what you need to do next based on your current progress.
"""

import os
import sys
from pathlib import Path


def check_exists(path, description):
    """Check if a file or directory exists."""
    exists = os.path.exists(path)
    status = "[OK]" if exists else "[--]"
    print(f"  {status} {description}: {path}")
    return exists


def main():
    print("="*70)
    print("World Model Project - Quick Start")
    print("="*70)
    print()
    
    # Check environment
    print("1. Checking Environment...")
    has_dataset_dir = check_exists("dataset", "Dataset directory")
    has_vqvae_checkpoint = check_exists("checkpoints/vqvae_best.pt", "VQ-VAE checkpoint")
    has_latents_dir = check_exists("latents", "Latents directory")
    has_seeds = check_exists("latents/seeds.pt", "Seed sequences")
    has_dynamics_checkpoint = check_exists("checkpoints/dynamics_best.pt", "Dynamics checkpoint")
    
    print()
    
    # Determine next step
    if not has_dataset_dir or not any(Path("dataset").glob("session_*")):
        print(">> NEXT STEP: Collect Dataset")
        print("-" * 70)
        print("You need to collect training data first.")
        print()
        print("Run:")
        print("  python collect_dataset.py")
        print()
        print("This will collect 100K frame pairs in ~5-6 minutes.")
        print("The game runs in auto-play mode at maximum speed (headless).")
        print()
        return
    
    if not has_vqvae_checkpoint:
        print(">> NEXT STEP: Train VQ-VAE")
        print("-" * 70)
        print("Train the image compression model (VQ-VAE).")
        print()
        print("Run:")
        print("  python -m vqvae.train --epochs 50 --batch_size 6")
        print()
        print("This will take ~2-3 hours on RTX 5070 Ti.")
        print("You can monitor progress with TensorBoard:")
        print("  tensorboard --logdir runs/")
        print()
        return
    
    if not has_latents_dir or not has_seeds:
        print(">> NEXT STEP: Precompute Latents")
        print("-" * 70)
        print("Encode all frames to latent space for efficient training.")
        print()
        print("Run:")
        print("  python precompute_latents.py --batch_size 64")
        print()
        print("This will take ~10-20 minutes depending on dataset size.")
        print("Creates:")
        print("  - latents/session_*.npz (encoded frames)")
        print("  - latents/seeds.pt (20 seed sequences for AI game)")
        print()
        return
    
    if not has_dynamics_checkpoint:
        print(">> NEXT STEP: Train Dynamics Model")
        print("-" * 70)
        print("Train the world model (flow matching with diffusion forcing).")
        print()
        print("Run:")
        print("  python -m dynamics.train --epochs 100 --batch_size 64")
        print()
        print("This will take ~4-6 hours on RTX 5070 Ti.")
        print("You can monitor progress with TensorBoard:")
        print("  tensorboard --logdir runs/")
        print()
        print("Optional: Adjust hyperparameters")
        print("  --lr 3e-4          Learning rate")
        print("  --batch_size 64    Batch size (reduce if OOM)")
        print("  --ode_steps 10     ODE steps during validation")
        print()
        return
    
    # All checkpoints exist!
    print(">> ALL CHECKPOINTS EXIST - READY TO PLAY!")
    print("-" * 70)
    print("Everything is set up. You can now play the AI-driven game!")
    print()
    print("Run:")
    print("  python play_world_model.py")
    print()
    print("Controls:")
    print("  SPACE - Jump")
    print("  R     - Restart (new seed)")
    print("  ESC   - Quit")
    print()
    print("Advanced options:")
    print("  --ode_steps 15           Use more ODE steps (better quality, slower)")
    print("  --solver midpoint        Use midpoint solver (2x slower, more accurate)")
    print("  --seed_index 5           Start with specific seed")
    print("  --no_compile             Disable torch.compile")
    print()
    print("-" * 70)
    print()
    print("Optional: Run evaluation")
    print("  python -m vqvae.evaluate --full_res")
    print()
    print("Optional: Test original game with VQ-VAE")
    print("  python game.py --vqvae")
    print()
    print("="*70)
    print("For detailed documentation, see README.md and TRAIN.MD")
    print("="*70)


if __name__ == "__main__":
    main()
