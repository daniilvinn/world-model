"""
Precompute latents from collected dataset using trained VQ-VAE.

This script:
1. Loads a trained VQ-VAE model
2. Processes all .npz frame pairs from dataset/
3. Encodes frames to codebook indices [32, 32]
4. Saves latents + actions + episode_ids to latents/ directory
5. Extracts and saves 20 seed sequences for play_world_model.py

Usage:
    python precompute_latents.py
    python precompute_latents.py --data_dir dataset --checkpoint checkpoints/vqvae_best.pt --batch_size 64
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from vqvae.model import VQVAE


def preprocess_frame(frame):
    """
    Preprocess frame for VQ-VAE encoding (must match vqvae/dataset.py exactly).
    
    Args:
        frame: numpy array (420, 840, 3) uint8
    
    Returns:
        tensor [3, 256, 512] float32 in [-1, 1]
    """
    # Convert to torch tensor: (H, W, C) -> (C, H, W)
    frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # [3, 420, 840]
    
    # Resize from 420x840 to 256x512 (preserves 2:1 aspect ratio)
    frame = TF.resize(
        frame,
        size=[256, 512],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    
    # Normalize to [-1, 1]
    frame = frame / 127.5 - 1.0
    
    return frame


def load_vqvae(checkpoint_path, device):
    """Load trained VQ-VAE model from checkpoint."""
    print(f"Loading VQ-VAE from {checkpoint_path}...")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    
    # Create model
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
    print(f"Loaded VQ-VAE from epoch {epoch}, val_loss={val_loss}")
    
    return model


def process_session(session_dir, vqvae_model, device, batch_size=64):
    """
    Process all pairs in a session directory.
    
    Returns:
        dict with keys:
            - indices: [N_frames, 32, 32] int16
            - actions: [N_frames - 1] int8 (action between frame i and frame i+1)
            - episode_ids: [N_frames] int32
    """
    # Scan all pair files
    pair_files = sorted(glob.glob(os.path.join(session_dir, "pair_*.npz")))
    
    if len(pair_files) == 0:
        return None
    
    print(f"\nProcessing {len(pair_files)} pairs from {os.path.basename(session_dir)}")
    
    # Load all pairs and group by episode
    episode_data = defaultdict(list)  # episode_id -> list of (pair_idx, data)
    
    for pair_file in pair_files:
        with np.load(pair_file) as data:
            episode_id = int(data['episode_id'])
            # Extract the data we need before exiting the context
            frame_t0 = data['frame_t0'].copy()
            frame_t1 = data['frame_t1'].copy()
            action = int(data['action'])
            episode_data[episode_id].append({
                'frame_t0': frame_t0,
                'frame_t1': frame_t1,
                'action': action
            })
    
    # Process each episode separately
    all_indices = []
    all_actions = []
    all_episode_ids = []
    
    for episode_id in sorted(episode_data.keys()):
        pairs = episode_data[episode_id]
        
        # Extract unique frames from episode
        # frame_t0 from first pair, then frame_t1 from all pairs
        frames = []
        actions = []
        
        # Add first frame
        first_data = pairs[0]
        frames.append(first_data['frame_t0'])
        
        # Add all subsequent frames and actions
        for data in pairs:
            frames.append(data['frame_t1'])
            actions.append(data['action'])
        
        # Encode frames in batches
        frame_indices = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = torch.stack([preprocess_frame(f) for f in batch_frames])
            batch_tensors = batch_tensors.to(device)
            
            # Encode
            with torch.no_grad():
                _, indices = vqvae_model.encode(batch_tensors)  # [B, 32, 32]
            
            frame_indices.append(indices.cpu().numpy().astype(np.int16))
        
        # Concatenate
        frame_indices = np.concatenate(frame_indices, axis=0)  # [N_frames, 32, 32]
        actions_array = np.array(actions, dtype=np.int8)  # [N_frames - 1]
        episode_ids_array = np.full(len(frames), episode_id, dtype=np.int32)
        
        # Append to session data
        all_indices.append(frame_indices)
        all_actions.append(actions_array)
        all_episode_ids.append(episode_ids_array)
        
        # Add -1 separator after each episode (except last)
        if episode_id != max(episode_data.keys()):
            all_actions.append(np.array([-1], dtype=np.int8))
    
    # Concatenate all episodes
    all_indices = np.concatenate(all_indices, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_episode_ids = np.concatenate(all_episode_ids, axis=0)
    
    return {
        'indices': all_indices,
        'actions': all_actions,
        'episode_ids': all_episode_ids,
    }


def extract_seed_sequences(latents_dir, vqvae_model, num_seeds=20, context_length=4):
    """
    Extract seed context sequences from the beginning of episodes.
    
    Args:
        latents_dir: directory containing session_*.npz files
        vqvae_model: VQ-VAE model (for codebook lookup)
        num_seeds: number of seeds to extract
        context_length: number of frames per seed
    
    Returns:
        dict with key 'contexts': tensor [num_seeds, context_length, 16, 32, 32]
    """
    print(f"\nExtracting {num_seeds} seed sequences...")
    
    # Load all session files
    session_files = sorted(glob.glob(os.path.join(latents_dir, "session_*.npz")))
    
    codebook = vqvae_model.quantizer.embedding  # [1024, 16]
    
    seeds = []
    
    for session_file in session_files:
        with np.load(session_file) as data:
            indices = data['indices'][:].copy()  # [N_frames, 32, 32]
            episode_ids = data['episode_ids'][:].copy()  # [N_frames]
        
        # Find episode boundaries (where episode_id changes)
        episode_starts = [0]
        for i in range(1, len(episode_ids)):
            if episode_ids[i] != episode_ids[i - 1]:
                episode_starts.append(i)
        
        # Extract context_length frames from each episode start
        for start_idx in episode_starts:
            if start_idx + context_length <= len(indices):
                # Get indices for seed
                seed_indices = indices[start_idx:start_idx + context_length]  # [4, 32, 32]
                
                # Convert to continuous latents
                seed_indices_tensor = torch.from_numpy(seed_indices).long()
                seed_zq = codebook[seed_indices_tensor]  # [4, 32, 32, 16]
                seed_zq = seed_zq.permute(0, 3, 1, 2)  # [4, 16, 32, 32]
                
                seeds.append(seed_zq)
                
                if len(seeds) >= num_seeds:
                    break
        
        if len(seeds) >= num_seeds:
            break
    
    if len(seeds) < num_seeds:
        print(f"Warning: Only found {len(seeds)} valid seed sequences (requested {num_seeds})")
    
    # Stack and return
    seeds_tensor = torch.stack(seeds[:num_seeds], dim=0)  # [num_seeds, 4, 16, 32, 32]
    
    print(f"Extracted {len(seeds[:num_seeds])} seed sequences, shape: {seeds_tensor.shape}")
    
    return {'contexts': seeds_tensor}


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load VQ-VAE
    vqvae_model = load_vqvae(args.checkpoint, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all session directories
    session_dirs = sorted(glob.glob(os.path.join(args.data_dir, "session_*")))
    
    if len(session_dirs) == 0:
        print(f"No session directories found in {args.data_dir}")
        return
    
    print(f"\nFound {len(session_dirs)} session(s)")
    
    # Process each session
    for session_dir in tqdm(session_dirs, desc="Processing sessions"):
        session_name = os.path.basename(session_dir)
        output_path = os.path.join(args.output_dir, f"{session_name}.npz")
        
        # Skip if already processed
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {session_name} (already exists)")
            continue
        
        # Process
        result = process_session(session_dir, vqvae_model, device, args.batch_size)
        
        if result is None:
            print(f"Warning: No pairs found in {session_name}")
            continue
        
        # Save
        np.savez_compressed(output_path, **result)
        
        print(f"Saved {session_name}.npz: {result['indices'].shape[0]} frames, "
              f"{result['actions'].shape[0]} actions")
    
    # Extract and save seed sequences
    seeds_path = os.path.join(args.output_dir, "seeds.pt")
    seeds = extract_seed_sequences(args.output_dir, vqvae_model, args.num_seeds, args.context_length)
    torch.save(seeds, seeds_path)
    print(f"\nSaved {args.num_seeds} seed sequences to {seeds_path}")
    
    print(f"\n{'='*60}")
    print("Latent precomputation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute latents from dataset")
    
    parser.add_argument("--data_dir", type=str, default="dataset",
                        help="Directory containing session_*/pair_*.npz files")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/vqvae_best.pt",
                        help="Path to VQ-VAE checkpoint")
    parser.add_argument("--output_dir", type=str, default="latents",
                        help="Output directory for latents")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--num_seeds", type=int, default=20,
                        help="Number of seed sequences to extract")
    parser.add_argument("--context_length", type=int, default=4,
                        help="Number of frames per seed sequence")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing latent files")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
