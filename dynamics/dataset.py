"""
Dataset for loading precomputed latent sequences for dynamics model training.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class LatentSequenceDataset(Dataset):
    """
    PyTorch Dataset that loads precomputed latent indices and actions.
    
    Returns context windows of consecutive frames from the same episode.
    
    Args:
        latents_dir: Directory containing session_*.npz files with precomputed latents
        context_length: Number of context frames (default 4)
        rollout_length: Number of target frames to predict (default 1)
        codebook: VQ-VAE codebook tensor [num_embeddings, embedding_dim] for index->z_q lookup
    """
    
    def __init__(self, latents_dir="latents", context_length=4, rollout_length=1, codebook=None):
        super().__init__()
        self.context_length = context_length
        self.rollout_length = rollout_length
        self.codebook = codebook
        
        if codebook is None:
            raise ValueError("codebook must be provided (vqvae.quantizer.embedding)")
        
        # Load all session files
        session_files = sorted(glob.glob(os.path.join(latents_dir, "session_*.npz")))
        
        if len(session_files) == 0:
            raise FileNotFoundError(
                f"No session files found in {latents_dir}\n"
                f"Run precompute_latents.py first to generate latents."
            )
        
        print(f"Loading {len(session_files)} session file(s) from {latents_dir}")
        
        # Load all sessions
        self.sessions = []
        for session_file in session_files:
            data = np.load(session_file)
            self.sessions.append({
                'indices': data['indices'],  # [N_frames, 32, 32] int16
                'actions': data['actions'],  # [N_frames - 1 or more] int8
                'episode_ids': data['episode_ids'],  # [N_frames] int32
            })
        
        # Build list of valid windows
        # Each window is (session_idx, start_frame_idx) where start_frame_idx is
        # the first frame of a context_length + rollout_length consecutive sequence within same episode
        self.windows = []
        
        for session_idx, session in enumerate(self.sessions):
            indices = session['indices']
            episode_ids = session['episode_ids']
            
            # Scan for valid windows
            # Need context_length + rollout_length consecutive frames in same episode
            total_frames_needed = context_length + rollout_length
            for start_idx in range(len(indices) - total_frames_needed + 1):
                # Check if all required frames are in the same episode
                episode_start = episode_ids[start_idx]
                episode_end = episode_ids[start_idx + total_frames_needed - 1]
                
                if episode_start == episode_end:
                    self.windows.append((session_idx, start_idx))
        
        print(f"Found {len(self.windows)} valid context windows "
              f"(context_length={context_length}, rollout_length={rollout_length})")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Returns:
            context_zq: [context_length, 16, 32, 32] float32 - context frames as continuous latents
            targets_zq: [rollout_length, 16, 32, 32] float32 - target frames as continuous latents
            actions: [rollout_length] int - actions for each transition
        """
        session_idx, start_idx = self.windows[idx]
        session = self.sessions[session_idx]
        
        # Extract indices for context and targets
        # context: frames at positions [start_idx, ..., start_idx + context_length - 1]
        # targets: frames at positions [start_idx + context_length, ..., start_idx + context_length + rollout_length - 1]
        context_indices = session['indices'][start_idx:start_idx + self.context_length]  # [ctx_len, 32, 32]
        target_indices = session['indices'][start_idx + self.context_length:start_idx + self.context_length + self.rollout_length]  # [rollout_len, 32, 32]
        
        # Get actions for each transition
        # action at position i means the action between frame i and frame i+1
        # We need actions from (last context frame -> first target), ..., (second-to-last target -> last target)
        action_start_idx = start_idx + self.context_length - 1
        actions = session['actions'][action_start_idx:action_start_idx + self.rollout_length]  # [rollout_len]
        
        # Convert indices to continuous latents via codebook lookup
        context_indices_tensor = torch.from_numpy(context_indices).long()  # [ctx_len, 32, 32]
        target_indices_tensor = torch.from_numpy(target_indices).long()  # [rollout_len, 32, 32]
        
        # Codebook lookup: [ctx_len, 32, 32] -> [ctx_len, 32, 32, 16]
        context_zq = self.codebook[context_indices_tensor]  # [ctx_len, 32, 32, 16]
        context_zq = context_zq.permute(0, 3, 1, 2)  # [ctx_len, 16, 32, 32]
        
        # Codebook lookup: [rollout_len, 32, 32] -> [rollout_len, 32, 32, 16]
        targets_zq = self.codebook[target_indices_tensor]  # [rollout_len, 32, 32, 16]
        targets_zq = targets_zq.permute(0, 3, 1, 2)  # [rollout_len, 16, 32, 32]
        
        # Convert actions to tensor
        actions_tensor = torch.from_numpy(actions).long()  # [rollout_len]
        
        return context_zq.float(), targets_zq.float(), actions_tensor


def create_dataloaders(
    latents_dir="latents",
    vqvae_checkpoint=None,
    context_length=4,
    rollout_length=1,
    batch_size=64,
    val_split=0.1,
    num_workers=4,
    seed=42,
):
    """
    Create train and validation DataLoaders.
    
    Args:
        latents_dir: Directory containing precomputed latents
        vqvae_checkpoint: Path to VQ-VAE checkpoint (to extract codebook)
        context_length: Number of context frames
        rollout_length: Number of target frames to predict (for rollout training)
        batch_size: Batch size
        val_split: Fraction of data for validation
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducible split
    
    Returns:
        (train_loader, val_loader, codebook)
    """
    # Load VQ-VAE codebook
    if vqvae_checkpoint is None:
        raise ValueError("vqvae_checkpoint must be provided")
    
    print(f"Loading VQ-VAE codebook from {vqvae_checkpoint}...")
    from vqvae.model import VQVAE
    
    ckpt = torch.load(vqvae_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    
    vqvae = VQVAE(
        latent_dim=config.get("latent_dim", 16),
        num_embeddings=config.get("num_embeddings", 1024),
        commitment_cost=config.get("commitment_cost", 0.25),
        ema_decay=config.get("ema_decay", 0.99),
    )
    vqvae.load_state_dict(ckpt["model_state_dict"])
    
    codebook = vqvae.quantizer.embedding  # [num_embeddings, embedding_dim]
    print(f"Codebook shape: {codebook.shape}")
    
    # Create dataset
    dataset = LatentSequenceDataset(
        latents_dir=latents_dir,
        context_length=context_length,
        rollout_length=rollout_length,
        codebook=codebook,
    )
    
    # Split into train/val
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    print(f"Split: {train_size} train, {val_size} val")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    
    return train_loader, val_loader, codebook
