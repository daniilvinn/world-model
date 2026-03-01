"""
GameFrameDataset for loading game screenshots from collected .npz frame pairs.

Loads both frame_t0 and frame_t1 from each pair as independent samples,
resizes from native 420x840 to 256x512 (preserving 2:1 aspect ratio),
and normalizes to [-1, 1].
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


class GameFrameDataset(Dataset):
    """
    PyTorch Dataset that loads game frames from .npz pair files.

    Each .npz contains frame_t0 and frame_t1 (both uint8, shape 420x840x3).
    Both frames are exposed as independent samples, doubling the effective
    dataset size.

    Preprocessing:
        1. Resize from 420x840 to 256x512 (bilinear, preserves 2:1 aspect ratio)
        2. Convert to float32 and normalize to [-1, 1]

    Output tensor shape: [3, 256, 512]
    """

    def __init__(self, data_dir="dataset", target_size=(256, 512)):
        """
        Args:
            data_dir: Root directory containing session_* subdirectories with .npz files.
            target_size: (H, W) tuple for resize. Default (256, 512) preserves 2:1 aspect ratio.
        """
        super().__init__()
        self.target_size = target_size

        # Scan all .npz files across all sessions
        pattern = os.path.join(data_dir, "session_*", "pair_*.npz")
        self.npz_files = sorted(glob.glob(pattern))

        if len(self.npz_files) == 0:
            raise FileNotFoundError(
                f"No .npz files found matching pattern: {pattern}\n"
                f"Make sure you have collected data using collect_dataset.py"
            )

        # Each .npz has 2 frames (frame_t0, frame_t1), so total samples = 2 * num_files
        self.num_files = len(self.npz_files)
        self.num_samples = self.num_files * 2

        print(f"GameFrameDataset: found {self.num_files} .npz files "
              f"({self.num_samples} frames) in {data_dir}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Map linear index to (file_index, frame_index)
        file_idx = idx // 2
        frame_idx = idx % 2  # 0 = frame_t0, 1 = frame_t1

        # Load the .npz file
        data = np.load(self.npz_files[file_idx])
        frame_key = "frame_t0" if frame_idx == 0 else "frame_t1"
        frame = data[frame_key]  # uint8, shape (420, 840, 3)

        # Convert to torch tensor: (H, W, C) -> (C, H, W)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # [3, 420, 840]

        # Resize from 420x840 to 256x512 (preserves 2:1 aspect ratio)
        frame = TF.resize(
            frame,
            size=list(self.target_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        # Normalize to [-1, 1]
        frame = frame / 127.5 - 1.0

        return frame


def create_dataloaders(
    data_dir="dataset",
    batch_size=16,
    val_split=0.1,
    num_workers=4,
    seed=42,
    max_samples=None,
):
    """
    Create train and validation DataLoaders with a 90/10 split.

    Args:
        data_dir: Root directory containing session_* subdirectories.
        batch_size: Batch size for both train and val loaders.
        val_split: Fraction of data to use for validation.
        num_workers: Number of DataLoader worker processes.
        seed: Random seed for reproducible split.
        max_samples: Maximum number of samples to use (None = use all).

    Returns:
        (train_loader, val_loader)
    """
    dataset = GameFrameDataset(data_dir=data_dir)

    # Limit dataset size if requested
    if max_samples is not None and max_samples < len(dataset):
        print(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
        indices = list(range(len(dataset)))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    # Compute split sizes
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    # For tiny smoke-test datasets (e.g. max_samples=4), int(total*val_split)
    # can become 0 and break validation logging downstream. Keep at least one
    # sample in each split whenever the dataset has at least 2 samples.
    if total >= 2:
        if val_size == 0 and val_split > 0:
            val_size = 1
            train_size = total - val_size
        elif train_size == 0 and val_size > 0:
            train_size = 1
            val_size = total - train_size

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"Split: {train_size} train, {val_size} val")

    train_drop_last = len(train_dataset) >= batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=train_drop_last,
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

    return train_loader, val_loader
