"""
FID / KID computation helpers using torch-fidelity.

Wraps numpy arrays into a torch-fidelity-compatible dataset so that
the metrics can be computed without writing images to disk.
"""

from __future__ import annotations

from typing import List, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import Dataset

T = TypeVar("T")


class _NumpyImageDataset(Dataset):
    """Dataset wrapper that yields uint8 HWC images as CHW tensors."""

    def __init__(self, images: List[np.ndarray]) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        return torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW


def ensure_min_samples(items: Sequence[T], min_samples: int) -> List[T]:
    if min_samples <= 0 or len(items) >= min_samples or len(items) == 0:
        return list(items)
    return [items[i % len(items)] for i in range(min_samples)]


def fid_from_arrays(
    real: List[np.ndarray],
    fake: List[np.ndarray],
) -> float:
    """Compute FID from two lists of uint8 [H, W, 3] numpy arrays."""
    try:
        import torch_fidelity
    except ImportError:
        raise ImportError(
            "torch-fidelity is required for FID computation. "
            "Install it with: pip install torch-fidelity"
        )

    metrics = torch_fidelity.calculate_metrics(
        input1=_NumpyImageDataset(fake),
        input2=_NumpyImageDataset(real),
        cuda=torch.cuda.is_available(),
        fid=True,
        kid=False,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])
