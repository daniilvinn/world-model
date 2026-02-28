"""
FVD computation with I3D features.

Uses a lightweight I3D feature extractor for Frechet Video Distance.
Falls back to frame-level Inception features averaged over time when
the full I3D model is unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


def get_fvd_features(
    videos: torch.Tensor,
    detector: Optional[torch.nn.Module] = None,
) -> np.ndarray:
    """
    Extract per-video feature vectors for FVD computation.

    Args:
        videos: [N, T, C, H, W] video clips in [0, 1].
        detector: Optional pre-loaded I3D model.  When *None*, uses
            frame-level Inception-v3 features averaged across time.

    Returns:
        [N, D] feature matrix as numpy array.
    """
    if detector is not None:
        return _extract_i3d(videos, detector)
    return _extract_inception_mean(videos)


def _extract_inception_mean(videos: torch.Tensor) -> np.ndarray:
    """Fallback: extract InceptionV3 features per frame, mean-pool over time."""
    try:
        from torchvision.models import inception_v3, Inception_V3_Weights
    except ImportError:
        raise ImportError("torchvision is required for FVD feature extraction.")

    device = videos.device
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier head with identity to get features
    model.fc = torch.nn.Identity()

    N, T, C, H, W = videos.shape
    all_features = []

    for i in range(N):
        frames = videos[i]  # [T, C, H, W]
        # Inception expects 299x299
        frames_resized = F.interpolate(frames, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            feats = model(frames_resized)  # [T, D]
        all_features.append(feats.mean(dim=0).cpu().numpy())

    return np.stack(all_features)


def _extract_i3d(videos: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """Extract features using a provided I3D model."""
    device = videos.device
    model = model.to(device).eval()
    N = videos.shape[0]
    all_features = []

    for i in range(N):
        clip = videos[i].unsqueeze(0)  # [1, T, C, H, W]
        clip = clip.permute(0, 2, 1, 3, 4)  # [1, C, T, H, W] for I3D
        with torch.no_grad():
            feat = model(clip)
        if feat.dim() > 2:
            feat = feat.flatten(1)
        all_features.append(feat[0].cpu().numpy())

    return np.stack(all_features)


def frechet_distance(feats1: np.ndarray, feats2: np.ndarray) -> float:
    """
    Compute Frechet Distance between two Gaussian-fitted feature sets.
    """
    mu1, sigma1 = feats1.mean(axis=0), np.cov(feats1, rowvar=False)
    mu2, sigma2 = feats2.mean(axis=0), np.cov(feats2, rowvar=False)

    diff = mu1 - mu2

    from scipy.linalg import sqrtm
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fd)
