"""
Video-level distribution metrics.

Covers Frechet Video Distance (FVD) at various clip lengths (16, 32, 64,
128, 256, 512) and optional JEDi / enhanced video quality hooks.

FVD uses an I3D backbone pretrained on Kinetics-400.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from logger.metric_names import M


@torch.no_grad()
def compute_fvd(
    real_videos: torch.Tensor,
    fake_videos: torch.Tensor,
    device: torch.device = torch.device("cuda"),
) -> float:
    """
    Frechet Video Distance between two sets of video clips.

    Args:
        real_videos: [N, T, C, H, W] real video clips in [0, 1].
        fake_videos: [N, T, C, H, W] generated video clips in [0, 1].
        device: Compute device.

    Returns:
        Scalar FVD value.
    """
    from evaluation._fvd_i3d import get_fvd_features, frechet_distance

    real_feats = get_fvd_features(real_videos.to(device))
    fake_feats = get_fvd_features(fake_videos.to(device))

    return frechet_distance(real_feats, fake_feats)


@torch.no_grad()
def evaluate_fvd_at_clip_lengths(
    real_videos: torch.Tensor,
    fake_videos: torch.Tensor,
    clip_lengths: List[int],
    device: torch.device = torch.device("cuda"),
) -> Dict[str, float]:
    """
    Compute FVD at multiple clip lengths by truncating.

    Args:
        real_videos: [N, T_max, C, H, W] real videos.
        fake_videos: [N, T_max, C, H, W] generated videos.
        clip_lengths: List of clip lengths to evaluate.

    Returns:
        Dict of named FVD metrics.
    """
    T_max = real_videos.shape[1]
    results: Dict[str, float] = {}

    for length in clip_lengths:
        if length > T_max:
            continue
        real_clip = real_videos[:, :length]
        fake_clip = fake_videos[:, :length]
        results[M.fvd(length)] = compute_fvd(real_clip, fake_clip, device)

    return results
