"""
Temporal stability metrics.

Covers: sequence LPIPS flicker (mean/std), optical flow EPE between GT
and inferred flow, and motion magnitude correlation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

from evaluation.single_frame_metrics import compute_lpips
from logger.metric_names import M


@torch.no_grad()
def compute_flicker(
    frames: torch.Tensor,
    vqvae_decoder: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """
    Measure temporal flicker via LPIPS between consecutive frames.

    Args:
        frames: [T, C, H, W] sequence of frames (latent or RGB).
        vqvae_decoder: If provided, decode latents to RGB before computing LPIPS.

    Returns:
        Dict with flicker mean and std.
    """
    T = frames.shape[0]
    if T < 2:
        return {M.flicker("Mean"): 0.0, M.flicker("Std"): 0.0}

    dists = []
    for i in range(T - 1):
        f1 = frames[i : i + 1]
        f2 = frames[i + 1 : i + 2]
        if vqvae_decoder is not None:
            f1 = vqvae_decoder(f1)
            f2 = vqvae_decoder(f2)
        dists.append(compute_lpips(f1, f2))

    arr = np.array(dists)
    return {
        M.flicker("Mean"): float(arr.mean()),
        M.flicker("Std"): float(arr.std()),
    }


@torch.no_grad()
def compute_optical_flow_epe(
    real_frames: torch.Tensor,
    generated_frames: torch.Tensor,
) -> float:
    """
    Average End-Point Error between optical flow of real and generated sequences.

    Uses ``cv2.calcOpticalFlowFarneback`` (grayscale Farneback flow) for a
    lightweight, dependency-minimal estimate.

    Args:
        real_frames: [T, 3, H, W] real RGB frames in [0, 1].
        generated_frames: [T, 3, H, W] generated RGB frames in [0, 1].

    Returns:
        Mean EPE across frame pairs.
    """
    try:
        import cv2
    except ImportError:
        return float("nan")

    T = real_frames.shape[0]
    if T < 2:
        return 0.0

    def _to_gray_np(t: torch.Tensor) -> np.ndarray:
        rgb = t.cpu().permute(1, 2, 0).numpy()
        rgb_u8 = (rgb * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)

    def _flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        return cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

    epes = []
    for i in range(T - 1):
        flow_real = _flow(_to_gray_np(real_frames[i]), _to_gray_np(real_frames[i + 1]))
        flow_gen = _flow(_to_gray_np(generated_frames[i]), _to_gray_np(generated_frames[i + 1]))
        diff = flow_real - flow_gen
        epe = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2).mean()
        epes.append(float(epe))

    return float(np.mean(epes))


@torch.no_grad()
def compute_motion_magnitude_correlation(
    real_frames: torch.Tensor,
    generated_frames: torch.Tensor,
) -> float:
    """
    Pearson correlation between per-frame motion magnitudes
    of real and generated sequences.

    Motion magnitude is approximated as the mean absolute pixel difference
    between consecutive frames (cheap, no OpenCV required).
    """
    T = real_frames.shape[0]
    if T < 2:
        return 0.0

    def _motion_magnitudes(frames: torch.Tensor) -> np.ndarray:
        diffs = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
        return diffs.cpu().numpy()

    real_mag = _motion_magnitudes(real_frames)
    gen_mag = _motion_magnitudes(generated_frames)

    if real_mag.std() < 1e-8 or gen_mag.std() < 1e-8:
        return 0.0

    return float(np.corrcoef(real_mag, gen_mag)[0, 1])


@torch.no_grad()
def evaluate_temporal(
    real_frames: torch.Tensor,
    generated_frames: torch.Tensor,
    vqvae_decoder: Optional[torch.nn.Module] = None,
    compute_flow: bool = True,
) -> Dict[str, float]:
    """
    Full temporal stability evaluation.

    Both inputs should be [T, C, H, W].  If they are latent tensors,
    pass ``vqvae_decoder`` to decode to RGB for LPIPS and flow metrics.
    """
    results: Dict[str, float] = {}

    results.update(compute_flicker(generated_frames, vqvae_decoder))

    if vqvae_decoder is not None:
        decoded_real = torch.cat([vqvae_decoder(real_frames[i:i+1]) for i in range(real_frames.shape[0])], dim=0)
        decoded_gen = torch.cat([vqvae_decoder(generated_frames[i:i+1]) for i in range(generated_frames.shape[0])], dim=0)
        rgb_real = (decoded_real + 1) / 2
        rgb_gen = (decoded_gen + 1) / 2
    else:
        rgb_real = (real_frames + 1) / 2
        rgb_gen = (generated_frames + 1) / 2

    results[M.motion_magnitude_corr()] = compute_motion_magnitude_correlation(rgb_real, rgb_gen)

    if compute_flow:
        results[M.optical_flow_epe()] = compute_optical_flow_epe(rgb_real, rgb_gen)

    return results
