"""
Failure / collapse detection metrics.

Covers: collapse rate at various horizons, average time-to-collapse,
and flat-world detection (a known failure mode in Neural Dash).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from logger.metric_names import M


def _infer_decoder_device(vqvae_decoder: Callable, fallback: torch.device) -> torch.device:
    """
    Resolve decoder device for both module-style and bound-method decoders.
    """
    if hasattr(vqvae_decoder, "parameters"):
        try:
            return next(vqvae_decoder.parameters()).device
        except (StopIteration, TypeError):
            return fallback
    owner = getattr(vqvae_decoder, "__self__", None)
    if owner is not None and hasattr(owner, "parameters"):
        try:
            return next(owner.parameters()).device
        except (StopIteration, TypeError):
            return fallback
    return fallback


@torch.no_grad()
def detect_collapse(
    frames: torch.Tensor,
    static_threshold: float = 0.01,
    blank_std_threshold: float = 0.02,
    flat_row_ratio: float = 0.8,
) -> bool:
    """
    Determine whether a frame has collapsed.

    A frame is "collapsed" if any of:
      - Nearly static: mean abs diff from previous frame < ``static_threshold``
      - Nearly blank: pixel std < ``blank_std_threshold``
      - Flat world: fraction of nearly-identical rows > ``flat_row_ratio``
    """
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)

    frame = frames[-1]
    if frame.std() < blank_std_threshold:
        return True

    if frames.shape[0] >= 2:
        diff = (frames[-1] - frames[-2]).abs().mean()
        if diff < static_threshold:
            return True

    row_means = frame.mean(dim=(0, 2))
    row_diffs = (row_means[1:] - row_means[:-1]).abs()
    flat_fraction = (row_diffs < 0.01).float().mean().item()
    if flat_fraction > flat_row_ratio:
        return True

    return False


@torch.no_grad()
def compute_collapse_rate(
    rollout_batch: List[torch.Tensor],
    horizon: int,
    static_threshold: float = 0.01,
    blank_std_threshold: float = 0.02,
    flat_row_ratio: float = 0.8,
) -> float:
    """
    Fraction of rollouts that have collapsed by step ``horizon``.

    Args:
        rollout_batch: List of [T, C, H, W] rollout tensors.
        horizon: Step at which to check.
    """
    if not rollout_batch:
        return 0.0

    collapsed = 0
    for rollout in rollout_batch:
        T = rollout.shape[0]
        h = min(horizon, T)
        if detect_collapse(
            rollout[max(0, h - 2) : h],
            static_threshold, blank_std_threshold, flat_row_ratio,
        ):
            collapsed += 1

    return collapsed / len(rollout_batch)


@torch.no_grad()
def compute_time_to_collapse(
    rollout: torch.Tensor,
    static_threshold: float = 0.01,
    blank_std_threshold: float = 0.02,
    flat_row_ratio: float = 0.8,
) -> int:
    """
    Number of steps until collapse is first detected.

    Returns ``rollout.shape[0]`` if no collapse is detected.
    """
    T = rollout.shape[0]
    for t in range(1, T):
        window = rollout[max(0, t - 1) : t + 1]
        if detect_collapse(window, static_threshold, blank_std_threshold, flat_row_ratio):
            return t
    return T


@torch.no_grad()
def evaluate_failure_metrics(
    rollout_batch: List[torch.Tensor],
    horizons: Sequence[int],
    vqvae_decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Full failure / collapse analysis.

    Args:
        rollout_batch: List of [T, C, H, W] rollout tensors (latent or RGB).
        horizons: H values to evaluate collapse rate at.
        vqvae_decoder: If provided, decode latents to RGB for analysis.
    """
    results: Dict[str, float] = {}

    decoded_batch = rollout_batch
    if vqvae_decoder is not None:
        decoded_batch = []
        fallback_device = rollout_batch[0].device if rollout_batch else torch.device("cpu")
        decoder_device = _infer_decoder_device(vqvae_decoder, fallback_device)
        for rollout in rollout_batch:
            T = rollout.shape[0]
            decoded_frames = torch.cat(
                [vqvae_decoder(rollout[i:i+1].to(decoder_device)) for i in range(T)], dim=0
            )
            decoded_batch.append(((decoded_frames + 1) / 2).cpu())

    for H in horizons:
        results[M.collapse_rate(H)] = compute_collapse_rate(decoded_batch, H)

    ttc_values = []
    for rollout in decoded_batch:
        ttc_values.append(compute_time_to_collapse(rollout))

    results[M.avg_time_to_collapse()] = float(np.mean(ttc_values)) if ttc_values else 0.0

    return results
