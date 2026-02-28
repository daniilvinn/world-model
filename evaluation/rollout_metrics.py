"""
Rollout evaluation metrics.

Short-horizon (4-64 steps) and long-horizon (64-512 steps) quality
assessment: per-step PSNR/SSIM/LPIPS, mean-over-window aggregates,
NLL of GT trajectory, per-step token entropy, and codebook perplexity.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from evaluation.single_frame_metrics import compute_lpips, compute_psnr, compute_ssim
from logger.metric_names import M


@torch.no_grad()
def evaluate_rollout_at_horizons(
    generated_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    horizons: Sequence[int],
    vqvae_decoder: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """
    Per-step and mean-over-window quality metrics for a generated rollout.

    Args:
        generated_frames: [T, C, H, W] generated latent (or RGB) frames.
        gt_frames: [T, C, H, W] ground-truth counterpart (same space).
        horizons: List of step indices to evaluate at.
        vqvae_decoder: If provided, decode latents to RGB before metric
            computation (both tensors must be latents in this case).

    Returns:
        Dict of named metrics for ``wandb.log``.
    """
    T = generated_frames.shape[0]
    results: Dict[str, float] = {}

    decode = vqvae_decoder is not None

    for h in horizons:
        if h > T:
            continue
        idx = h - 1
        gen = generated_frames[idx : idx + 1]
        gt = gt_frames[idx : idx + 1]

        if decode:
            gen = vqvae_decoder(gen)
            gt = vqvae_decoder(gt)

        results[M.inference_at("PSNR", h)] = compute_psnr(gen, gt)
        results[M.inference_at("SSIM", h)] = compute_ssim(gen, gt)["ssim"]
        results[M.inference_at("LPIPS", h)] = compute_lpips(gen, gt)

    for H in [16, 64]:
        if H > T:
            continue

        psnr_vals, ssim_vals, lpips_vals = [], [], []
        for i in range(H):
            gen = generated_frames[i : i + 1]
            gt = gt_frames[i : i + 1]
            if decode:
                gen = vqvae_decoder(gen)
                gt = vqvae_decoder(gt)
            psnr_vals.append(compute_psnr(gen, gt))
            ssim_vals.append(compute_ssim(gen, gt)["ssim"])
            lpips_vals.append(compute_lpips(gen, gt))

        results[M.inference_mean("PSNR", H)] = sum(psnr_vals) / len(psnr_vals)
        results[M.inference_mean("SSIM", H)] = sum(ssim_vals) / len(ssim_vals)
        results[M.inference_mean("LPIPS", H)] = sum(lpips_vals) / len(lpips_vals)

    return results


@torch.no_grad()
def compute_rollout_nll(
    gt_latent_trajectory: torch.Tensor,
    predicted_logits_per_step: List[torch.Tensor],
    H: int,
) -> float:
    """
    Negative log-likelihood of ground-truth latent trajectory over 1..H.

    Args:
        gt_latent_trajectory: [T, H_lat, W_lat] ground-truth token indices.
        predicted_logits_per_step: List of [K, H_lat, W_lat] logit tensors,
            one per rollout step.
        H: Horizon to evaluate.
    """
    total_nll = 0.0
    count = min(H, len(predicted_logits_per_step), gt_latent_trajectory.shape[0])
    if count == 0:
        return 0.0

    for t in range(count):
        logits = predicted_logits_per_step[t]
        target = gt_latent_trajectory[t]
        K = logits.shape[0]
        logits_flat = logits.permute(1, 2, 0).reshape(-1, K)
        target_flat = target.reshape(-1).long()
        nll = F.cross_entropy(logits_flat.float(), target_flat, reduction="mean")
        total_nll += nll.item()

    return total_nll / count


@torch.no_grad()
def compute_per_step_token_entropy(
    predicted_indices: torch.Tensor,
    num_embeddings: int,
) -> float:
    """
    Entropy of token distribution at a single rollout step.

    Args:
        predicted_indices: [H, W] token indices at one step.
        num_embeddings: Codebook size K.
    """
    flat = predicted_indices.reshape(-1).long()
    counts = torch.zeros(num_embeddings, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
    probs = counts / counts.sum().clamp(min=1)
    probs_nz = probs[probs > 0]
    entropy = -(probs_nz * probs_nz.log()).sum().item()
    return entropy


@torch.no_grad()
def compute_per_step_codebook_perplexity(
    predicted_indices: torch.Tensor,
    num_embeddings: int,
) -> float:
    """Perplexity = exp(entropy) at a single rollout step."""
    return math.exp(compute_per_step_token_entropy(predicted_indices, num_embeddings))


@torch.no_grad()
def compute_error_growth_rate(
    generated_frames: torch.Tensor,
    gt_frames: torch.Tensor,
    metric_fn,
    horizons: Optional[Sequence[int]] = None,
    vqvae_decoder: Optional[torch.nn.Module] = None,
) -> float:
    """
    Fit a linear slope to metric(h) vs h to measure error growth.

    Uses least-squares on the provided horizons.  Positive slope means
    the metric increases (e.g. LPIPS growing = degradation).
    """
    T = generated_frames.shape[0]
    if horizons is None:
        horizons = list(range(1, T + 1))

    xs, ys = [], []
    for h in horizons:
        if h > T:
            continue
        idx = h - 1
        gen = generated_frames[idx : idx + 1]
        gt = gt_frames[idx : idx + 1]
        if vqvae_decoder is not None:
            gen = vqvae_decoder(gen)
            gt = vqvae_decoder(gt)
        val = metric_fn(gen, gt)
        xs.append(float(h))
        ys.append(val)

    if len(xs) < 2:
        return 0.0

    import numpy as np
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    slope, _ = np.polyfit(xs_arr, ys_arr, 1)
    return float(slope)
