"""Rollout evaluation metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

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
        results[M.inference_at("SSIM", h)] = compute_ssim(gen, gt)
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
            ssim_vals.append(compute_ssim(gen, gt))
            lpips_vals.append(compute_lpips(gen, gt))

        results[M.inference_mean("PSNR", H)] = sum(psnr_vals) / len(psnr_vals)
        results[M.inference_mean("SSIM", H)] = sum(ssim_vals) / len(ssim_vals)
        results[M.inference_mean("LPIPS", H)] = sum(lpips_vals) / len(lpips_vals)

    return results
