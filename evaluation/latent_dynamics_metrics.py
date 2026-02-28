"""
Latent-level dynamics metrics.

Covers: cross-entropy between GT and predicted token distributions,
token prediction accuracy, KL and JS divergence of predicted token
distributions vs empirical dataset distribution.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from logger.metric_names import M


@torch.no_grad()
def compute_cross_entropy(
    logits: torch.Tensor,
    target_indices: torch.Tensor,
) -> float:
    """
    Cross-entropy between predicted token logits and ground-truth indices.

    Args:
        logits: [B, K, H, W] or [B*H*W, K] predicted logit distribution.
        target_indices: [B, H, W] or [B*H*W] ground-truth codebook indices.
    """
    if logits.dim() == 4:
        B, K, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, K)
        target_indices = target_indices.reshape(-1)

    return F.cross_entropy(logits.float(), target_indices.long()).item()


@torch.no_grad()
def compute_token_accuracy(
    predicted_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> float:
    """
    Fraction of spatial positions where the predicted token index
    exactly matches the ground truth.
    """
    return (predicted_indices == target_indices).float().mean().item()


@torch.no_grad()
def compute_token_distribution(
    indices: torch.Tensor,
    num_embeddings: int,
) -> torch.Tensor:
    """Return empirical probability distribution [K] from a flat index tensor."""
    flat = indices.reshape(-1).long()
    counts = torch.zeros(num_embeddings, device=flat.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=counts.dtype))
    return counts / counts.sum().clamp(min=1)


@torch.no_grad()
def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(P || Q) where P and Q are discrete probability vectors."""
    mask = (p > 0) & (q > 0)
    if mask.sum() == 0:
        return float("inf")
    p_m, q_m = p[mask], q[mask]
    return (p_m * (p_m / q_m).log()).sum().item()


@torch.no_grad()
def compute_js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence (symmetric) between two distributions."""
    m = 0.5 * (p + q)
    kl_pm = compute_kl_divergence(p, m)
    kl_qm = compute_kl_divergence(q, m)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def evaluate_latent_dynamics(
    predicted_indices: torch.Tensor,
    target_indices: torch.Tensor,
    num_embeddings: int,
    dataset_distribution: Optional[torch.Tensor] = None,
    predicted_logits: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all latent-level dynamics metrics.

    Args:
        predicted_indices: [B, H, W] predicted token indices.
        target_indices: [B, H, W] ground-truth token indices.
        num_embeddings: Codebook size K.
        dataset_distribution: [K] empirical token distribution from training set.
            When provided, KL/JS divergence is computed.
        predicted_logits: [B, K, H, W] raw logits for cross-entropy.
            When ``None``, cross-entropy is skipped.

    Returns:
        Dict ready for ``wandb.log``.
    """
    results: Dict[str, float] = {}

    results[M.token_accuracy()] = compute_token_accuracy(predicted_indices, target_indices)

    if predicted_logits is not None:
        results[M.cross_entropy_latent()] = compute_cross_entropy(
            predicted_logits, target_indices
        )

    if dataset_distribution is not None:
        pred_dist = compute_token_distribution(predicted_indices, num_embeddings)
        results[M.kl_divergence_tokens()] = compute_kl_divergence(pred_dist, dataset_distribution)
        results[M.js_divergence_tokens()] = compute_js_divergence(pred_dist, dataset_distribution)

    return results
