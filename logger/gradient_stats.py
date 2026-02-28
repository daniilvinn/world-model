"""
Gradient statistics collection for W&B logging.

Computes global and per-module gradient norm / mean / std / max
after ``loss.backward()`` and before ``optimizer.step()``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from logger.metric_names import M


def _grad_stats_for_params(
    params: List[torch.Tensor],
) -> Dict[str, float]:
    """Return norm / mean / std / max over a flat concatenation of gradients."""
    grads = [p.grad.detach().float().flatten() for p in params if p.grad is not None]
    if not grads:
        return {}
    flat = torch.cat(grads)
    return {
        "Norm": flat.norm().item(),
        "Mean": flat.mean().item(),
        "Std": flat.std().item(),
        "Max": flat.abs().max().item(),
    }


def compute_gradient_stats(
    model: nn.Module,
    module_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute gradient statistics for a model.

    Args:
        model: The model whose parameters have ``.grad`` populated.
        module_names: Named children to compute per-module stats for.
            If *None*, only global stats are returned.

    Returns:
        Dictionary of ``{metric_name: value}`` ready for ``wandb.log``.
        Keys use the ``M.gradient(stat, scope)`` naming convention.
    """
    all_params = [p for p in model.parameters() if p.grad is not None]
    if not all_params:
        return {}

    results: Dict[str, float] = {}

    global_stats = _grad_stats_for_params(all_params)
    for stat, value in global_stats.items():
        results[M.gradient(stat, "Global")] = value

    if module_names:
        named_children = dict(model.named_children())
        for name in module_names:
            child = named_children.get(name)
            if child is None:
                continue
            params = [p for p in child.parameters() if p.grad is not None]
            stats = _grad_stats_for_params(params)
            label = _module_label(name)
            for stat, value in stats.items():
                results[M.gradient(stat, label)] = value

    return results


def _module_label(name: str) -> str:
    """Convert snake_case module name to Title Case label."""
    return name.replace("_", " ").title()
