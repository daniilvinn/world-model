"""
Model architecture serialization for W&B artifact logging.

Exports a JSON representation of a model's layer structure, shapes,
parameter counts, and hyperparameters — uploaded as a W&B artifact file,
NOT as a plot.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch.nn as nn


def _layer_info(name: str, module: nn.Module) -> Dict[str, Any]:
    """Summarize a single layer."""
    info: Dict[str, Any] = {
        "type": module.__class__.__name__,
        "params": sum(p.numel() for p in module.parameters(recurse=False)),
        "trainable_params": sum(
            p.numel() for p in module.parameters(recurse=False) if p.requires_grad
        ),
    }

    if isinstance(module, nn.Conv2d):
        info["in_channels"] = module.in_channels
        info["out_channels"] = module.out_channels
        info["kernel_size"] = list(module.kernel_size)
        info["stride"] = list(module.stride)
        info["padding"] = list(module.padding)
    elif isinstance(module, nn.Linear):
        info["in_features"] = module.in_features
        info["out_features"] = module.out_features
    elif isinstance(module, nn.Embedding):
        info["num_embeddings"] = module.num_embeddings
        info["embedding_dim"] = module.embedding_dim
    elif isinstance(module, nn.GroupNorm):
        info["num_groups"] = module.num_groups
        info["num_channels"] = module.num_channels
    elif isinstance(module, nn.MultiheadAttention):
        info["embed_dim"] = module.embed_dim
        info["num_heads"] = module.num_heads

    return info


def serialize_model_architecture(
    model: nn.Module,
    model_name: str = "model",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable dict describing the model architecture.

    Args:
        model: PyTorch model.
        model_name: Human-readable name (e.g. ``"VQ-VAE"``, ``"Dynamics UNet"``).
        extra_metadata: Optional dict merged into the top-level output
            (e.g. hyperparameters, input/output shapes).

    Returns:
        Nested dict ready for ``json.dumps``.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layers = OrderedDict()
    for name, module in model.named_modules():
        if name == "":
            continue
        layers[name] = _layer_info(name, module)

    arch: Dict[str, Any] = {
        "model_name": model_name,
        "class": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_M": round(total_params / 1e6, 2),
        "layers": layers,
    }

    if extra_metadata:
        arch["metadata"] = extra_metadata

    return arch


def save_architecture_json(
    model: nn.Module,
    path: str | Path,
    model_name: str = "model",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize architecture to a JSON file on disk and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arch = serialize_model_architecture(model, model_name, extra_metadata)
    path.write_text(json.dumps(arch, indent=2), encoding="utf-8")
    return path
