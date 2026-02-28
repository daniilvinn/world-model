"""
Single-frame visual quality metrics.

Covers: MSE, MAE, Huber, PSNR, SSIM (with l/c/s components), LPIPS,
FID, KID, codebook perplexity, and codebook utilization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger.metric_names import M


# ---------------------------------------------------------------------------
# Pixel-level
# ---------------------------------------------------------------------------


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error over a batch.  Inputs in [-1, 1]."""
    return F.mse_loss(pred, target).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred, target).item()


def compute_huber(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> float:
    return F.smooth_l1_loss(pred, target, beta=delta).item()


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Both inputs are expected in [-1, 1] (dynamic range = 2).
    """
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(4.0 / mse)


# ---------------------------------------------------------------------------
# SSIM (with l / c / s components)
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d[:, None] * k1d[None, :]
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    return_components: bool = False,
) -> Dict[str, float]:
    """
    Structural Similarity Index Measure.

    Inputs in [-1, 1].  Returns dict with keys ``"ssim"`` and optionally
    ``"l"``, ``"c"``, ``"s"`` luminance / contrast / structure components.
    """
    C1 = (0.01 * 2) ** 2
    C2 = (0.03 * 2) ** 2
    C3 = C2 / 2

    channels = pred.shape[1]
    kernel = _gaussian_kernel_2d(window_size, 1.5, channels, pred.device)
    padding = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=padding, groups=channels)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred ** 2, kernel, padding=padding, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target ** 2, kernel, padding=padding, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_xy

    sigma_x2 = sigma_x2.clamp(min=0)
    sigma_y2 = sigma_y2.clamp(min=0)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    result: Dict[str, float] = {"ssim": ssim_map.mean().item()}

    if return_components:
        l_comp = (2 * mu_xy + C1) / (mu_x2 + mu_y2 + C1)
        sigma_x = sigma_x2.sqrt()
        sigma_y = sigma_y2.sqrt()
        c_comp = (2 * sigma_x * sigma_y + C2) / (sigma_x2 + sigma_y2 + C2)
        s_comp = (sigma_xy + C3) / (sigma_x * sigma_y + C3)
        result["l"] = l_comp.mean().item()
        result["c"] = c_comp.mean().item()
        result["s"] = s_comp.mean().item()

    return result


# ---------------------------------------------------------------------------
# LPIPS (wraps vqvae.losses.PerceptualLoss or lpips directly)
# ---------------------------------------------------------------------------


_lpips_model: Optional[Any] = None


def _get_lpips(device: torch.device) -> Any:
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net="vgg").eval().to(device)
        for p in _lpips_model.parameters():
            p.requires_grad = False
    elif next(_lpips_model.parameters()).device != device:
        _lpips_model = _lpips_model.to(device)
    return _lpips_model


@torch.no_grad()
def compute_lpips(pred: torch.Tensor, target: torch.Tensor) -> float:
    """LPIPS perceptual distance.  Inputs in [-1, 1]."""
    model = _get_lpips(pred.device)
    return model(pred.float(), target.float()).mean().item()


# ---------------------------------------------------------------------------
# FID / KID (using clean-fid or torch-fidelity)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_fid(
    real_images: List[np.ndarray],
    fake_images: List[np.ndarray],
) -> float:
    """
    Frechet Inception Distance between two sets of uint8 HWC images.

    Uses ``torch_fidelity`` for reliable computation.
    """
    from evaluation._fidelity_helpers import fid_from_arrays
    return fid_from_arrays(real_images, fake_images)


@torch.no_grad()
def compute_kid(
    real_images: List[np.ndarray],
    fake_images: List[np.ndarray],
) -> float:
    """Kernel Inception Distance between two sets of uint8 HWC images."""
    from evaluation._fidelity_helpers import kid_from_arrays
    return kid_from_arrays(real_images, fake_images)


# ---------------------------------------------------------------------------
# Codebook statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_codebook_stats(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute codebook perplexity and utilization over a dataloader.

    Expects ``model`` to be a VQ-VAE with ``.encoder`` and ``.quantizer``.
    """
    model.eval()
    num_embeddings = model.quantizer.num_embeddings
    usage_counts = torch.zeros(num_embeddings, dtype=torch.long, device=device)
    total_tokens = 0

    for batch in dataloader:
        x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
        z_e = model.encoder(x)
        _, _, indices, _ = model.quantizer(z_e)
        flat = indices.reshape(-1)
        usage_counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
        total_tokens += flat.numel()

    usage_counts = usage_counts.cpu().float()
    used = (usage_counts > 0).sum().item()
    utilization = used / num_embeddings

    probs = usage_counts / usage_counts.sum().clamp(min=1)
    probs_nz = probs[probs > 0]
    entropy = -(probs_nz * probs_nz.log()).sum().item()
    perplexity = math.exp(entropy)

    return {
        M.codebook_perplexity(): perplexity,
        M.codebook_utilization(): utilization,
    }


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_single_frame(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_components: bool = False,
) -> Dict[str, float]:
    """
    Compute all single-frame pixel/perceptual metrics on a batch.

    Returns a dict with named keys ready for ``wandb.log``.
    """
    results: Dict[str, float] = {}

    results["MSE"] = compute_mse(pred, target)
    results["MAE"] = compute_mae(pred, target)
    results["PSNR"] = compute_psnr(pred, target)

    ssim_out = compute_ssim(pred, target, return_components=compute_components)
    results["SSIM"] = ssim_out["ssim"]
    if compute_components:
        results["SSIM (Luminance)"] = ssim_out["l"]
        results["SSIM (Contrast)"] = ssim_out["c"]
        results["SSIM (Structure)"] = ssim_out["s"]

    results["LPIPS"] = compute_lpips(pred, target)

    return results
