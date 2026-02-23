"""
Loss functions for VQ-VAE training.

Combined loss:
    total = 1.0 * L1_recon + 0.1 * perceptual_LPIPS + 0.25 * commitment

All inputs expected in [-1, 1] range.
"""

import torch
import torch.nn.functional as F
import lpips


class PerceptualLoss(torch.nn.Module):
    """
    Perceptual loss using LPIPS with a pretrained VGG16 backbone.

    LPIPS compares images at multiple VGG feature layers, capturing structural
    and textural similarity beyond pixel-level metrics.

    The VGG model is frozen (no gradient updates).
    Both inputs must be in [-1, 1] range and have 3 channels.
    """

    def __init__(self, net="vgg"):
        super().__init__()
        # lpips.LPIPS returns a model with requires_grad=False for the backbone
        self.loss_fn = lpips.LPIPS(net=net)
        self.loss_fn.eval()
        # Freeze all parameters
        for param in self.loss_fn.parameters():
            param.requires_grad = False

    def forward(self, x_recon, x_target):
        """
        Args:
            x_recon: Reconstructed images [B, 3, H, W] in [-1, 1].
            x_target: Target images [B, 3, H, W] in [-1, 1].

        Returns:
            Scalar perceptual loss (mean over batch).
        """
        # LPIPS expects float32 inputs
        return self.loss_fn(x_recon.float(), x_target.float()).mean()


def compute_loss(x_recon, x_target, commit_loss, perceptual_loss_fn,
                 recon_weight=1.0, perceptual_weight=0.1, commit_weight=0.25):
    """
    Compute combined VQ-VAE loss.

    Args:
        x_recon: Reconstructed images [B, 3, H, W] in [-1, 1].
        x_target: Target images [B, 3, H, W] in [-1, 1].
        commit_loss: Scalar commitment loss from VectorQuantizerEMA.
        perceptual_loss_fn: PerceptualLoss module instance.
        recon_weight: Weight for L1 reconstruction loss.
        perceptual_weight: Weight for LPIPS perceptual loss.
        commit_weight: Weight for VQ commitment loss.

    Returns:
        total_loss: Scalar total loss (for backward).
        loss_dict: Dictionary of individual loss components (detached, for logging).
    """
    # L1 reconstruction loss
    recon_loss = F.l1_loss(x_recon, x_target)

    # Perceptual loss (no gradient through VGG, but gradient flows through x_recon)
    with torch.no_grad():
        p_loss = perceptual_loss_fn(x_recon, x_target)

    # Combined loss
    total_loss = (
        recon_weight * recon_loss
        + perceptual_weight * p_loss
        + commit_weight * commit_loss
    )

    loss_dict = {
        "recon_l1": recon_loss.detach(),
        "perceptual": p_loss.detach(),
        "commitment": commit_loss.detach(),
        "total": total_loss.detach(),
    }

    return total_loss, loss_dict
