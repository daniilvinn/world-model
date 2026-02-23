"""
VQ-VAE model for game screenshot compression.

Architecture:
    Encoder:  [B, 3, 256, 512] -> [B, 16, 32, 32]
    VQ:       quantizes 16-dim vectors at each of 32x32 spatial positions
    Decoder:  [B, 16, 32, 32] -> [B, 3, 256, 512]

Uses 3 uniform stride-2 layers + 1 asymmetric stride-(1,2) layer in the encoder
to handle the 2:1 input aspect ratio while producing a square 32x32 latent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """
    Pre-activation residual block with GroupNorm and SiLU.

    If out_channels differs from in_channels, a 1x1 conv skip connection is used.
    """

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return self.skip(x) + h


class SelfAttention(nn.Module):
    """
    Multi-head self-attention applied to spatial positions.

    Only used at the 32x32 bottleneck (1024 tokens) where it is cheap.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


class Downsample(nn.Module):
    """
    Spatial downsampling via strided convolution.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        stride: Tuple (stride_h, stride_w). Use (2,2) for uniform or (1,2) for
                asymmetric (width-only) downsampling.
    """

    def __init__(self, in_ch, out_ch, stride=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """
    Spatial upsampling via nearest-neighbor interpolation + 3x3 conv.

    Uses nearest-neighbor (NOT transposed convolution) to avoid checkerboard
    artifacts, which are especially visible on sharp-edged game graphics.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        scale_factor: Tuple (scale_h, scale_w). Use (2,2) for uniform or (1,2)
                      for asymmetric (width-only) upsampling.
    """

    def __init__(self, in_ch, out_ch, scale_factor=(2, 2)):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """
    Encodes [B, 3, 256, 512] images to [B, 16, 32, 32] latent representations.

    Spatial path (H x W):
        256x512 -> 128x256 -> 64x128 -> 32x64 -> 32x32
        (3 uniform stride-2 + 1 asymmetric stride-(1,2))
    """

    def __init__(self, latent_dim=16):
        super().__init__()

        # Initial convolution: 3 -> 128, no spatial change
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)

        # Down Block 1: 256x512 -> 128x256
        self.down1 = Downsample(128, 128, stride=(2, 2))
        self.res1a = ResBlock(128)
        self.res1b = ResBlock(128)

        # Down Block 2: 128x256 -> 64x128
        self.down2 = Downsample(128, 256, stride=(2, 2))
        self.res2a = ResBlock(256)
        self.res2b = ResBlock(256)

        # Down Block 3: 64x128 -> 32x64
        self.down3 = Downsample(256, 256, stride=(2, 2))
        self.res3a = ResBlock(256)
        self.res3b = ResBlock(256)

        # Down Block 4 (ASYMMETRIC): 32x64 -> 32x32 (only W halved)
        self.down4 = Downsample(256, 256, stride=(1, 2))
        self.res4a = ResBlock(256)
        self.res4b = ResBlock(256)

        # Mid block with self-attention at 32x32
        self.mid_res1 = ResBlock(256)
        self.mid_attn = SelfAttention(256, num_heads=4)
        self.mid_res2 = ResBlock(256)

        # Pre-quantize projection: 256 -> latent_dim
        self.norm_out = nn.GroupNorm(32, 256)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(256, latent_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, 256, 512]
        h = self.conv_in(x)  # [B, 128, 256, 512]

        h = self.down1(h)    # [B, 128, 128, 256]
        h = self.res1a(h)
        h = self.res1b(h)

        h = self.down2(h)    # [B, 256, 64, 128]
        h = self.res2a(h)
        h = self.res2b(h)

        h = self.down3(h)    # [B, 256, 32, 64]
        h = self.res3a(h)
        h = self.res3b(h)

        h = self.down4(h)    # [B, 256, 32, 32]
        h = self.res4a(h)
        h = self.res4b(h)

        h = self.mid_res1(h)  # [B, 256, 32, 32]
        h = self.mid_attn(h)
        h = self.mid_res2(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)  # [B, 16, 32, 32]
        return h


# ---------------------------------------------------------------------------
# Vector Quantizer (EMA)
# ---------------------------------------------------------------------------


class VectorQuantizerEMA(nn.Module):
    """
    Vector Quantization with Exponential Moving Average (EMA) codebook updates.

    Replaces each 16-dim encoder output vector with its nearest codebook entry.
    Uses straight-through estimator for gradient flow.

    Features:
        - EMA codebook updates (more stable than gradient-based)
        - Dead code revival every `dead_code_revival_interval` steps to prevent
          codebook collapse
        - Tracks codebook utilization

    Args:
        num_embeddings: Number of codebook entries (K).
        embedding_dim: Dimension of each codebook vector.
        commitment_cost: Weight for commitment loss ||z_e - sg(z_q)||^2.
        ema_decay: Decay factor for EMA updates.
        dead_code_revival_interval: Check for dead codes every N forward passes.
    """

    def __init__(
        self,
        num_embeddings=1024,
        embedding_dim=16,
        commitment_cost=0.25,
        ema_decay=0.99,
        dead_code_revival_interval=100,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.dead_code_revival_interval = dead_code_revival_interval

        # Codebook embeddings (not a learnable parameter -- updated via EMA)
        self.register_buffer("embedding", torch.randn(num_embeddings, embedding_dim))
        # EMA cluster sizes
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        # EMA sum of assigned encoder outputs
        self.register_buffer("ema_weight", self.embedding.clone())
        # Step counter for dead code revival
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # Initialize codebook uniformly
        nn.init.uniform_(self.embedding, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.ema_weight.copy_(self.embedding)

    def forward(self, z_e):
        """
        Args:
            z_e: Encoder output, shape [B, embedding_dim, H, W].

        Returns:
            z_q_st: Quantized output (with straight-through gradient), same shape as z_e.
            commitment_loss: Scalar commitment loss.
            indices: Codebook indices, shape [B, H, W].
            usage: Fraction of codebook entries used in this batch (0 to 1).
        """
        B, C, H, W = z_e.shape
        assert C == self.embedding_dim, (
            f"Expected {self.embedding_dim} channels, got {C}"
        )

        # Force float32 for numerical stability under mixed precision
        z_e_float = z_e.float()
        embedding_float = self.embedding.float()

        # Reshape: [B, C, H, W] -> [BHW, C]
        z_flat = z_e_float.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

        # Compute distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z.e
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + embedding_float.pow(2).sum(dim=1, keepdim=False)
            - 2.0 * z_flat @ embedding_float.t()
        )  # [BHW, K]

        # Find nearest codebook entry
        indices_flat = distances.argmin(dim=1)  # [BHW]
        indices = indices_flat.reshape(B, H, W)

        # Look up quantized vectors
        z_q_flat = embedding_float[indices_flat]  # [BHW, C]
        z_q = z_q_flat.reshape(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)
        # z_q: [B, C, H, W] in float32

        # EMA codebook update (training only)
        if self.training:
            self.step_counter += 1

            with torch.no_grad():
                # One-hot encode assignments: [BHW, K]
                encodings = F.one_hot(indices_flat, self.num_embeddings).float()

                # Count assignments per codebook entry
                n_k = encodings.sum(dim=0)  # [K]

                # Sum of encoder outputs assigned to each entry
                sum_k = encodings.t() @ z_flat  # [K, C]

                # EMA update
                self.ema_count.mul_(self.ema_decay).add_(
                    n_k, alpha=1.0 - self.ema_decay
                )
                self.ema_weight.mul_(self.ema_decay).add_(
                    sum_k, alpha=1.0 - self.ema_decay
                )

                # Laplace smoothing to avoid division by zero
                n = self.ema_count.sum()
                count_smoothed = (
                    (self.ema_count + 1e-5)
                    / (n + self.num_embeddings * 1e-5)
                    * n
                )

                # Update codebook
                self.embedding.copy_(self.ema_weight / count_smoothed.unsqueeze(1))

                # Dead code revival
                if self.step_counter % self.dead_code_revival_interval == 0:
                    dead_mask = self.ema_count < 1.0  # [K]
                    num_dead = dead_mask.sum().item()
                    if num_dead > 0:
                        # Replace dead codes with random encoder outputs from batch
                        num_replace = min(num_dead, z_flat.shape[0])
                        random_indices = torch.randperm(
                            z_flat.shape[0], device=z_flat.device
                        )[:num_replace]
                        replacements = z_flat[random_indices]

                        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_replace]
                        self.embedding[dead_indices] = replacements
                        self.ema_weight[dead_indices] = replacements
                        self.ema_count[dead_indices] = 1.0

        # Commitment loss: ||z_e - sg(z_q)||^2
        commitment_loss = F.mse_loss(z_e_float, z_q.detach())

        # Straight-through estimator: gradients bypass quantization
        z_q_st = z_e + (z_q.to(z_e.dtype) - z_e).detach()

        # Codebook utilization: fraction of codes used in this batch
        unique_codes = len(indices_flat.unique())
        usage = unique_codes / self.num_embeddings

        return z_q_st, commitment_loss, indices, usage


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """
    Decodes [B, 16, 32, 32] latent representations to [B, 3, 256, 512] images.

    Spatial path (H x W):
        32x32 -> 32x64 -> 64x128 -> 128x256 -> 256x512
        (1 asymmetric scale-(1,2) + 3 uniform scale-2)
    """

    def __init__(self, latent_dim=16):
        super().__init__()

        # Post-quantize projection: latent_dim -> 256
        self.conv_in = nn.Conv2d(latent_dim, 256, kernel_size=3, padding=1)

        # Mid block with self-attention at 32x32
        self.mid_res1 = ResBlock(256)
        self.mid_attn = SelfAttention(256, num_heads=4)
        self.mid_res2 = ResBlock(256)

        # Up Block 1 (ASYMMETRIC): 32x32 -> 32x64 (only W doubled)
        self.up1 = Upsample(256, 256, scale_factor=(1, 2))
        self.res1a = ResBlock(256)
        self.res1b = ResBlock(256)

        # Up Block 2: 32x64 -> 64x128
        self.up2 = Upsample(256, 256, scale_factor=(2, 2))
        self.res2a = ResBlock(256)
        self.res2b = ResBlock(256)

        # Up Block 3: 64x128 -> 128x256
        self.up3 = Upsample(256, 128, scale_factor=(2, 2))
        self.res3a = ResBlock(128)
        self.res3b = ResBlock(128)

        # Up Block 4: 128x256 -> 256x512
        self.up4 = Upsample(128, 128, scale_factor=(2, 2))
        self.res4a = ResBlock(128)
        self.res4b = ResBlock(128)

        # Output convolution
        self.norm_out = nn.GroupNorm(32, 128)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(128, 3, kernel_size=3, padding=1)

    def forward(self, z_q):
        # z_q: [B, 16, 32, 32]
        h = self.conv_in(z_q)  # [B, 256, 32, 32]

        h = self.mid_res1(h)   # [B, 256, 32, 32]
        h = self.mid_attn(h)
        h = self.mid_res2(h)

        h = self.up1(h)        # [B, 256, 32, 64]
        h = self.res1a(h)
        h = self.res1b(h)

        h = self.up2(h)        # [B, 256, 64, 128]
        h = self.res2a(h)
        h = self.res2b(h)

        h = self.up3(h)        # [B, 128, 128, 256]
        h = self.res3a(h)
        h = self.res3b(h)

        h = self.up4(h)        # [B, 128, 256, 512]
        h = self.res4a(h)
        h = self.res4b(h)

        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)   # [B, 3, 256, 512]
        h = torch.tanh(h)      # Output in [-1, 1]
        return h


# ---------------------------------------------------------------------------
# VQVAE Wrapper
# ---------------------------------------------------------------------------


class VQVAE(nn.Module):
    """
    Full VQ-VAE model: Encoder -> Vector Quantizer -> Decoder.

    Input:  [B, 3, 256, 512]  (game frames resized from 420x840, normalized to [-1,1])
    Latent: [B, 16, 32, 32]   (quantized through 1024-entry codebook)
    Output: [B, 3, 256, 512]  (reconstructed frames in [-1,1])
    """

    def __init__(
        self,
        latent_dim=16,
        num_embeddings=1024,
        commitment_cost=0.25,
        ema_decay=0.99,
    ):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        """
        Full forward pass: encode -> quantize -> decode.

        Args:
            x: Input images [B, 3, 256, 512] in [-1, 1].

        Returns:
            x_recon: Reconstructed images [B, 3, 256, 512] in [-1, 1].
            commit_loss: Scalar commitment loss.
            indices: Codebook indices [B, 32, 32].
            usage: Codebook utilization fraction (0 to 1).
        """
        z_e = self.encoder(x)
        z_q, commit_loss, indices, usage = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, commit_loss, indices, usage

    @torch.no_grad()
    def encode(self, x):
        """
        Encode to quantized latent (inference).

        Returns:
            z_q: Quantized latent [B, 16, 32, 32].
            indices: Codebook indices [B, 32, 32].
        """
        z_e = self.encoder(x)
        z_q, _, indices, _ = self.quantizer(z_e)
        return z_q, indices

    @torch.no_grad()
    def decode(self, z_q):
        """
        Decode quantized latent to images (inference).

        Returns:
            x_recon: Reconstructed images [B, 3, 256, 512] in [-1, 1].
        """
        return self.decoder(z_q)

    @torch.no_grad()
    def decode_indices(self, indices):
        """
        Decode from codebook indices directly (for world model integration).

        Args:
            indices: Codebook indices [B, 32, 32].

        Returns:
            x_recon: Reconstructed images [B, 3, 256, 512] in [-1, 1].
        """
        B, H, W = indices.shape
        indices_flat = indices.reshape(-1)
        z_q_flat = self.quantizer.embedding[indices_flat]  # [BHW, C]
        z_q = z_q_flat.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # [B, C, H, W]
        return self.decoder(z_q)
