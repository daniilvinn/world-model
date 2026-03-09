"""
Dynamics model: conditional U-Net for token-logit flow matching on VQ-VAE latents.

Architecture:
    - Input: noisy target latent [16, 32, 32] + context frames [64, 32, 32] = [80, 32, 32]
    - Conditioning: flow time (sinusoidal embedding) + action (embedding)
    - Output: token logits [K, 32, 32], where K = num codebook entries

    Spatial: 32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32
    Channels: 128 -> 256 -> 256 -> 256 -> 128
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def sinusoidal_embedding(t, dim=256):
    """
    Sinusoidal time embedding (from DDPM).
    
    Args:
        t: [B] float in [0, 1]
        dim: embedding dimension
    
    Returns:
        [B, dim] float
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
    emb = t[:, None] * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditioningMLP(nn.Module):
    """
    Combines flow time embedding + action embedding into conditioning vector.
    
    Time: sinusoidal positional embedding (256-dim) -> MLP -> 256-dim
    Action: nn.Embedding(2, 64) -> 64-dim
    Combined: concat [256 + 64] -> Linear -> SiLU -> Linear -> 256-dim
    """
    
    def __init__(self, num_actions=2, time_dim=256, action_dim=64, cond_dim=256):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_dim)
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Combine time + action
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + action_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
    
    def forward(self, t, action):
        """
        Args:
            t: [B] flow time in [0, 1]
            action: [B] int action index
        
        Returns:
            [B, cond_dim] conditioning vector
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, dim=256)  # [B, 256]
        t_emb = self.time_mlp(t_emb)  # [B, 256]
        
        # Action embedding
        a_emb = self.action_embed(action)  # [B, action_dim]
        
        # Combine
        cond = torch.cat([t_emb, a_emb], dim=1)  # [B, 256 + action_dim]
        cond = self.cond_mlp(cond)  # [B, cond_dim]
        
        return cond


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------


class AdaGNResBlock(nn.Module):
    """
    Pre-activation ResBlock with Adaptive Group Normalization.
    
    AdaGN: the conditioning vector is projected to (scale, shift) pairs
    that modulate the group norm output: h = scale * GroupNorm(h) + shift
    
    Architecture:
        GroupNorm -> AdaGN modulate -> SiLU -> Conv3x3 ->
        GroupNorm -> AdaGN modulate -> SiLU -> Conv3x3 -> + skip
    """
    
    def __init__(self, in_ch, out_ch, cond_dim=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        
        # AdaGN projections
        self.cond_proj1 = nn.Linear(cond_dim, 2 * in_ch)
        self.cond_proj2 = nn.Linear(cond_dim, 2 * out_ch)
        
        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x, cond):
        """
        Args:
            x: [B, in_ch, H, W]
            cond: [B, cond_dim] conditioning vector
        
        Returns:
            [B, out_ch, H, W]
        """
        h = self.norm1(x)
        
        # AdaGN modulation 1
        scale1, shift1 = self.cond_proj1(cond).chunk(2, dim=1)  # [B, in_ch], [B, in_ch]
        h = h * (1 + scale1[:, :, None, None]) + shift1[:, :, None, None]
        
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        
        # AdaGN modulation 2
        scale2, shift2 = self.cond_proj2(cond).chunk(2, dim=1)  # [B, out_ch], [B, out_ch]
        h = h * (1 + scale2[:, :, None, None]) + shift2[:, :, None, None]
        
        h = self.act(h)
        h = self.conv2(h)
        
        return self.skip(x) + h


class SelfAttention(nn.Module):
    """
    Multi-head self-attention applied to spatial positions.
    
    Reused from vqvae/model.py -- same implementation.
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
    """Spatial downsampling via strided convolution."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Spatial upsampling via nearest-neighbor interpolation + conv."""
    
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class DynamicsUNet(nn.Module):
    """
    Small U-Net for flow matching on 32x32 VQ-VAE latents.
    
    Input channels: latent_dim (noisy target) + latent_dim*context_length (context frames)
                    = latent_dim * (1 + context_length). E.g. 80 for latent_dim=16, context_length=4.
    Output channels: num_embeddings (token logits)
    
    Spatial path: 32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32
    Channel path:  128  ->  256  -> 256 ->  256  ->  128
    
    Self-attention: ONLY at 8x8 resolution (64 tokens)
    Conditioning: AdaGN in every ResBlock, from time+action embedding
    
    ~5-6M parameters
    """
    
    def __init__(
        self,
        in_channels=80,
        num_embeddings=1024,
        bottleneck_dim=64,
        base_channels=128,
        channel_mults=(1, 2, 2),
        cond_dim=256,
        context_length=4,
        num_actions=2,
        attn_resolution=8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_embeddings = num_embeddings
        self.bottleneck_dim = bottleneck_dim
        self.context_length = context_length
        
        # Conditioning network
        self.cond_net = ConditioningMLP(num_actions=num_actions, cond_dim=cond_dim)
        
        # Channel dimensions at each level
        channels = [base_channels * mult for mult in channel_mults]
        
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        ch_in = base_channels
        for i, ch_out in enumerate(channels):
            # Two ResBlocks per level
            blocks = nn.ModuleList([
                AdaGNResBlock(ch_in, ch_out, cond_dim=cond_dim),
                AdaGNResBlock(ch_out, ch_out, cond_dim=cond_dim),
            ])
            
            # Add self-attention at 8x8 resolution (level 2)
            if 32 // (2 ** i) == attn_resolution:
                blocks.append(SelfAttention(ch_out, num_heads=4))
            
            self.encoder_blocks.append(blocks)
            
            # Downsample (except at last level)
            # After downsample, output channels become channels[i+1]
            if i < len(channels) - 1:
                self.downsamples.append(Downsample(ch_out, channels[i + 1]))
                ch_in = channels[i + 1]  # Next level input matches downsample output
            else:
                ch_in = ch_out  # Last level, no downsample
        
        # Bottleneck (at 8x8)
        self.mid_block1 = AdaGNResBlock(channels[-1], channels[-1], cond_dim=cond_dim)
        self.mid_attn = SelfAttention(channels[-1], num_heads=4)
        self.mid_block2 = AdaGNResBlock(channels[-1], channels[-1], cond_dim=cond_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Reverse channels for decoder (mirrors encoder)
        channels_reversed = list(reversed(channels))
        
        for i, ch_out in enumerate(channels_reversed):
            # Input channels from previous decoder level (or bottleneck)
            # At i=0: coming from bottleneck with channels[-1] = channels_reversed[0]
            # At i>0: coming from previous decoder level's Upsample output
            #         Upsample at level i-1 outputs channels_reversed[i]
            ch_in = channels_reversed[i]
            
            # Skip connection from encoder
            # Decoder i=0 gets skip from encoder level (len-1), i=1 gets (len-2), etc.
            # These have the channel counts from the original encoder outputs
            ch_skip = channels[len(channels) - 1 - i]
            ch_in_with_skip = ch_in + ch_skip
            
            # Two ResBlocks per level
            blocks = nn.ModuleList([
                AdaGNResBlock(ch_in_with_skip, ch_out, cond_dim=cond_dim),
                AdaGNResBlock(ch_out, ch_out, cond_dim=cond_dim),
            ])
            
            # Add self-attention at 8x8 resolution
            if 8 * (2 ** i) == attn_resolution:
                blocks.append(SelfAttention(ch_out, num_heads=4))
            
            self.decoder_blocks.append(blocks)
            
            # Upsample (except at last level)
            if i < len(channels_reversed) - 1:
                self.upsamples.append(Upsample(ch_out, channels_reversed[i + 1]))
        
        # Output convolution
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(base_channels, bottleneck_dim, kernel_size=3, padding=1)
        self.logit_proj = nn.Conv2d(bottleneck_dim, num_embeddings, kernel_size=1)
    
    def forward(self, x_t, t, context, action):
        """
        Args:
            x_t:     [B, C, 32, 32] noisy target latent at flow time t (C = latent_dim)
            t:       [B] flow time in [0, 1]
            context: [B, context_length*C, 32, 32] concatenated context frames
            action:  [B] int action index (0 or 1)
        
        Returns:
            logits:  [B, K, 32, 32] token logits over codebook entries
        """
        # Get conditioning vector
        cond = self.cond_net(t, action)  # [B, cond_dim]
        
        # Concatenate noisy target with context
        x = torch.cat([x_t, context], dim=1)  # [B, in_channels, 32, 32]
        
        # Input conv
        h = self.conv_in(x)  # [B, base_channels, 32, 32]
        
        # Encoder
        skip_connections = []
        
        for i, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                if isinstance(block, AdaGNResBlock):
                    h = block(h, cond)
                else:  # SelfAttention
                    h = block(h)
            
            skip_connections.append(h)
            
            # Downsample
            if i < len(self.downsamples):
                h = self.downsamples[i](h)
        
        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        
        # Decoder
        for i, blocks in enumerate(self.decoder_blocks):
            # Add skip connection
            skip = skip_connections[-(i + 1)]
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, AdaGNResBlock):
                    h = block(h, cond)
                else:  # SelfAttention
                    h = block(h)
            
            # Upsample
            if i < len(self.upsamples):
                h = self.upsamples[i](h)
        
        # Output
        h = self.act_out(self.norm_out(h))
        h = self.conv_out(h)  # [B, bottleneck_dim, 32, 32]
        logits = self.logit_proj(h)  # [B, num_embeddings, 32, 32]
        
        return logits
