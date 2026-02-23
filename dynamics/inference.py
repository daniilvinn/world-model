"""
Inference utilities for dynamics model: ODE solvers and autoregressive rollout.
"""

import torch


def _get_model_attr(model, attr_name, default=None):
    """
    Read attributes from either a raw nn.Module or torch.compile wrapper.
    """
    if hasattr(model, attr_name):
        return getattr(model, attr_name)
    orig_mod = getattr(model, "_orig_mod", None)
    if orig_mod is not None and hasattr(orig_mod, attr_name):
        return getattr(orig_mod, attr_name)
    return default


def _align_context_length(context_zq, expected_context_length):
    """
    Align context frame count to what the model expects.

    - If too long: keep most recent frames.
    - If too short: left-pad with the oldest available frame.
    """
    if expected_context_length is None:
        return context_zq

    ctx_len = context_zq.shape[1]
    if ctx_len == expected_context_length:
        return context_zq

    if ctx_len > expected_context_length:
        return context_zq[:, -expected_context_length:]

    pad_frames = expected_context_length - ctx_len
    oldest = context_zq[:, :1]
    pad = oldest.expand(-1, pad_frames, -1, -1, -1)
    return torch.cat([pad, context_zq], dim=1)


def quantize_latent(z, codebook):
    """
    Quantize continuous latent through VQ-VAE codebook (nearest-neighbor lookup).
    
    This is critical for autoregressive generation: the dynamics model was trained
    on quantized latents (codebook vectors), so its outputs must be quantized before
    being fed back as context to keep the autoregressive loop in-distribution.
    
    Without this step, small deviations from codebook vectors accumulate over
    autoregressive steps, causing:
      - Mode collapse to flat terrain (most common training pattern)
      - Loss of action responsiveness (OOD context drowns out action signal)
      - Visual artifacts (VQ-VAE decoder receives off-codebook inputs)
    
    Args:
        z: [B, C, H, W] continuous latent from ODE integration
        codebook: [K, C] VQ-VAE codebook embeddings
    
    Returns:
        z_q: [B, C, H, W] quantized latent (each spatial position snapped to nearest codebook vector)
    """
    B, C, H, W = z.shape
    z_flat = z.float().permute(0, 2, 3, 1).reshape(-1, C)
    cb = codebook.float()
    
    # Squared L2 distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
    distances = (
        z_flat.pow(2).sum(dim=1, keepdim=True)
        + cb.pow(2).sum(dim=1, keepdim=False)
        - 2.0 * z_flat @ cb.t()
    )  # [BHW, K]
    
    indices = distances.argmin(dim=1)  # [BHW]
    z_q = cb[indices].reshape(B, H, W, C).permute(0, 3, 1, 2)
    return z_q


def apply_context_noise(context_zq, noise_levels):
    """
    Corrupt context frames with per-frame noise levels.

    Args:
        context_zq: [B, ctx_len, C, H, W] context frames
        noise_levels: None, scalar, [ctx_len], or [B, ctx_len] in [0, 1]

    Returns:
        Noised context with same shape as context_zq
    """
    if noise_levels is None:
        return context_zq

    if not torch.is_tensor(noise_levels):
        noise_levels = torch.tensor(noise_levels, device=context_zq.device, dtype=context_zq.dtype)
    else:
        noise_levels = noise_levels.to(device=context_zq.device, dtype=context_zq.dtype)

    if noise_levels.dim() == 0:
        noise_levels = noise_levels.view(1, 1, 1, 1, 1)
    elif noise_levels.dim() == 1:
        noise_levels = noise_levels.view(1, -1, 1, 1, 1)
    elif noise_levels.dim() == 2:
        noise_levels = noise_levels.view(noise_levels.shape[0], noise_levels.shape[1], 1, 1, 1)
    else:
        raise ValueError("noise_levels must be a scalar, [ctx_len], or [B, ctx_len].")

    noise_levels = noise_levels.clamp(0.0, 1.0)
    noise = torch.randn_like(context_zq)
    return (1 - noise_levels) * context_zq + noise_levels * noise


@torch.no_grad()
def predict_next_frame(
    model,
    context_zq,
    action,
    num_steps=20,
    solver="euler",
    device="cuda",
    codebook=None,
    context_noise_levels=None,
):
    """
    Predict z_{t+1} given context frames and action using ODE integration.
    
    Args:
        model: DynamicsUNet (in eval mode)
        context_zq: [1, ctx_len, 16, 32, 32] context latent frames
        action: int (0 or 1)
        num_steps: number of ODE integration steps (more = better quality)
        solver: "euler" or "midpoint"
        device: torch device
        codebook: [K, C] VQ-VAE codebook for post-ODE quantization (strongly recommended
                  for autoregressive use; keeps outputs in-distribution with training data)
        context_noise_levels: None, scalar, [ctx_len], or [B, ctx_len] noise levels in [0, 1]
    
    Returns:
        z_next: [1, 16, 32, 32] predicted next latent frame (quantized if codebook provided)
    """
    if context_zq.dim() != 5:
        raise ValueError(f"context_zq must be [B, ctx_len, C, H, W], got shape {tuple(context_zq.shape)}")

    # Match runtime context length to training-time model expectation.
    expected_ctx_len = _get_model_attr(model, "context_length", None)
    context_zq = _align_context_length(context_zq, expected_ctx_len)

    # Optionally corrupt context frames (Diffusion Forcing sampling)
    context_zq = apply_context_noise(context_zq, context_noise_levels)

    B, _, C, H, W = context_zq.shape

    # Flatten context: [B, ctx_len, C, H, W] -> [B, ctx_len*C, H, W]
    context_flat = context_zq.reshape(B, -1, H, W)

    if torch.is_tensor(action):
        action_tensor = action.to(device=device, dtype=torch.long)
        if action_tensor.dim() == 0:
            action_tensor = action_tensor.view(1).expand(B)
        elif action_tensor.dim() == 1 and action_tensor.shape[0] == 1 and B > 1:
            action_tensor = action_tensor.expand(B)
    else:
        action_tensor = torch.full((B,), int(action), device=device, dtype=torch.long)
    
    # Start from pure noise
    x = torch.randn(B, C, H, W, device=device)
    
    dt = 1.0 / num_steps
    
    if solver == "euler":
        # Euler method (1st-order)
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=device)
            v = model(x, t, context_flat, action_tensor)
            x = x + v * dt
    
    elif solver == "midpoint":
        # Midpoint method (2nd-order, better accuracy, 2x NFE)
        for i in range(num_steps):
            t = torch.tensor([i * dt], device=device)
            t_mid = torch.tensor([i * dt + 0.5 * dt], device=device)
            
            # First velocity at current point
            v1 = model(x, t, context_flat, action_tensor)
            
            # Midpoint
            x_mid = x + v1 * 0.5 * dt
            
            # Velocity at midpoint
            v2 = model(x_mid, t_mid, context_flat, action_tensor)
            
            # Full step using midpoint velocity
            x = x + v2 * dt
    
    else:
        raise ValueError(f"Unknown solver: {solver}. Use 'euler' or 'midpoint'.")
    
    # Quantize through VQ-VAE codebook to stay in-distribution for autoregressive use
    if codebook is not None:
        x = quantize_latent(x, codebook)
    
    return x  # predicted z_{t+1}


@torch.no_grad()
def rollout(
    model,
    initial_context_zq,
    actions,
    num_ode_steps=20,
    solver="euler",
    device="cuda",
    codebook=None,
    context_noise_levels=None,
    context_noise_schedule=None,
):
    """
    Autoregressively generate a sequence of frames.
    
    Args:
        model: DynamicsUNet (in eval mode)
        initial_context_zq: [1, ctx_len, 16, 32, 32] starting context
        actions: list of ints, length = number of frames to generate
        num_ode_steps: ODE integration steps per frame
        solver: "euler" or "midpoint"
        device: torch device
        codebook: [K, C] VQ-VAE codebook for post-ODE quantization (strongly recommended)
        context_noise_levels: None, scalar, [ctx_len], or [B, ctx_len] noise levels in [0, 1]
        context_noise_schedule: optional list of noise_levels per rollout step (len = actions)
    
    Returns:
        frames: [N, 16, 32, 32] generated latent frames (stacked, not batched)
    """
    context = initial_context_zq.clone()
    generated = []
    
    for step_idx, action in enumerate(actions):
        step_noise_levels = context_noise_levels
        if context_noise_schedule is not None:
            if step_idx >= len(context_noise_schedule):
                raise ValueError("context_noise_schedule length must match actions length.")
            step_noise_levels = context_noise_schedule[step_idx]

        z_next = predict_next_frame(
            model,
            context,
            action,
            num_ode_steps,
            solver,
            device,
            codebook,
            context_noise_levels=step_noise_levels,
        )
        generated.append(z_next)
        
        # Shift context window: drop oldest, append newest
        context = torch.cat([context[:, 1:], z_next.unsqueeze(1)], dim=1)
    
    # Stack generated frames
    return torch.cat(generated, dim=0)  # [N, 16, 32, 32]
