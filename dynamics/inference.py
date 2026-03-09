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
    Legacy utility: nearest-neighbor quantization through VQ-VAE codebook.
    
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


def _sample_indices_from_logits(logits, temperature=1.0, top_k=None):
    """
    Sample discrete token indices from logits.

    Args:
        logits: [B, K, H, W]
        temperature: >0 for stochastic sampling, <=0 for argmax
        top_k: if set (>0), sample only from top-k logits per position

    Returns:
        indices: [B, H, W] long
    """
    if temperature <= 0:
        return logits.argmax(dim=1)

    sampling_logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(int(top_k), sampling_logits.shape[1])
        topk_vals, _ = torch.topk(sampling_logits, k, dim=1)
        kth_vals = topk_vals[:, -1:, :, :]
        sampling_logits = sampling_logits.masked_fill(sampling_logits < kth_vals, float("-inf"))

    probs = torch.softmax(sampling_logits, dim=1)
    B, K, H, W = probs.shape
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)
    sampled = torch.multinomial(probs_flat, num_samples=1).squeeze(1)
    return sampled.view(B, H, W)


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
    temperature=1.0,
    top_k=None,
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
        codebook: [K, C] VQ-VAE codebook embeddings (required)
        temperature: sampling temperature for final token sampling (>0 stochastic, <=0 argmax)
        top_k: if set (>0), sample final tokens only from top-k logits per position
        context_noise_levels: None, scalar, [ctx_len], or [B, ctx_len] noise levels in [0, 1]
    
    Returns:
        z_next: [B, 16, 32, 32] predicted next latent frame (embedded from sampled indices)
        indices: [B, 32, 32] sampled token indices
    """
    if context_zq.dim() != 5:
        raise ValueError(f"context_zq must be [B, ctx_len, C, H, W], got shape {tuple(context_zq.shape)}")
    if codebook is None:
        raise ValueError("codebook is required for logits->embedding conversion during inference")

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
    
    # Start from pure Gaussian noise (same as training interpolation anchor)
    x = torch.randn(B, C, H, W, device=device)
    codebook = codebook.to(device=device, dtype=x.dtype)
    
    dt = 1.0 / num_steps
    
    if solver == "euler":
        # Euler method (1st-order), x0-prediction converted to velocity.
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            logits = model(x, t, context_flat, action_tensor)
            probs = torch.softmax(logits, dim=1)
            x1_hat = torch.einsum("bkhw,kc->bchw", probs, codebook)
            denom = (1.0 - t[:, None, None, None]).clamp(min=1e-5)
            v = (x1_hat - x) / denom
            x = x + v * dt
    
    elif solver == "midpoint":
        # Midpoint method (2nd-order, better accuracy, 2x NFE)
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_mid = torch.full((B,), i * dt + 0.5 * dt, device=device)
            
            logits1 = model(x, t, context_flat, action_tensor)
            probs1 = torch.softmax(logits1, dim=1)
            x1_hat_1 = torch.einsum("bkhw,kc->bchw", probs1, codebook)
            denom1 = (1.0 - t[:, None, None, None]).clamp(min=1e-5)
            v1 = (x1_hat_1 - x) / denom1
            
            # Midpoint
            x_mid = x + v1 * 0.5 * dt
            
            logits2 = model(x_mid, t_mid, context_flat, action_tensor)
            probs2 = torch.softmax(logits2, dim=1)
            x1_hat_2 = torch.einsum("bkhw,kc->bchw", probs2, codebook)
            denom2 = (1.0 - t_mid[:, None, None, None]).clamp(min=1e-5)
            v2 = (x1_hat_2 - x_mid) / denom2
            
            # Full step using midpoint velocity
            x = x + v2 * dt
    
    else:
        raise ValueError(f"Unknown solver: {solver}. Use 'euler' or 'midpoint'.")
    
    # Sample final tokens from logits at t=1, then map back to codebook embeddings.
    t_final = torch.full((B,), 1.0 - 1e-4, device=device)
    logits_final = model(x, t_final, context_flat, action_tensor)
    indices = _sample_indices_from_logits(logits_final, temperature=temperature, top_k=top_k)
    z_q = codebook[indices].permute(0, 3, 1, 2)
    
    return z_q, indices


@torch.no_grad()
def rollout(
    model,
    initial_context_zq,
    actions,
    num_ode_steps=20,
    solver="euler",
    device="cuda",
    codebook=None,
    temperature=1.0,
    top_k=None,
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
        codebook: [K, C] VQ-VAE codebook embeddings (required)
        temperature: sampling temperature for logits (>0 stochastic, <=0 argmax)
        top_k: if set (>0), sample final tokens only from top-k logits per position
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

        z_next, _ = predict_next_frame(
            model,
            context,
            action,
            num_ode_steps,
            solver,
            device,
            codebook,
            temperature=temperature,
            top_k=top_k,
            context_noise_levels=step_noise_levels,
        )
        generated.append(z_next)
        
        # Shift context window: drop oldest, append newest
        context = torch.cat([context[:, 1:], z_next.unsqueeze(1)], dim=1)
    
    # Stack generated frames
    return torch.cat(generated, dim=0)  # [N, 16, 32, 32]
