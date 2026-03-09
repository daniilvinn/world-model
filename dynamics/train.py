"""Training script for the dynamics model."""

import argparse
import math
import os
import tempfile
import time
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dynamics.model import DynamicsUNet
from dynamics.dataset import create_dataloaders
from dynamics.inference import rollout

from evaluation.config import load_eval_config
from evaluation.orchestrator import EvalOrchestrator
from logger.wandb_logger import WandbLogger
from logger.metric_names import M
from logger.gradient_stats import compute_gradient_stats


class EMA:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}
    
    def update(self, model):
        """Update EMA weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
    
    def apply_shadow(self, model):
        """Apply EMA weights to model (for inference)."""
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])
    
    def store(self, model):
        """Store current model weights (before applying EMA)."""
        self.backup = {name: param.clone() for name, param in model.named_parameters()}
    
    def restore(self, model):
        """Restore model weights (after applying EMA)."""
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])


def get_ss_probability(epoch, total_epochs, ss_start_epoch, ss_max_prob=0.5):
    """
    Scheduled sampling probability schedule.
    
    Linear ramp from 0 to ss_max_prob starting at ss_start_epoch.
    """
    if epoch < ss_start_epoch:
        return 0.0
    progress = (epoch - ss_start_epoch) / max(total_epochs - ss_start_epoch, 1)
    return min(progress * ss_max_prob, ss_max_prob)


def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch, min_lr):
    """
    Build a built-in scheduler chain: Linear warmup -> Cosine annealing.
    """
    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_steps = max(int(total_epochs) * steps_per_epoch, 1)
    warmup_steps = max(0, int(warmup_epochs) * steps_per_epoch)

    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )

    if warmup_steps >= total_steps:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=total_steps,
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def _indices_to_embeddings(indices, codebook):
    """
    Convert token indices to codebook embeddings.

    Args:
        indices: [B, H, W] or [B, T, H, W] long
        codebook: [K, C]

    Returns:
        [B, C, H, W] or [B, T, C, H, W] float
    """
    if indices.dim() == 3:
        # [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
        return codebook[indices].permute(0, 3, 1, 2)
    if indices.dim() == 4:
        # [B, T, H, W] -> [B, T, H, W, C] -> [B, T, C, H, W]
        return codebook[indices].permute(0, 1, 4, 2, 3)
    raise ValueError(f"indices must be [B,H,W] or [B,T,H,W], got shape {tuple(indices.shape)}")


def training_step(
    model,
    context_indices,
    target_indices,
    action,
    codebook,
    diffusion_forcing=True,
    max_context_noise=0.2,
):
    """
    Flow matching training step with optional diffusion forcing.
    """
    B = target_indices.shape[0]
    device = target_indices.device

    context_zq = _indices_to_embeddings(context_indices, codebook)  # [B, ctx_len, 16, 32, 32]
    target_zq = _indices_to_embeddings(target_indices, codebook)  # [B, 16, 32, 32]
    
    if diffusion_forcing and model.training:
        if torch.rand(1).item() < 0.5:
            noise_levels = torch.rand(B, context_zq.shape[1], 1, 1, 1, device=device) * max_context_noise
            noise = torch.randn_like(context_zq)
            context_zq = (1 - noise_levels) * context_zq + noise_levels * noise
    
    context_flat = context_zq.reshape(B, -1, 32, 32)
    
    x_0 = torch.randn_like(target_zq)
    x_1 = target_zq
    
    t = torch.rand(B, device=device)
    
    t_expand = t[:, None, None, None]
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    
    logits = model(x_t, t, context_flat, action)
    loss = F.cross_entropy(logits, target_indices)
    
    return loss


def rollout_training_step(
    model,
    context_indices,
    targets_indices,
    actions,
    codebook,
    p_ss=0.5,
    temperature=1.0,
    rollout_ode_steps=4,
    diffusion_forcing=True,
    max_context_noise=0.2,
):
    """
    Multi-step rollout training with scheduled sampling.
    """
    B, rollout_length = actions.shape
    device = targets_indices.device
    
    current_context_indices = context_indices.clone()
    
    total_loss = 0.0
    
    for step in range(rollout_length):
        if diffusion_forcing and model.training:
            if torch.rand(1).item() < 0.5:
                current_context = _indices_to_embeddings(current_context_indices, codebook)
                noise_levels = torch.rand(B, current_context.shape[1], 1, 1, 1, device=device) * max_context_noise
                noise = torch.randn_like(current_context)
                step_context = (1 - noise_levels) * current_context + noise_levels * noise
            else:
                step_context = _indices_to_embeddings(current_context_indices, codebook)
        else:
            step_context = _indices_to_embeddings(current_context_indices, codebook)
        
        context_flat = step_context.reshape(B, -1, 32, 32)
        
        target_indices = targets_indices[:, step]
        target_zq = _indices_to_embeddings(target_indices, codebook)
        action = actions[:, step]
        
        x_0 = torch.randn_like(target_zq)
        x_1 = target_zq
        
        t = torch.rand(B, device=device)
        
        t_expand = t[:, None, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        logits = model(x_t, t, context_flat, action)
        step_loss = F.cross_entropy(logits, target_indices)
        total_loss += step_loss
        
        if step < rollout_length - 1:
            with torch.no_grad():
                x = torch.randn_like(target_zq)
                ss_dt = 1.0 / rollout_ode_steps
                for ode_i in range(rollout_ode_steps):
                    t_ode = torch.full((B,), ode_i * ss_dt, device=device)
                    logits_ode = model(x, t_ode, context_flat, action)
                    probs_ode = torch.softmax(logits_ode, dim=1)
                    x1_hat = torch.einsum("bkhw,kc->bchw", probs_ode, codebook)
                    denom = (1.0 - t_ode[:, None, None, None]).clamp(min=1e-5)
                    v = (x1_hat - x) / denom
                    x = x + v * ss_dt

                t_final = torch.full((B,), 1.0 - 1e-4, device=device)
                pred_logits = model(x, t_final, context_flat, action)

                if temperature > 0:
                    probs = torch.softmax(pred_logits / temperature, dim=1)
                    Bp, K, H, W = probs.shape
                    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, K)
                    predicted_indices = torch.multinomial(probs_flat, num_samples=1).squeeze(1).view(Bp, H, W)
                else:
                    predicted_indices = pred_logits.argmax(dim=1)
            
            use_prediction = torch.rand(B, device=device) < p_ss

            next_frame_indices = torch.where(
                use_prediction[:, None, None],
                predicted_indices,
                targets_indices[:, step]
            )

            current_context_indices = torch.cat([
                current_context_indices[:, 1:],
                next_frame_indices.unsqueeze(1)
            ], dim=1)
    
    avg_loss = total_loss / rollout_length
    
    return avg_loss


@torch.no_grad()
def validate(model, val_loader, device, rollout_length=1,
             codebook=None, temperature=1.0, rollout_ode_steps=4,
             diffusion_forcing=True, max_context_noise=0.2):
    """
    Run validation and return average loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        context_indices, targets_indices, actions = batch
        context_indices = context_indices.to(device)
        targets_indices = targets_indices.to(device)
        actions = actions.to(device)
        
        if rollout_length == 1:
            target_indices = targets_indices[:, 0] if targets_indices.dim() == 4 else targets_indices
            action = actions[:, 0] if actions.dim() == 2 else actions
            loss = training_step(
                model,
                context_indices,
                target_indices,
                action,
                codebook,
                diffusion_forcing,
                max_context_noise,
            )
        else:
            loss = rollout_training_step(
                model,
                context_indices,
                targets_indices,
                actions,
                codebook=codebook,
                p_ss=0.0,
                temperature=temperature,
                rollout_ode_steps=rollout_ode_steps,
                diffusion_forcing=diffusion_forcing,
                max_context_noise=max_context_noise
            )
        
        total_loss += loss.item()
        num_batches += 1
    
    model.train()
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def generate_rollout_video(
    model,
    ema,
    vqvae_model,
    val_loader,
    codebook,
    device,
    num_frames=32,
    num_actions=2,
    ode_steps=10,
    temperature=1.0,
    eval_chunk_size=16,
    fps=10,
):
    """
    Generate a short random rollout GIF and return its file path.
    """
    num_val_batches = len(val_loader)
    if num_val_batches == 0 or num_frames <= 0:
        return None

    random_batch_idx = torch.randint(0, num_val_batches, (1,)).item()
    sampled_batch = None
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx == random_batch_idx:
            sampled_batch = batch
            break

    if sampled_batch is None:
        return None

    context_indices, _, _ = sampled_batch
    if context_indices.shape[0] == 0:
        return None
    sample_idx = torch.randint(0, context_indices.shape[0], (1,)).item()
    context_one_idx = context_indices[sample_idx:sample_idx + 1].to(device, non_blocking=True)
    context_one = codebook[context_one_idx].permute(0, 1, 4, 2, 3)

    action_seq = torch.randint(0, num_actions, (num_frames,), device=device, dtype=torch.long)
    was_training = model.training
    gif_path = None

    try:
        ema.store(model)
        ema.apply_shadow(model)
        model.eval()
        vqvae_model.eval()

        rollout_latents = rollout(
            model,
            context_one,
            action_seq,
            num_ode_steps=ode_steps,
            device=device,
            codebook=codebook,
            temperature=temperature,
        )

        decoded_chunks = []
        chunk_size = max(1, int(eval_chunk_size))
        for start in range(0, rollout_latents.shape[0], chunk_size):
            end = min(start + chunk_size, rollout_latents.shape[0])
            latent_chunk = rollout_latents[start:end].to(device, non_blocking=True)
            decoded_chunks.append(vqvae_model.decode(latent_chunk).detach().cpu())

        if not decoded_chunks:
            return None

        decoded = torch.cat(decoded_chunks, dim=0)  # [T, 3, 256, 512]
        video_uint8 = ((decoded.clamp(-1, 1) + 1.0) * 127.5).byte()
        frames = [Image.fromarray(video_uint8[i].permute(1, 2, 0).numpy()) for i in range(video_uint8.shape[0])]
        if not frames:
            return None

        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp_file:
            gif_path = tmp_file.name

        frame_duration_ms = max(1, int(1000 / max(1, int(fps))))
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )
        return gif_path
    finally:
        ema.restore(model)
        if was_training:
            model.train()


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Data (training loader)
    train_loader, val_loader, codebook = create_dataloaders(
        latents_dir=args.latents_dir,
        vqvae_checkpoint=args.vqvae_checkpoint,
        context_length=args.context_length,
        rollout_length=args.rollout_length,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    
    codebook = codebook.detach().to(device)
    
    # Get latent_dim from VQ-VAE config so dynamics input channels match (latent_dim for target + context)
    vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location="cpu", weights_only=False)
    vqvae_cfg = vqvae_ckpt.get("config", {})
    latent_dim = vqvae_cfg.get("latent_dim", 16)
    in_channels = latent_dim * (1 + args.context_length)
    print(f"VQ-VAE latent_dim={latent_dim} -> dynamics in_channels={in_channels}")
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    print(f"Rollout length: {args.rollout_length} ({'single-step' if args.rollout_length == 1 else 'multi-step rollout'})")
    
    # Model
    model = DynamicsUNet(
        in_channels=in_channels,
        num_embeddings=args.num_embeddings,
        bottleneck_dim=args.bottleneck_dim,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        cond_dim=args.cond_dim,
        context_length=args.context_length,
        num_actions=args.num_actions,
        attn_resolution=args.attn_resolution,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    if args.warmup_epochs > args.epochs:
        raise ValueError(
            f"warmup_epochs ({args.warmup_epochs}) cannot be greater than epochs ({args.epochs})"
        )

    # Scheduler
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        min_lr=args.min_lr,
    )
    
    # EMA
    ema = EMA(model, decay=args.ema_decay)
    
    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    if device.type == "cuda":
        print(f"Mixed precision (FP16) enabled")

    eval_config = load_eval_config(args.wandb_config)
    run_config = {
        "model": "Dynamics UNet",
        "context_length": args.context_length,
        "base_channels": args.base_channels,
        "channel_mults": args.channel_mults,
        "cond_dim": args.cond_dim,
        "num_actions": args.num_actions,
        "attn_resolution": args.attn_resolution,
        "num_embeddings": args.num_embeddings,
        "latent_dim": latent_dim,
        "in_channels": in_channels,
        "bottleneck_dim": args.bottleneck_dim,
        "temperature": args.temperature,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "diffusion_forcing": args.diffusion_forcing,
        "max_context_noise": args.max_context_noise,
        "rollout_length": args.rollout_length,
        "rollout_ode_steps": args.rollout_ode_steps,
        "ss_start_epoch": args.ss_start_epoch,
        "ss_max_prob": args.ss_max_prob,
        "ema_decay": args.ema_decay,
        "eval_chunk_size": args.eval_chunk_size,
        "num_eval_sequences": args.num_eval_sequences,
        "eval_ode_steps": args.eval_ode_steps,
        "max_one_step_batches": args.max_one_step_batches,
        "eval_config": eval_config.raw,
    }
    wb = WandbLogger(
        config=run_config,
        run_name=args.run_name,
        group="Dynamics",
        tags=["dynamics", "training"],
        enabled=not args.no_wandb,
        metric_enabled_fn=eval_config.is_metric_enabled,
    )

    wb.log_architecture(model, "Dynamics UNet", extra_metadata={
        "in_channels": in_channels,
        "out_channels": args.num_embeddings,
        "bottleneck_dim": args.bottleneck_dim,
        "spatial_path": "32x32 -> 16x16 -> 8x8 -> 16x16 -> 32x32",
        "num_actions": args.num_actions,
    })

    eval_rollout_length = args.rollout_length
    if eval_config.short_horizons:
        eval_rollout_length = max(eval_rollout_length, max(eval_config.short_horizons))
    if eval_config.fvd_clip_lengths:
        eval_rollout_length = max(eval_rollout_length, max(eval_config.fvd_clip_lengths))

    eval_val_loader = val_loader
    if eval_rollout_length > args.rollout_length:
        print(
            f"Building eval validation loader with rollout_length={eval_rollout_length} "
            f"(train rollout_length={args.rollout_length})"
        )
        _, eval_val_loader, _ = create_dataloaders(
            latents_dir=args.latents_dir,
            vqvae_checkpoint=args.vqvae_checkpoint,
            context_length=args.context_length,
            rollout_length=eval_rollout_length,
            batch_size=args.batch_size,
            val_split=0.1,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )

    vqvae_model = None
    if os.path.exists(args.vqvae_checkpoint):
        from vqvae.model import VQVAE
        vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=False)
        vqvae_cfg = vqvae_ckpt.get("config", {})
        vqvae_model = VQVAE(
            latent_dim=vqvae_cfg.get("latent_dim", 16),
            num_embeddings=vqvae_cfg.get("num_embeddings", 1024),
            commitment_cost=vqvae_cfg.get("commitment_cost", 0.25),
            ema_decay=vqvae_cfg.get("ema_decay", 0.99),
        ).to(device)
        vqvae_model.load_state_dict(vqvae_ckpt["model_state_dict"])
        vqvae_model.eval()

    orchestrator = EvalOrchestrator(
        eval_config,
        wb,
        device,
        eval_chunk_size=args.eval_chunk_size,
        num_eval_sequences=args.num_eval_sequences,
        eval_ode_steps=args.eval_ode_steps,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ema.shadow = ckpt["ema_state_dict"]
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step = ckpt.get("global_step", start_epoch * steps_per_epoch)
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    grad_enabled = eval_config.gradient_stats_enabled
    grad_modules = eval_config.gradient_modules("dynamics")

    print(f"\n{'='*60}")
    print(f"Training Dynamics Model for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        epoch_step = epoch + 1
        epoch_ss_prob = get_ss_probability(
            epoch, args.epochs, args.ss_start_epoch, args.ss_max_prob
        ) if args.rollout_length > 1 else 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (context_indices, targets_indices, actions) in enumerate(pbar):
            context_indices = context_indices.to(device, non_blocking=True)
            targets_indices = targets_indices.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                if args.rollout_length == 1:
                    target_indices = targets_indices[:, 0]
                    action = actions[:, 0]
                    loss = training_step(
                        model, context_indices, target_indices, action, codebook,
                        diffusion_forcing=args.diffusion_forcing,
                        max_context_noise=args.max_context_noise
                    )
                else:
                    loss = rollout_training_step(
                        model, context_indices, targets_indices, actions,
                        codebook=codebook,
                        p_ss=epoch_ss_prob,
                        temperature=args.temperature,
                        rollout_ode_steps=args.rollout_ode_steps,
                        diffusion_forcing=args.diffusion_forcing,
                        max_context_noise=args.max_context_noise
                    )
            
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            ema.update(model)
            
            global_step += 1
            epoch_loss += loss.item()
            
            if args.rollout_length > 1:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "ss_prob": f"{epoch_ss_prob:.3f}"})
            else:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
        train_ppl = math.exp(avg_epoch_loss)
        print(f"\nEpoch {epoch+1} train (avg) | loss: {avg_epoch_loss:.4f} | ppl: {train_ppl:.1f} | time: {epoch_time:.1f}s")

        current_lr = optimizer.param_groups[0]["lr"]
        train_log = {
            M.train_loss("Token CE"): avg_epoch_loss,
            M.train_perplexity(): train_ppl,
            M.learning_rate(): current_lr,
        }
        if args.rollout_length > 1:
            train_log[M.scheduled_sampling_prob()] = epoch_ss_prob
        wb.log(train_log, step=epoch_step)

        if grad_enabled:
            grad_stats = compute_gradient_stats(model, grad_modules)
            wb.log(grad_stats, step=epoch_step)
        
        val_loss = validate(
            model, val_loader, device,
            rollout_length=args.rollout_length,
            codebook=codebook,
            temperature=args.temperature,
            rollout_ode_steps=args.rollout_ode_steps,
            diffusion_forcing=args.diffusion_forcing,
            max_context_noise=args.max_context_noise
        )
        val_ppl = math.exp(val_loss)
        print(f"Epoch {epoch+1} val   (avg) | loss: {val_loss:.4f} | ppl: {val_ppl:.1f}")
        
        wb.log({
            M.val_loss("Token CE"): val_loss,
            M.val_perplexity(): val_ppl,
        }, step=epoch_step)

        if vqvae_model is not None:
            orchestrator.run_dynamics_epoch_eval(
                model, vqvae_model, eval_val_loader,
                epoch_step, epoch_step, codebook,
                temperature=args.temperature,
                max_one_step_batches=args.max_one_step_batches,
            )

        if vqvae_model is not None and wb.enabled:
            rollout_gif_path = generate_rollout_video(
                model=model,
                ema=ema,
                vqvae_model=vqvae_model,
                val_loader=val_loader,
                codebook=codebook,
                device=device,
                num_frames=32,
                num_actions=args.num_actions,
                ode_steps=args.eval_ode_steps,
                temperature=args.temperature,
                eval_chunk_size=args.eval_chunk_size,
                fps=10,
            )
            if rollout_gif_path is not None:
                wb.log_video(
                    M.dynamics_rollout_video(),
                    rollout_gif_path,
                    step=epoch_step,
                    caption=f"epoch={epoch_step}, frames=32, random policy",
                    fps=10,
                )
                try:
                    os.remove(rollout_gif_path)
                except OSError:
                    pass
        
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": {
                "context_length": args.context_length,
                "base_channels": args.base_channels,
                "channel_mults": args.channel_mults,
                "cond_dim": args.cond_dim,
                "num_actions": args.num_actions,
                "attn_resolution": args.attn_resolution,
                "num_embeddings": args.num_embeddings,
                "latent_dim": latent_dim,
                "in_channels": in_channels,
                "bottleneck_dim": args.bottleneck_dim,
                "temperature": args.temperature,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "epochs": args.epochs,
                "diffusion_forcing": args.diffusion_forcing,
                "max_context_noise": args.max_context_noise,
                "rollout_length": args.rollout_length,
                "rollout_ode_steps": args.rollout_ode_steps,
                "ss_start_epoch": args.ss_start_epoch,
                "ss_max_prob": args.ss_max_prob,
                "ema_decay": args.ema_decay,
                "eval_chunk_size": args.eval_chunk_size,
                "num_eval_sequences": args.num_eval_sequences,
                "eval_ode_steps": args.eval_ode_steps,
                "max_one_step_batches": args.max_one_step_batches,
            },
        }
        
        latest_path = os.path.join(args.checkpoint_dir, "dynamics_latest.pt")
        torch.save(checkpoint, latest_path)
        if eval_config.log_latest_checkpoint:
            wb.log_checkpoint(latest_path, "dynamics-latest", metadata={"epoch": epoch, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.checkpoint_dir, "dynamics_best.pt")
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, best_path)
            print(f"New best model! val_loss={val_loss:.4f}")
            if eval_config.log_best_checkpoint:
                wb.log_checkpoint(best_path, "dynamics-best", metadata={"epoch": epoch, "val_loss": val_loss})
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{args.patience})")
        
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"(no improvement for {args.patience} epochs)")
            break
        
        print()
    
    wb.log_summary({"best_val_loss": best_val_loss})
    wb.finish()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.checkpoint_dir, 'dynamics_best.pt')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dynamics Model")
    
    # Data
    parser.add_argument("--latents_dir", type=str, default="latents",
                        help="Directory containing precomputed latents")
    parser.add_argument("--vqvae_checkpoint", type=str, default="checkpoints/vqvae_best.pt",
                        help="Path to VQ-VAE checkpoint (to extract codebook)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (None = use all)")
    
    # Model
    parser.add_argument("--context_length", type=int, default=4,
                        help="Number of context frames")
    parser.add_argument("--base_channels", type=int, default=128,
                        help="Base channel dimension")
    parser.add_argument("--channel_mults", type=int, nargs="+", default=[1, 2, 2],
                        help="Channel multipliers for each level")
    parser.add_argument("--cond_dim", type=int, default=256,
                        help="Conditioning vector dimension")
    parser.add_argument("--num_actions", type=int, default=2,
                        help="Number of action classes")
    parser.add_argument("--attn_resolution", type=int, default=8,
                        help="Resolution at which to apply self-attention")
    parser.add_argument("--num_embeddings", type=int, default=1024,
                        help="Number of codebook entries (token vocabulary size)")
    parser.add_argument("--bottleneck_dim", type=int, default=64,
                        help="Bottleneck channels before 1x1 logit projection")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5,
                        help="Minimum learning rate (end of cosine decay)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="Linear warmup epochs")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    
    # Flow matching & diffusion forcing
    parser.add_argument("--diffusion_forcing", action="store_true", default=True,
                        help="Enable diffusion forcing (context corruption)")
    parser.add_argument("--max_context_noise", type=float, default=0.2,
                        help="Maximum noise level for diffusion forcing")
    
    # Rollout training & scheduled sampling
    parser.add_argument("--rollout_length", type=int, default=1,
                        help="Number of frames to predict in rollout (1 = single-step, 2-4 = multi-step)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for scheduled sampling and eval-time generation")
    parser.add_argument("--rollout_ode_steps", type=int, default=4,
                        help="ODE integration steps for scheduled sampling predictions during rollout training")
    parser.add_argument("--ss_start_epoch", type=int, default=20,
                        help="Epoch to start scheduled sampling")
    parser.add_argument("--ss_max_prob", type=float, default=0.5,
                        help="Maximum scheduled sampling probability (use model predictions)")

    # Evaluation controls
    parser.add_argument("--eval_chunk_size", type=int, default=16,
                        help="Chunk size for latent->RGB decode during heavy eval")
    parser.add_argument("--num_eval_sequences", type=int, default=4,
                        help="Number of sequences sampled for heavy rollout eval")
    parser.add_argument("--eval_ode_steps", type=int, default=10,
                        help="ODE integration steps used by evaluation rollouts/one-step metrics")
    parser.add_argument("--max_one_step_batches", type=int, default=50,
                        help="Maximum batches for one-step eval in each eval epoch (0 disables cap)")
    
    # EMA
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=50,
                        help="Log to W&B every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint save directory")
    
    # W&B
    parser.add_argument("--wandb_config", type=str, default="wandb_config.json",
                        help="Path to wandb/eval config JSON")
    parser.add_argument("--run_name", type=str, default=None,
                        help="W&B run name override")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
