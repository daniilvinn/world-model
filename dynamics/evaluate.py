"""Standalone evaluation for the dynamics model."""

import argparse
import os

import torch
from tqdm import tqdm

from dynamics.model import DynamicsUNet
from dynamics.inference import predict_next_frame, rollout as run_rollout
from vqvae.model import VQVAE

from evaluation.config import load_eval_config
from evaluation.single_frame_metrics import evaluate_single_frame
from evaluation.rollout_metrics import evaluate_rollout_at_horizons
from evaluation.temporal_metrics import evaluate_temporal
from evaluation.action_metrics import evaluate_action_metrics
from evaluation.failure_metrics import evaluate_failure_metrics
from evaluation.runtime_metrics import RuntimeProfiler
from logger.wandb_logger import WandbLogger
from logger.metric_names import M


def _load_dynamics(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = DynamicsUNet(
        in_channels=16 + 16 * cfg.get("context_length", 4),
        out_channels=16,
        base_channels=cfg.get("base_channels", 128),
        channel_mults=tuple(cfg.get("channel_mults", [1, 2, 2])),
        cond_dim=cfg.get("cond_dim", 256),
        context_length=cfg.get("context_length", 4),
        num_actions=cfg.get("num_actions", 2),
        attn_resolution=cfg.get("attn_resolution", 8),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def _load_vqvae(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = VQVAE(
        latent_dim=cfg.get("latent_dim", 16),
        num_embeddings=cfg.get("num_embeddings", 1024),
        commitment_cost=cfg.get("commitment_cost", 0.25),
        ema_decay=cfg.get("ema_decay", 0.99),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading dynamics model: {args.dynamics_checkpoint}")
    dynamics_model, dyn_cfg, dyn_ckpt = _load_dynamics(args.dynamics_checkpoint, device)
    print(f"  Epoch: {dyn_ckpt.get('epoch', '?')}, val_loss: {dyn_ckpt.get('val_loss', '?')}")

    print(f"Loading VQ-VAE: {args.vqvae_checkpoint}")
    vqvae_model, _ = _load_vqvae(args.vqvae_checkpoint, device)

    codebook = vqvae_model.quantizer.embedding.detach().clone()

    from dynamics.dataset import create_dataloaders
    _, val_loader, _ = create_dataloaders(
        latents_dir=args.latents_dir,
        vqvae_checkpoint=args.vqvae_checkpoint,
        context_length=dyn_cfg.get("context_length", 4),
        rollout_length=args.eval_rollout_length,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
    )

    eval_config = load_eval_config(args.wandb_config)

    wb = WandbLogger(
        config={
            "model": "Dynamics UNet",
            "mode": "evaluation",
            "checkpoint": args.dynamics_checkpoint,
            "epoch": dyn_ckpt.get("epoch", "?"),
            **dyn_cfg,
        },
        run_name=args.run_name,
        tags=["dynamics", "evaluation"],
        enabled=not args.no_wandb,
        metric_enabled_fn=eval_config.is_metric_enabled,
    )
    wb.log_architecture(dynamics_model, "Dynamics UNet", extra_metadata={
        "context_length": dyn_cfg.get("context_length", 4),
        "num_actions": dyn_cfg.get("num_actions", 2),
    })

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nComputing one-step metrics...")
    one_step_accum = {}
    num_batches = 0

    for batch in tqdm(val_loader, desc="One-step eval"):
        context_zq, targets_zq, actions = batch
        context_zq = context_zq.to(device)
        targets_zq = targets_zq.to(device)
        actions = actions.to(device)

        target_zq = targets_zq[:, 0] if targets_zq.dim() == 5 else targets_zq
        action = actions[:, 0] if actions.dim() == 2 else actions

        with torch.no_grad():
            pred_zq = predict_next_frame(
                dynamics_model, context_zq[:1], action[0].item(),
                num_steps=args.eval_ode_steps, device=device, codebook=codebook,
            )

        gt_decoded = vqvae_model.decode(target_zq[:1])
        pred_decoded = vqvae_model.decode(pred_zq[:1])

        metrics = evaluate_single_frame(pred_decoded, gt_decoded)
        for k, v in metrics.items():
            one_step_accum[k] = one_step_accum.get(k, 0.0) + v
        num_batches += 1

        if args.max_one_step_batches > 0 and num_batches >= args.max_one_step_batches:
            break

    one_step_avg = {M.one_step(k): v / num_batches for k, v in one_step_accum.items()}
    print("One-step metrics:")
    for k, v in one_step_avg.items():
        print(f"  {k}: {v:.4f}")
    wb.log(one_step_avg)

    print("\nGenerating rollouts for evaluation...")
    horizons = eval_config.short_horizons
    max_h = max(horizons) if horizons else 64

    rollout_batch = []
    gt_batch = []
    action_batch = []
    count = 0

    for batch in tqdm(val_loader, desc="Rollout generation"):
        context_zq, targets_zq, actions = batch
        context_zq = context_zq.to(device)
        targets_zq = targets_zq.to(device)
        actions = actions.to(device)

        B = context_zq.shape[0]
        for b in range(B):
            if count >= args.num_eval_sequences:
                break

            action_seq_len = min(max_h, targets_zq.shape[1] if targets_zq.dim() == 5 else 1)
            action_seq = actions[b, :action_seq_len].cpu().tolist() if actions.dim() == 2 else [actions[b].item()]
            if len(action_seq) < max_h:
                action_seq = action_seq * (max_h // max(len(action_seq), 1) + 1)
                action_seq = action_seq[:max_h]

            with torch.no_grad():
                gen_frames = run_rollout(
                    dynamics_model, context_zq[b:b+1],
                    action_seq[:max_h],
                    num_ode_steps=args.eval_ode_steps,
                    device=device, codebook=codebook,
                )

            rollout_batch.append(gen_frames)
            if targets_zq.dim() == 5:
                gt_batch.append(targets_zq[b, :gen_frames.shape[0]])
            action_batch.append(torch.tensor(action_seq[:gen_frames.shape[0]]))
            count += 1

        if count >= args.num_eval_sequences:
            break

    rollout_results = {}
    for i, gen in enumerate(rollout_batch):
        if i < len(gt_batch) and gt_batch[i].shape[0] == gen.shape[0]:
            r = evaluate_rollout_at_horizons(
                gen, gt_batch[i], horizons,
                vqvae_decoder=vqvae_model.decode,
            )
            for k, v in r.items():
                rollout_results[k] = rollout_results.get(k, 0.0) + v / len(rollout_batch)

    print("Rollout metrics:")
    for k, v in rollout_results.items():
        print(f"  {k}: {v:.4f}")
    wb.log(rollout_results)

    print("\nComputing temporal metrics...")
    temporal_results = {}
    for i, gen in enumerate(rollout_batch):
        if i < len(gt_batch) and gt_batch[i].shape[0] == gen.shape[0]:
            t = evaluate_temporal(
                gt_batch[i], gen,
                vqvae_decoder=vqvae_model.decode,
                compute_flow=args.heavy_metrics,
            )
            for k, v in t.items():
                temporal_results[k] = temporal_results.get(k, 0.0) + v / len(rollout_batch)

    for k, v in temporal_results.items():
        print(f"  {k}: {v:.4f}")
    wb.log(temporal_results)

    if args.heavy_metrics:
        print("\nComputing action metrics...")
        action_results = {}
        for i, gen in enumerate(rollout_batch):
            if i < len(action_batch):
                a = evaluate_action_metrics(
                    action_batch[i].to(device), gen,
                    vqvae_decoder=vqvae_model.decode,
                )
                for k, v in a.items():
                    action_results[k] = action_results.get(k, 0.0) + v / len(rollout_batch)

        for k, v in action_results.items():
            print(f"  {k}: {v:.4f}")
        wb.log(action_results)

    print("\nComputing failure metrics...")
    collapse_horizons = [h for h in [64, 128, 256] if h <= max_h]
    if collapse_horizons and rollout_batch:
        failure_results = evaluate_failure_metrics(
            rollout_batch, collapse_horizons,
            vqvae_decoder=vqvae_model.decode,
        )
        for k, v in failure_results.items():
            print(f"  {k}: {v:.4f}")
        wb.log(failure_results)

    print("\nProfiling runtime...")
    profiler = RuntimeProfiler(device)
    profiler.start()
    test_context = rollout_batch[0][:1].unsqueeze(0)[:, :dyn_cfg.get("context_length", 4)] if rollout_batch else None
    if test_context is not None and test_context.shape[1] >= dyn_cfg.get("context_length", 4):
        for _ in range(50):
            with torch.no_grad():
                predict_next_frame(
                    dynamics_model, test_context, 0,
                    num_steps=args.eval_ode_steps, device=device, codebook=codebook,
                )
            profiler.tick()
        runtime_results = profiler.finish()
        for k, v in runtime_results.items():
            print(f"  {k}: {v:.2f}")
        wb.log(runtime_results)

    wb.finish()
    print("\nEvaluation complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Dynamics Model")

    parser.add_argument("--dynamics_checkpoint", type=str, default="checkpoints/dynamics_best.pt")
    parser.add_argument("--vqvae_checkpoint", type=str, default="checkpoints/vqvae_best.pt")
    parser.add_argument("--latents_dir", type=str, default="latents")
    parser.add_argument("--output_dir", type=str, default="eval_output_dynamics")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_ode_steps", type=int, default=10,
                        help="ODE integration steps used in evaluation")
    parser.add_argument("--eval_chunk_size", type=int, default=16,
                        help="Chunk size for decode-heavy eval steps (reserved for parity with training CLI)")
    parser.add_argument("--num_eval_sequences", type=int, default=20,
                        help="Number of rollout sequences for evaluation")
    parser.add_argument("--max_one_step_batches", type=int, default=100,
                        help="Maximum batches for one-step eval (0 disables cap)")
    parser.add_argument("--eval_rollout_length", type=int, default=4,
                        help="Rollout length for dataset loading")
    parser.add_argument("--heavy_metrics", action="store_true",
                        help="Compute heavy metrics (FVD, action metrics, optical flow)")

    # W&B
    parser.add_argument("--wandb_config", type=str, default="wandb_config.json")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
