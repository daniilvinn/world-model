"""
Evaluation orchestrator.

Schedules per-epoch core metrics and periodic heavy metrics based on
``EvalConfig``, dispatching to the individual metric modules and pushing
results through a ``WandbLogger``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from evaluation.config import EvalConfig
from logger.metric_names import M
from logger.wandb_logger import WandbLogger


def _repeat_to_min_samples(items: list, min_samples: int) -> list:
    """Repeat elements cyclically so the returned list has at least min_samples."""
    if min_samples <= 0 or len(items) >= min_samples or not items:
        return items
    return [items[i % len(items)] for i in range(min_samples)]


@torch.no_grad()
def _decode_rollout_sequence(vqvae_model: nn.Module, latent_seq: torch.Tensor) -> torch.Tensor:
    """Decode latent rollout [T, C, H, W] into RGB [T, 3, H, W]."""
    return vqvae_model.decode(latent_seq)


class EvalOrchestrator:
    """
    Coordinates all evaluation work for a training run.

    Instantiate once per training script, then call
    :meth:`run_epoch_eval` at the end of each epoch.
    """

    def __init__(
        self,
        config: EvalConfig,
        logger: WandbLogger,
        device: torch.device,
    ) -> None:
        self.cfg = config
        self.logger = logger
        self.device = device

    # ------------------------------------------------------------------
    # VQ-VAE evaluation
    # ------------------------------------------------------------------

    def run_vqvae_epoch_eval(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        commit: bool = True,
    ) -> Dict[str, float]:
        """
        Run core + scheduled heavy metrics for VQ-VAE.

        Always computes: PSNR, SSIM, LPIPS on validation reconstructions.
        Periodically computes: FID, KID, codebook stats, SSIM components.
        """
        from evaluation.single_frame_metrics import (
            compute_codebook_stats,
            evaluate_single_frame,
        )

        model.eval()

        all_metrics: Dict[str, float] = {}
        num_batches = 0
        accum: Dict[str, float] = {}

        heavy = self.cfg.should_run_heavy_metrics(epoch)
        real_images_np: list = []
        fake_images_np: list = []

        for batch in val_loader:
            x = batch.to(self.device) if not isinstance(batch, (list, tuple)) else batch[0].to(self.device)

            with torch.no_grad():
                x_recon, _, _, _ = model(x)

            frame_metrics = evaluate_single_frame(
                x_recon, x, compute_components=True
            )
            for k, v in frame_metrics.items():
                accum[k] = accum.get(k, 0.0) + v
            num_batches += 1

            if heavy and self.cfg.should_run_metric("fid", epoch):
                real_np = ((x + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
                fake_np = ((x_recon + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
                for i in range(real_np.shape[0]):
                    real_images_np.append(real_np[i])
                    fake_images_np.append(fake_np[i])

        if num_batches > 0:
            for k, v in accum.items():
                avg = v / num_batches
                key = M.val_loss(k) if k in ("MSE", "MAE") else f"Evaluation/{k}"
                all_metrics[key] = avg

        if heavy:
            cb_stats = compute_codebook_stats(model, val_loader, self.device)
            all_metrics.update(cb_stats)

            if self.cfg.should_run_metric("fid", epoch) and len(real_images_np) > 0:
                try:
                    from evaluation.single_frame_metrics import compute_fid, compute_kid

                    num = min(len(real_images_np), self.cfg.metric_num_samples("fid"))
                    min_fid = self.cfg.metric_min_samples("fid", default=1)
                    real_fid = _repeat_to_min_samples(real_images_np[:num], min_fid)
                    fake_fid = _repeat_to_min_samples(fake_images_np[:num], min_fid)
                    all_metrics[M.fid()] = compute_fid(real_fid, fake_fid)

                    if self.cfg.should_run_metric("kid", epoch):
                        num_kid = min(len(real_images_np), self.cfg.metric_num_samples("kid"))
                        min_kid = self.cfg.metric_min_samples("kid", default=1)
                        real_kid = _repeat_to_min_samples(real_images_np[:num_kid], min_kid)
                        fake_kid = _repeat_to_min_samples(fake_images_np[:num_kid], min_kid)
                        all_metrics[M.kid()] = compute_kid(real_kid, fake_kid)
                except ImportError:
                    pass
                except Exception:
                    # Smoke runs may still fail in backend metric libs for tiny sets.
                    # Keep training running and report other metrics.
                    pass

        self.logger.log(all_metrics, step=global_step, commit=commit)
        model.train()
        return all_metrics

    # ------------------------------------------------------------------
    # Dynamics evaluation
    # ------------------------------------------------------------------

    def run_dynamics_epoch_eval(
        self,
        model: nn.Module,
        vqvae_model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        codebook: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run core + scheduled heavy metrics for the dynamics model.

        Always computes: one-step reconstruction metrics.
        Periodically computes: rollout quality, temporal stability, FVD,
        action metrics, failure analysis.
        """
        from dynamics.inference import predict_next_frame, rollout

        model.eval()
        vqvae_model.eval()

        all_metrics: Dict[str, float] = {}
        num_batches = 0
        one_step_accum: Dict[str, float] = {}

        for batch in val_loader:
            context_zq, targets_zq, actions = batch
            context_zq = context_zq.to(self.device)
            targets_zq = targets_zq.to(self.device)
            actions = actions.to(self.device)

            target_zq = targets_zq[:, 0] if targets_zq.dim() == 5 else targets_zq
            action = actions[:, 0] if actions.dim() == 2 else actions

            with torch.no_grad():
                pred_zq = predict_next_frame(
                    model, context_zq, action[0].item(),
                    num_steps=10, device=self.device, codebook=codebook,
                )

            gt_decoded = vqvae_model.decode(target_zq[:1])
            pred_decoded = vqvae_model.decode(pred_zq[:1])

            from evaluation.single_frame_metrics import evaluate_single_frame
            frame_metrics = evaluate_single_frame(pred_decoded, gt_decoded)

            for k, v in frame_metrics.items():
                one_step_accum[k] = one_step_accum.get(k, 0.0) + v
            num_batches += 1

            if num_batches >= 50:
                break

        if num_batches > 0:
            for k, v in one_step_accum.items():
                all_metrics[M.one_step(k)] = v / num_batches

        heavy = self.cfg.should_run_heavy_metrics(epoch)

        if heavy:
            self._run_heavy_dynamics_eval(
                model, vqvae_model, val_loader, epoch,
                global_step, codebook, all_metrics,
            )

        self.logger.log(all_metrics, step=global_step)
        model.train()
        return all_metrics

    def _run_heavy_dynamics_eval(
        self,
        model: nn.Module,
        vqvae_model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        codebook: Optional[torch.Tensor],
        results: Dict[str, float],
    ) -> None:
        """Run expensive rollout, temporal, action, and failure metrics."""
        from dynamics.inference import rollout as run_rollout

        horizons = self.cfg.short_horizons

        rollout_batch: list = []
        gt_batch: list = []
        action_batch: list = []

        max_h = max(horizons) if horizons else 64
        num_eval_sequences = 10

        count = 0
        for batch in val_loader:
            context_zq, targets_zq, actions = batch
            context_zq = context_zq.to(self.device)
            targets_zq = targets_zq.to(self.device)
            actions = actions.to(self.device)

            B = context_zq.shape[0]
            for b in range(B):
                if count >= num_eval_sequences:
                    break

                action_seq_len = min(max_h, targets_zq.shape[1] if targets_zq.dim() == 5 else 1)
                action_seq = actions[b, :action_seq_len].cpu().tolist() if actions.dim() == 2 else [actions[b].item()]

                if len(action_seq) < 2:
                    action_seq = action_seq * max_h

                with torch.no_grad():
                    gen_frames = run_rollout(
                        model,
                        context_zq[b:b+1],
                        action_seq[:max_h],
                        num_ode_steps=10,
                        device=self.device,
                        codebook=codebook,
                    )

                rollout_batch.append(gen_frames)
                if targets_zq.dim() == 5:
                    gt_batch.append(targets_zq[b, :gen_frames.shape[0]])
                action_batch.append(torch.tensor(action_seq[:gen_frames.shape[0]]))
                count += 1

            if count >= num_eval_sequences:
                break

        if rollout_batch:
            from evaluation.rollout_metrics import evaluate_rollout_at_horizons
            from evaluation.temporal_metrics import evaluate_temporal
            from evaluation.failure_metrics import evaluate_failure_metrics

            for i, gen in enumerate(rollout_batch):
                if i < len(gt_batch) and gt_batch[i].shape[0] == gen.shape[0]:
                    rollout_results = evaluate_rollout_at_horizons(
                        gen, gt_batch[i], horizons,
                        vqvae_decoder=vqvae_model.decode,
                    )
                    for k, v in rollout_results.items():
                        results[k] = results.get(k, 0.0) + v / len(rollout_batch)

                    temporal_results = evaluate_temporal(
                        gt_batch[i], gen,
                        vqvae_decoder=vqvae_model.decode,
                        compute_flow=self.cfg.should_run_metric("optical_flow", epoch),
                    )
                    for k, v in temporal_results.items():
                        results[k] = results.get(k, 0.0) + v / len(rollout_batch)

            # FVD (video-level) at configured clip lengths/cadence.
            if self.cfg.should_run_metric("fvd", epoch):
                try:
                    from evaluation.video_metrics import evaluate_fvd_at_clip_lengths

                    # Keep only aligned pairs, cap by configured sample budget.
                    aligned_pairs = [
                        (gen, gt)
                        for gen, gt in zip(rollout_batch, gt_batch)
                        if gen.shape[0] == gt.shape[0]
                    ]
                    max_fvd_samples = self.cfg.metric_num_samples("fvd", default=1000)
                    aligned_pairs = aligned_pairs[:max_fvd_samples]

                    if aligned_pairs:
                        min_fvd_samples = self.cfg.metric_min_samples("fvd", default=2)
                        if len(aligned_pairs) < min_fvd_samples:
                            aligned_pairs = _repeat_to_min_samples(aligned_pairs, min_fvd_samples)

                        decoded_gen = []
                        decoded_gt = []
                        for gen_latent, gt_latent in aligned_pairs:
                            gen_rgb = _decode_rollout_sequence(vqvae_model, gen_latent).detach()
                            gt_rgb = _decode_rollout_sequence(vqvae_model, gt_latent).detach()
                            decoded_gen.append(gen_rgb)
                            decoded_gt.append(gt_rgb)

                        # Use the maximum common length across sequences.
                        common_t = min(t.shape[0] for t in decoded_gen)
                        if common_t > 0:
                            fake_videos = torch.stack(
                                [v[:common_t].clamp(-1, 1).add(1).mul(0.5) for v in decoded_gen], dim=0
                            )
                            real_videos = torch.stack(
                                [v[:common_t].clamp(-1, 1).add(1).mul(0.5) for v in decoded_gt], dim=0
                            )
                            fvd_results = evaluate_fvd_at_clip_lengths(
                                real_videos=real_videos,
                                fake_videos=fake_videos,
                                clip_lengths=self.cfg.fvd_clip_lengths,
                                device=self.device,
                            )
                            results.update(fvd_results)
                except ImportError:
                    pass
                except Exception:
                    # Keep training robust; FVD backends can fail on tiny smoke runs.
                    pass

            collapse_horizons = [h for h in [64, 128, 256] if h <= max_h]
            if collapse_horizons and self.cfg.should_run_metric("failure_metrics", epoch):
                failure_results = evaluate_failure_metrics(
                    rollout_batch, collapse_horizons,
                    vqvae_decoder=vqvae_model.decode,
                )
                results.update(failure_results)

            if self.cfg.should_run_metric("action_metrics", epoch):
                from evaluation.action_metrics import evaluate_action_metrics
                for i, gen in enumerate(rollout_batch):
                    if i < len(action_batch):
                        actions_t = action_batch[i].to(self.device)
                        action_results = evaluate_action_metrics(
                            actions_t, gen,
                            vqvae_decoder=vqvae_model.decode,
                        )
                        for k, v in action_results.items():
                            results[k] = results.get(k, 0.0) + v / len(rollout_batch)

    # ------------------------------------------------------------------
    # Runtime profiling
    # ------------------------------------------------------------------

    def run_runtime_profiling(
        self,
        model: nn.Module,
        generate_fn,
        num_frames: int,
        global_step: int,
    ) -> Dict[str, float]:
        """Profile inference and log runtime metrics."""
        from evaluation.runtime_metrics import profile_inference

        results = profile_inference(model, generate_fn, num_frames, self.device)
        self.logger.log(results, step=global_step)
        return results
