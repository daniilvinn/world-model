"""
Evaluation orchestrator.

Schedules per-epoch core metrics and periodic heavy metrics based on
``EvalConfig``, dispatching to the individual metric modules and pushing
results through a ``WandbLogger``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from evaluation.config import EvalConfig
from evaluation._fidelity_helpers import ensure_min_samples
from logger.metric_names import M
from logger.wandb_logger import WandbLogger


def _cuda_cleanup(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def _rollout_latent_batch(
    model: nn.Module,
    predict_next_frame_fn,
    initial_context: torch.Tensor,
    action_seq: torch.Tensor,
    num_ode_steps: int,
    device: torch.device,
    codebook: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Batched autoregressive rollout.

    Args:
        initial_context: [B, ctx, C, H, W]
        action_seq: [B, T] action ids

    Returns:
        [B, T, C, H, W] generated latent rollouts.
    """
    B, T = action_seq.shape
    context = initial_context
    generated_steps_cpu: list = []

    for t in range(T):
        actions_t = action_seq[:, t]
        z_next = predict_next_frame_fn(
            model,
            context,
            actions_t,
            num_steps=num_ode_steps,
            device=device,
            codebook=codebook,
        )
        generated_steps_cpu.append(z_next.detach().cpu())
        context = torch.cat([context[:, 1:], z_next.unsqueeze(1)], dim=1)
        del z_next
        _cuda_cleanup(device)

    return torch.stack(generated_steps_cpu, dim=1)


@torch.no_grad()
def _decode_rollout_sequence_chunked(
    vqvae_model: nn.Module,
    latent_seq: torch.Tensor,
    device: torch.device,
    chunk_size: int = 16,
) -> torch.Tensor:
    """
    Decode rollout [T, C, H, W] in chunks to keep peak VRAM bounded.
    """
    decoded_chunks: list = []
    T = latent_seq.shape[0]
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        latent_gpu = latent_seq[start:end].to(device, non_blocking=True)
        decoded_chunks.append(vqvae_model.decode(latent_gpu).detach().cpu())
        del latent_gpu
        _cuda_cleanup(device)
    return torch.cat(decoded_chunks, dim=0)


class EvalOrchestrator:
    def __init__(
        self,
        config: EvalConfig,
        logger: WandbLogger,
        device: torch.device,
        eval_chunk_size: int = 16,
        num_eval_sequences: int = 4,
        eval_ode_steps: int = 10,
    ) -> None:
        self.cfg = config
        self.logger = logger
        self.device = device
        self.eval_chunk_size = max(1, int(eval_chunk_size))
        self.num_eval_sequences = max(1, int(num_eval_sequences))
        self.eval_ode_steps = max(1, int(eval_ode_steps))

    def run_vqvae_epoch_eval(
        self,
        model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        commit: bool = True,
    ) -> Dict[str, float]:
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

            frame_metrics = evaluate_single_frame(x_recon, x)
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
                    from evaluation.single_frame_metrics import compute_fid

                    num = min(len(real_images_np), self.cfg.metric_num_samples("fid"))
                    min_fid = self.cfg.metric_min_samples("fid", default=1)
                    real_fid = ensure_min_samples(real_images_np[:num], min_fid)
                    fake_fid = ensure_min_samples(fake_images_np[:num], min_fid)
                    all_metrics[M.fid()] = compute_fid(real_fid, fake_fid)
                except ImportError:
                    pass
                except Exception:
                    pass

        self.logger.log(all_metrics, step=global_step, commit=commit)
        model.train()
        return all_metrics

    def run_dynamics_epoch_eval(
        self,
        model: nn.Module,
        vqvae_model: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        epoch: int,
        global_step: int,
        codebook: Optional[torch.Tensor] = None,
        max_one_step_batches: int = 50,
    ) -> Dict[str, float]:
        from dynamics.inference import predict_next_frame

        model.eval()
        vqvae_model.eval()

        all_metrics: Dict[str, float] = {}
        num_batches = 0
        one_step_accum: Dict[str, float] = {}
        heavy = self.cfg.should_run_heavy_metrics(epoch)
        real_images_np: list = []
        fake_images_np: list = []

        for batch in val_loader:
            context_zq, targets_zq, actions = batch
            context_one = context_zq[:1].to(self.device, non_blocking=True)
            if targets_zq.dim() == 5:
                target_zq = targets_zq[:1, 0].to(self.device, non_blocking=True)
            else:
                target_zq = targets_zq[:1].to(self.device, non_blocking=True)
            if actions.dim() == 2:
                action = actions[0, 0].to(self.device, non_blocking=True)
            else:
                action = actions[0].to(self.device, non_blocking=True)

            with torch.no_grad():
                pred_zq = predict_next_frame(
                    model, context_one, action.item(),
                    num_steps=self.eval_ode_steps, device=self.device, codebook=codebook,
                )

            gt_decoded = vqvae_model.decode(target_zq)
            pred_decoded = vqvae_model.decode(pred_zq[:1])

            from evaluation.single_frame_metrics import evaluate_single_frame
            frame_metrics = evaluate_single_frame(pred_decoded, gt_decoded)

            for k, v in frame_metrics.items():
                one_step_accum[k] = one_step_accum.get(k, 0.0) + v

            if heavy and self.cfg.should_run_metric("fid", epoch):
                real_np = ((gt_decoded + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
                fake_np = ((pred_decoded + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
                for i in range(real_np.shape[0]):
                    real_images_np.append(real_np[i])
                    fake_images_np.append(fake_np[i])

            del context_one, target_zq, action, pred_zq, gt_decoded, pred_decoded, frame_metrics
            _cuda_cleanup(self.device)
            num_batches += 1

            if max_one_step_batches > 0 and num_batches >= max_one_step_batches:
                break

        if num_batches > 0:
            for k, v in one_step_accum.items():
                all_metrics[M.one_step(k)] = v / num_batches

        if heavy and self.cfg.should_run_metric("fid", epoch) and len(real_images_np) > 0:
            try:
                from evaluation.single_frame_metrics import compute_fid

                num = min(len(real_images_np), self.cfg.metric_num_samples("fid"))
                min_fid = self.cfg.metric_min_samples("fid", default=1)
                real_fid = ensure_min_samples(real_images_np[:num], min_fid)
                fake_fid = ensure_min_samples(fake_images_np[:num], min_fid)
                all_metrics[M.fid()] = compute_fid(real_fid, fake_fid)
            except ImportError:
                print(
                    "Skipping FID: torch-fidelity is not installed. "
                    "Install with `pip install torch-fidelity`."
                )
            except Exception as e:
                print(f"Skipping FID: metric backend failed ({type(e).__name__}: {e}).")

        if heavy:
            self._run_heavy_dynamics_eval(
                model, vqvae_model, val_loader, epoch,
                codebook, all_metrics,
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
        codebook: Optional[torch.Tensor],
        results: Dict[str, float],
    ) -> None:
        from dynamics.inference import predict_next_frame

        horizons = self.cfg.short_horizons
        fvd_horizons = self.cfg.fvd_clip_lengths if self.cfg.should_run_metric("fvd", epoch) else []

        rollout_batch: list = []
        gt_batch: list = []
        action_batch: list = []

        required_horizons = [*horizons, *fvd_horizons]
        max_h = max(required_horizons) if required_horizons else 64
        num_eval_sequences = self.num_eval_sequences
        if max_h >= 512:
            stream_batch_size = 1
        elif max_h >= 256:
            stream_batch_size = 2
        else:
            stream_batch_size = max(1, min(getattr(val_loader, "batch_size", 1) or 1, 4))

        count = 0
        for batch in val_loader:
            context_zq, targets_zq, actions = batch

            B = context_zq.shape[0]
            for start in range(0, B, stream_batch_size):
                if count >= num_eval_sequences:
                    break
                chunk_cap = min(stream_batch_size, num_eval_sequences - count)
                end = min(start + chunk_cap, B)

                ctx_chunk = context_zq[start:end].to(self.device, non_blocking=True)
                actions_chunk = actions[start:end]

                if actions_chunk.dim() == 2:
                    action_seq_len = min(max_h, actions_chunk.shape[1])
                    action_seq = actions_chunk[:, :action_seq_len]
                    if action_seq_len < max_h:
                        pad = action_seq[:, -1:].repeat(1, max_h - action_seq_len)
                        action_seq = torch.cat([action_seq, pad], dim=1)
                else:
                    action_seq = actions_chunk.reshape(-1, 1).repeat(1, max_h)
                action_seq = action_seq.to(self.device, dtype=torch.long, non_blocking=True)

                gen_chunk = _rollout_latent_batch(
                    model=model,
                    predict_next_frame_fn=predict_next_frame,
                    initial_context=ctx_chunk,
                    action_seq=action_seq,
                    num_ode_steps=self.eval_ode_steps,
                    device=self.device,
                    codebook=codebook,
                )
                gen_chunk_cpu = gen_chunk.detach().cpu()

                if targets_zq.dim() == 5:
                    gt_chunk_cpu = targets_zq[start:end, :gen_chunk_cpu.shape[1]].detach().cpu()
                else:
                    gt_chunk_cpu = None
                action_seq_cpu = action_seq.detach().cpu()

                for i in range(gen_chunk_cpu.shape[0]):
                    rollout_batch.append(gen_chunk_cpu[i])
                    if gt_chunk_cpu is not None:
                        gt_batch.append(gt_chunk_cpu[i])
                    action_batch.append(action_seq_cpu[i, :gen_chunk_cpu.shape[1]])

                del ctx_chunk, action_seq, gen_chunk, gen_chunk_cpu, action_seq_cpu
                if gt_chunk_cpu is not None:
                    del gt_chunk_cpu
                _cuda_cleanup(self.device)
                count += end - start

            if count >= num_eval_sequences:
                break

        if rollout_batch:
            from evaluation.rollout_metrics import evaluate_rollout_at_horizons
            from evaluation.temporal_metrics import evaluate_temporal
            from evaluation.failure_metrics import evaluate_failure_metrics

            for i, gen in enumerate(rollout_batch):
                if i < len(gt_batch):
                    common_t = min(gen.shape[0], gt_batch[i].shape[0])
                    if common_t <= 0:
                        continue
                    gen_common = gen[:common_t].to(self.device, non_blocking=True)
                    gt_common = gt_batch[i][:common_t].to(self.device, non_blocking=True)
                    rollout_results = evaluate_rollout_at_horizons(
                        gen_common, gt_common, horizons,
                        vqvae_decoder=vqvae_model.decode,
                    )
                    for k, v in rollout_results.items():
                        results[k] = results.get(k, 0.0) + v / len(rollout_batch)

                    temporal_results = evaluate_temporal(
                        gt_common, gen_common,
                        vqvae_decoder=vqvae_model.decode,
                        compute_flow=self.cfg.should_run_metric("optical_flow", epoch),
                    )
                    for k, v in temporal_results.items():
                        results[k] = results.get(k, 0.0) + v / len(rollout_batch)
                    del gen_common, gt_common, rollout_results, temporal_results
                    _cuda_cleanup(self.device)

            if self.cfg.should_run_metric("fvd", epoch):
                try:
                    from evaluation.video_metrics import evaluate_fvd_at_clip_lengths

                    aligned_pairs = []
                    for gen, gt in zip(rollout_batch, gt_batch):
                        common_t = min(gen.shape[0], gt.shape[0])
                        if common_t > 0:
                            aligned_pairs.append((gen[:common_t], gt[:common_t]))
                    max_fvd_samples = self.cfg.metric_num_samples("fvd", default=1000)
                    aligned_pairs = aligned_pairs[:max_fvd_samples]

                    if aligned_pairs:
                        min_fvd_samples = self.cfg.metric_min_samples("fvd", default=2)
                        if len(aligned_pairs) < min_fvd_samples:
                            aligned_pairs = ensure_min_samples(aligned_pairs, min_fvd_samples)

                        decoded_gen = []
                        decoded_gt = []
                        for gen_latent, gt_latent in aligned_pairs:
                            gen_latent_gpu = gen_latent.to(self.device, non_blocking=True)
                            gt_latent_gpu = gt_latent.to(self.device, non_blocking=True)
                            gen_rgb = _decode_rollout_sequence_chunked(
                                vqvae_model, gen_latent_gpu, self.device, chunk_size=self.eval_chunk_size
                            )
                            gt_rgb = _decode_rollout_sequence_chunked(
                                vqvae_model, gt_latent_gpu, self.device, chunk_size=self.eval_chunk_size
                            )
                            decoded_gen.append(gen_rgb)
                            decoded_gt.append(gt_rgb)
                            del gen_latent_gpu, gt_latent_gpu
                            _cuda_cleanup(self.device)

                        common_t = min(t.shape[0] for t in decoded_gen)
                        if common_t > 0:
                            fake_videos = torch.stack(
                                [v[:common_t].clamp(-1, 1).add(1).mul(0.5) for v in decoded_gen], dim=0
                            )
                            real_videos = torch.stack(
                                [v[:common_t].clamp(-1, 1).add(1).mul(0.5) for v in decoded_gt], dim=0
                            )
                            clip_lengths = [h for h in self.cfg.fvd_clip_lengths if h <= common_t]
                            if not clip_lengths:
                                clip_lengths = [common_t]
                            fvd_results = evaluate_fvd_at_clip_lengths(
                                real_videos=real_videos,
                                fake_videos=fake_videos,
                                clip_lengths=clip_lengths,
                                device=self.device,
                            )
                            results.update(fvd_results)
                            del fake_videos, real_videos, fvd_results
                        del decoded_gen, decoded_gt
                        _cuda_cleanup(self.device)
                except ImportError:
                    pass
                except Exception:
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
                        actions_t = action_batch[i].to(self.device, non_blocking=True)
                        gen_gpu = gen.to(self.device, non_blocking=True)
                        action_results = evaluate_action_metrics(
                            actions_t, gen_gpu,
                            vqvae_decoder=vqvae_model.decode,
                        )
                        for k, v in action_results.items():
                            results[k] = results.get(k, 0.0) + v / len(rollout_batch)
                        del gen_gpu, actions_t, action_results
                        _cuda_cleanup(self.device)

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
