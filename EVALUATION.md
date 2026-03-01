# Evaluation Guide

This document describes the evaluation pipeline for the World Model project: how metrics are computed, when they run, and how they are logged to Weights & Biases.

---

## Overview

Evaluation is driven by an **orchestrator** that runs at the end of each training epoch (VQ-VAE and dynamics) and by **standalone evaluation scripts** for one-off runs. All scalar metrics are sent to W&B and organized into namespaces (Optimization, Validation, Evaluation, Runtime). Heavy metrics (FID, FVD, optical flow, action/failure metrics) run only on a configurable cadence to control compute cost.

**Two entry points:**

1. **During training** — `EvalOrchestrator` is invoked each epoch from `vqvae.train` and `dynamics.train`; it uses `EvalConfig` (from `wandb_config.json`) to decide what to run.
2. **Standalone** — `python -m vqvae.evaluate` and `python -m dynamics.evaluate` run full evaluation once (reconstruction grids, one-step/rollout metrics, optional heavy metrics).

---

## W&B Setup

Evaluation uses the same W&B environment variables and project as training. See **[TRAIN.MD](TRAIN.MD)** for `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, and `--no_wandb`.

### Step axis

All evaluation (and training) metrics use **Epochs** as the step axis. The logger attaches `Epochs` to every `log()` call so charts are aligned by epoch.

### Metric filtering

`wandb_config.json` can toggle metrics via `metric_logging`:

- `default_enabled` — global default for logging.
- `overrides` — exact metric key → `true`/`false`.
- `prefix_overrides` — prefix → `true`/`false` (e.g. `"Evaluation/": true`).

The `WandbLogger` is constructed with `metric_enabled_fn=eval_config.is_metric_enabled` so only enabled metrics are sent to W&B.

---

## Evaluation Configuration

Configuration is loaded from **`wandb_config.json`** (path can be overridden by `--wandb_config`). The evaluation layer uses `evaluation.config.load_eval_config()` and the typed `EvalConfig` wrapper.

### Key settings

| Field | Default | Description |
|-------|---------|-------------|
| `metrics_gather_frequency` | 5 | Run heavy metrics every N epochs (0 = never). |
| `metric_overrides` | `{}` | Per-metric `frequency`, `num_samples`, `min_samples`. |
| `rollout.short_horizons` | [1, 4, 8, 16, 32, 64] | Step indices for per-horizon rollout metrics. |
| `rollout.fvd_clip_lengths` | [16, 32, 64, 128, 256, 512] | Clip lengths for FVD. |
| `gradient_stats.enabled` | true | Log gradient statistics. |
| `gradient_stats.frequency` | 50 | Gradient stats every N steps (training). |
| `runtime_profiling.enabled` | true | Allow runtime profiling. |
| `runtime_profiling.frequency` | 1 | How often to run profiling (e.g. per epoch). |
| `artifacts.log_best` / `log_latest` | true | Upload best/latest checkpoints to W&B. |

### Per-metric overrides

Under `metric_overrides`, each key (e.g. `fid`, `fvd`, `optical_flow`, `action_metrics`, `failure_metrics`) can have:

- **`frequency`** — run this metric every N epochs (0 = skip).
- **`num_samples`** — cap for samples (e.g. FID 5000, FVD 1000).
- **`min_samples`** — minimum samples; if fewer, samples are repeated via `ensure_min_samples()`.

Heavy metrics are gated by `should_run_heavy_metrics(epoch)` (global cadence) and `should_run_metric(metric_key, epoch)` (per-metric frequency).

---

## Metrics We Log

All metric **names** are produced by `logger.metric_names.M` so they stay consistent across training and evaluation. Namespaces: **Optimization**, **Gradients**, **Validation**, **Evaluation**, **Runtime**.

### Optimization (training)

Logged by the training loops, not by the evaluator:

- `Optimization/Train Loss (...)` — e.g. Total, L1 Reconstruction, LPIPS, Commitment (VQ-VAE); Velocity MSE (dynamics).
- `Optimization/Val Loss (...)` — same loss names.
- `Optimization/Learning Rate`
- `Optimization/Codebook Usage (Train)` and `(Val)` — VQ-VAE only.
- `Optimization/Scheduled Sampling Probability` — dynamics only when rollout_length > 1.

### Gradients

Logged by training when `gradient_stats.enabled` and at `gradient_stats.frequency`:

- `Gradients/Gradient {stat} ({scope})` — stat in Norm, Mean, Std, Max; scope is module name or "Global".

### Validation (one-step and rollout)

Produced by the orchestrator and by the standalone eval scripts:

- **One-step (dynamics)**  
  - `Validation/One-Step MSE`, `Validation/One-Step MAE`, `Validation/One-Step PSNR`, `Validation/One-Step SSIM`, `Validation/One-Step LPIPS`  
  - Comparison of one predicted frame vs. ground truth after decoding.

- **Rollout at horizon h**  
  - `Validation/Rollout PSNR @ h`, `Validation/Rollout SSIM @ h`, `Validation/Rollout LPIPS @ h`  
  - For each `short_horizons` value (e.g. 1, 4, 8, 16, 32, 64), quality at step h.

- **Rollout mean over window**  
  - `Validation/Rollout Mean PSNR (1-H)`, same for SSIM and LPIPS, for H in {16, 64} (if sequence length allows).

Single-frame and rollout metrics use the same pixel/perceptual functions: MSE, MAE, PSNR, SSIM, LPIPS (see **Single-frame metrics** below).

### Evaluation (benchmark and codebook)

- **FID** — `Evaluation/FID`  
  - Frechet Inception Distance between real and fake image sets (VQ-VAE reconstructions or dynamics one-step/rollout decoded to RGB).  
  - Uses `torch_fidelity` via `evaluation._fidelity_helpers.fid_from_arrays`; requires `pip install torch-fidelity`.

- **FVD** — `Evaluation/FVD-{n}`  
  - Frechet Video Distance at clip length n (e.g. 16, 64, 512).  
  - Computed in `evaluation.video_metrics` using I3D or Inception features from `evaluation._fvd_i3d`.  
  - Only in dynamics evaluation (rollout vs. ground-truth video clips).

- **Codebook (VQ-VAE)**  
  - `Evaluation/Codebook Perplexity`  
  - `Evaluation/Codebook Utilization`  
  - From `evaluation.single_frame_metrics.compute_codebook_stats` over the validation set.

- **Temporal**  
  - `Evaluation/Flicker LPIPS (Mean)` and `(Std)` — LPIPS between consecutive generated frames (temporal stability).  
  - `Evaluation/Optical Flow EPE` — mean end-point error between real and generated optical flow (Farneback). Requires OpenCV.  
  - `Evaluation/Motion Magnitude Correlation` — Pearson correlation of per-frame motion magnitudes (real vs. generated).

- **Action / control**  
  - `Evaluation/Controllability Score` — correlation between commanded actions and observed frame-to-frame displacement.  
  - `Evaluation/Action Success Rate` — fraction of jump commands that produce a detectable visual change within a small lag.

- **Failure / collapse**  
  - `Evaluation/Collapse Rate @ H` — fraction of rollouts collapsed by step H (H in e.g. 64, 128, 256).  
  - `Evaluation/Average Time-to-Collapse` — mean steps until first collapse.

### Runtime

- `Runtime/FPS` — inference frames per second.
- `Runtime/Frame Time (ms)` — average time per frame.
- `Runtime/VRAM Peak (GB)` and `Runtime/VRAM Average (GB)` — GPU memory (when CUDA).

These come from `evaluation.runtime_metrics.RuntimeProfiler` / `profile_inference`. The orchestrator exposes `run_runtime_profiling()`; standalone dynamics eval runs a short inference loop and logs the same metrics.

---

## How evaluation runs

### 1. VQ-VAE training (per epoch)

- **Validation loss** — computed by the training script (reconstruction + commitment, etc.).
- **Orchestrator** — `EvalOrchestrator.run_vqvae_epoch_eval(model, val_loader, epoch, global_step)`:
  - Iterates validation batches, runs `model(x)` → reconstruction.
  - **Every epoch:** single-frame metrics (MSE, MAE, PSNR, SSIM, LPIPS) on recon vs. input. MSE and MAE are logged as `Optimization/Val Loss (MSE)` and `(MAE)`; PSNR, SSIM, LPIPS as `Evaluation/PSNR`, `Evaluation/SSIM`, `Evaluation/LPIPS`.
  - **Heavy (when `should_run_heavy_metrics(epoch)`):**  
    - Codebook stats (perplexity, utilization) over the val loader.  
    - If `should_run_metric("fid", epoch)`: collect real/fake images, then `compute_fid(real, fake)` and log `Evaluation/FID`.
- Reconstruction grid is saved and logged by the training script; the orchestrator does not duplicate it.

### 2. Dynamics training (per epoch)

- **Validation loss** — velocity MSE from the training script.
- **Orchestrator** — `EvalOrchestrator.run_dynamics_epoch_eval(model, vqvae_model, val_loader, epoch, global_step, codebook=..., max_one_step_batches=...)`:
  - **One-step:** for a cap of batches, context + action → `predict_next_frame` → decode pred and GT → single-frame metrics; logged as `Validation/One-Step *`.
  - **Heavy + FID:** if enabled, collect one-step real/fake images and log `Evaluation/FID`.
  - **Heavy rollout:** `_run_heavy_dynamics_eval`:
    - Builds rollout batches (latent sequences) with `_rollout_latent_batch` for horizons from `short_horizons` and `fvd_clip_lengths`.
    - For each rollout vs. GT: **rollout_metrics** (PSNR/SSIM/LPIPS at each horizon, mean over 16/64), **temporal_metrics** (flicker, motion correlation, optional optical flow EPE).
    - **FVD:** when `should_run_metric("fvd", epoch)`, decodes latent rollouts to RGB, truncates to clip lengths, runs `evaluate_fvd_at_clip_lengths` → `Evaluation/FVD-{n}`.
    - **Failure metrics:** when `should_run_metric("failure_metrics", epoch)`, collapse rate at configured horizons and average time-to-collapse.
    - **Action metrics:** when `should_run_metric("action_metrics", epoch)`, controllability score and action success rate.
- All results are passed to `logger.log(metrics, step=global_step)`.

### 3. Standalone VQ-VAE evaluation

- `python -m vqvae.evaluate` loads checkpoint and val loader, creates a `WandbLogger` with `metric_enabled_fn` from `load_eval_config(wandb_config)`.
- **Reconstruction grid** — saved to disk and logged as `Evaluation/Reconstruction Grid`.
- **Single-frame metrics** — averaged over val set; logged under `Evaluation/*` (MSE, MAE, PSNR, SSIM, LPIPS).
- **Codebook stats** — full validation set; logs `Evaluation/Codebook Perplexity` and `Evaluation/Codebook Utilization`; optional histogram image `Evaluation/Codebook Usage Histogram`.
- **FID** — only if `--heavy_metrics`; uses same `metric_num_samples` / `metric_min_samples` from config.
- Optional: `--full_res` saves 840×420 PNGs; architecture is logged as an artifact.

### 4. Standalone dynamics evaluation

- `python -m dynamics.evaluate` loads dynamics + VQ-VAE, codebook, and latent val loader.
- **One-step metrics** — same as orchestrator; logged as `Validation/One-Step *`.
- **Rollouts** — generated for `num_eval_sequences` sequences; **rollout metrics** (per horizon + mean) and **temporal metrics** (flicker, motion correlation, optical flow if `--heavy_metrics`).
- **Action metrics** and **failure metrics** — when `--heavy_metrics`.
- **Runtime** — short inference loop with `RuntimeProfiler`, logs FPS, frame time, VRAM.
- FVD is not run in the current standalone script; it is only run inside the orchestrator’s heavy dynamics eval.

---

## Single-frame and rollout metric definitions

- **MSE / MAE** — pixel-wise mean squared / absolute error (inputs in [-1, 1]).
- **PSNR** — `10 * log10(4 / mse)` (dynamic range 2).
- **SSIM** — structural similarity (gaussian window, default size 11); in `evaluation.single_frame_metrics.compute_ssim`.
- **LPIPS** — VGG-based perceptual distance; cached model in `single_frame_metrics`, inputs in [-1, 1].

Rollout metrics apply these between generated and ground-truth frames at each horizon (and, for mean metrics, over the first 16 or 64 steps). Latent rollouts are decoded with the VQ-VAE decoder before computing pixel/perceptual metrics.

---

## Logging flow

1. **Config** — `load_eval_config(path)` → `EvalConfig` (defaults + `wandb_config.json`).
2. **Logger** — `WandbLogger(config=..., metric_enabled_fn=eval_config.is_metric_enabled)`. All `log()` / `log_image()` calls include the step (Epochs) and are filtered by `is_metric_enabled`.
3. **Orchestrator** — receives `EvalConfig`, `WandbLogger`, device, and optional kwargs (eval_chunk_size, num_eval_sequences, eval_ode_steps). It runs the right subset of metrics and passes a single dict to `logger.log(metrics, step=step)`.
4. **Naming** — All keys use `logger.metric_names.M` (e.g. `M.fid()`, `M.fvd(n)`, `M.one_step("PSNR")`, `M.inference_at("LPIPS", h)`) so dashboards get consistent names and namespaces.

---

## Optional dependencies

- **FID** — `pip install torch-fidelity`
- **Optical flow EPE** — `pip install opencv-python`
- **Codebook histogram (standalone)** — `pip install matplotlib`

---

## File reference

| Path | Role |
|------|------|
| `evaluation/config.py` | Load/merge `wandb_config.json`, `EvalConfig`, per-metric frequency/samples. |
| `evaluation/orchestrator.py` | `EvalOrchestrator`: VQ-VAE epoch eval, dynamics epoch eval, heavy rollout/FVD/failure/action, runtime profiling. |
| `evaluation/single_frame_metrics.py` | MSE, MAE, PSNR, SSIM, LPIPS; FID wrapper; codebook perplexity/utilization. |
| `evaluation/rollout_metrics.py` | Per-horizon and mean PSNR/SSIM/LPIPS for rollout vs. GT. |
| `evaluation/temporal_metrics.py` | Flicker (LPIPS), optical flow EPE, motion magnitude correlation. |
| `evaluation/action_metrics.py` | Controllability score, action success rate. |
| `evaluation/failure_metrics.py` | Collapse detection, collapse rate @ H, average time-to-collapse. |
| `evaluation/video_metrics.py` | FVD at multiple clip lengths; uses `_fvd_i3d`. |
| `evaluation/_fvd_i3d.py` | I3D or Inception feature extraction, Frechet distance. |
| `evaluation/_fidelity_helpers.py` | FID from in-memory arrays via torch-fidelity. |
| `evaluation/runtime_metrics.py` | RuntimeProfiler, profile_inference (FPS, frame time, VRAM). |
| `logger/metric_names.py` | Central metric name strings (namespaces + M.*). |
| `logger/wandb_logger.py` | W&B init, log, log_image, log_architecture, metric filtering. |

For full training CLI options and W&B setup, see **[TRAIN.MD](TRAIN.MD)**.
