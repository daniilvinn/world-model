# Training Guide

This document describes the full training pipeline for the World Model project, including all CLI options and recommended settings.

---

## Overview

The pipeline has five stages:

1. **Data collection** — Collect frame pairs from the game
2. **VQ-VAE training** — Train image compression model
3. **Latent precomputation** — Encode dataset to latent indices
4. **Dynamics training** — Train world model (flow matching)
5. **Playback** — Run the AI-driven game

Approximate timings on RTX 5070 Ti: data collection ~5–6 min (100K frames), VQ-VAE ~2–3 h, precompute ~10–20 min, dynamics ~4–6 h.

---

## 1. Data Collection

Collect frame pairs `(frame_t0, frame_t1, action, ...)` from the headless game.

### Command

```bash
python collect_dataset.py [OPTIONS]
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_frames` | 10000 | Number of frame pairs to collect |
| `--play_mode` | mixed | `optimal`, `random`, `noisy`, `mixed` |
| `--terrain_mode` | balanced | `default`, `obstacle_rich`, `balanced` |
| `--random_jump_prob` | 0.15 | Probability of random jumps (random mode) |
| `--miss_prob` | 0.10 | Probability of missing jumps (noisy mode) |
| `--phased` | — | Phased collection (40% optimal, 30% random, 30% noisy) |

### Play Modes

- **optimal** — Best-effort play (fewer deaths, higher scores)
- **random** — Random jumps with `random_jump_prob`
- **noisy** — Optimal with `miss_prob` chance of missing jumps
- **mixed** — Randomly pick mode per episode

### Terrain Modes

- **default** — Standard obstacle density
- **obstacle_rich** — More obstacles
- **balanced** — Mix of densities

### Phased Collection

Use `--phased` for diverse data:

- Phase A: 40% optimal / default terrain
- Phase B: 30% random / balanced terrain
- Phase C: 30% noisy / obstacle_rich terrain

### Examples

```bash
# Default: 10K frames, mixed play, balanced terrain
python collect_dataset.py

# Larger dataset with phased collection
python collect_dataset.py --num_frames 100000 --phased

# Optimal-only for cleaner data
python collect_dataset.py --num_frames 50000 --play_mode optimal --terrain_mode default
```

---

## 2. VQ-VAE Training

Train the image compression model on collected frames.

### Command

```bash
python -m vqvae.train [OPTIONS]
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| **Data** | | |
| `--data_dir` | dataset | Directory with `session_*/pair_*.npz` |
| `--num_workers` | 4 | DataLoader workers |
| `--max_samples` | None | Max samples (None = all) |
| **Model** | | |
| `--latent_dim` | 16 | Latent channel dimension |
| `--num_embeddings` | 1024 | Codebook size |
| `--commitment_cost` | 0.25 | Commitment loss weight |
| `--ema_decay` | 0.99 | EMA decay for codebook |
| **Training** | | |
| `--epochs` | 50 | Training epochs |
| `--batch_size` | 6 | Batch size |
| `--lr` | 3e-4 | Peak learning rate |
| `--min_lr` | 1e-5 | Min LR (cosine decay end) |
| `--weight_decay` | 1e-6 | AdamW weight decay |
| `--warmup_steps` | 500 | Linear warmup steps |
| `--patience` | 10 | Early stopping patience |
| **Logging** | | |
| `--log_every` | 50 | TensorBoard log interval |
| `--log_dir` | runs/vqvae | TensorBoard log dir |
| `--checkpoint_dir` | checkpoints | Checkpoint save dir |
| `--recon_dir` | reconstructions | Reconstruction grid dir |
| **Resume** | | |
| `--resume` | None | Path to checkpoint to resume |

### Examples

```bash
# Default training
python -m vqvae.train

# Larger batch, more epochs
python -m vqvae.train --epochs 100 --batch_size 12

# Resume from checkpoint
python -m vqvae.train --resume checkpoints/vqvae_latest.pt

# Subset for quick test
python -m vqvae.train --max_samples 1000 --epochs 5
```

### Monitoring

```bash
tensorboard --logdir runs/
```

### Outputs

- `checkpoints/vqvae_best.pt` — Best validation loss
- `checkpoints/vqvae_latest.pt` — Latest checkpoint
- `reconstructions/recon_epoch_*.png` — Validation reconstruction grids

---

## 3. Latent Precomputation

Encode all frames to VQ-VAE latent indices for efficient dynamics training.

### Command

```bash
python precompute_latents.py [OPTIONS]
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | dataset | Dataset root |
| `--checkpoint` | checkpoints/vqvae_best.pt | VQ-VAE checkpoint |
| `--output_dir` | latents | Output directory |
| `--batch_size` | 64 | Batch size |
| `--num_seeds` | 20 | Number of seed sequences |
| `--context_length` | 4 | Frames per seed |
| `--overwrite` | — | Overwrite existing files |

### Examples

```bash
# Default
python precompute_latents.py

# Custom paths
python precompute_latents.py --data_dir dataset --checkpoint checkpoints/vqvae_best.pt --output_dir latents

# Re-run and overwrite
python precompute_latents.py --overwrite
```

### Outputs

- `latents/session_*.npz` — Encoded frames per session
- `latents/seeds.pt` — 20 seed sequences for `play_world_model.py`

---

## 4. Dynamics Model Training

Train the flow-matching world model on latent sequences.

### Command

```bash
python -m dynamics.train [OPTIONS]
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| **Data** | | |
| `--latents_dir` | latents | Precomputed latents root |
| `--vqvae_checkpoint` | checkpoints/vqvae_best.pt | VQ-VAE checkpoint (codebook) |
| `--num_workers` | 4 | DataLoader workers |
| **Model** | | |
| `--context_length` | 4 | Context frames |
| `--base_channels` | 128 | Base channel dimension |
| `--channel_mults` | 1 2 2 | Channel multipliers |
| `--cond_dim` | 256 | Conditioning dimension |
| `--num_actions` | 2 | Action classes |
| `--attn_resolution` | 8 | Self-attention resolution |
| **Training** | | |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 3e-4 | Peak learning rate |
| `--min_lr` | 1e-5 | Min LR (cosine decay) |
| `--weight_decay` | 1e-4 | AdamW weight decay |
| `--warmup_steps` | 1000 | Linear warmup steps |
| `--patience` | 15 | Early stopping patience |
| `--grad_clip` | 1.0 | Gradient clipping max norm |
| **Flow matching** | | |
| `--diffusion_forcing` | True | Context corruption |
| `--max_context_noise` | 0.2 | Max context noise level |
| **Rollout & SS** | | |
| `--rollout_length` | 1 | Target frames (1–4) |
| `--rollout_mode` | fast | `fast` or `full_ode` |
| `--rollout_ode_steps` | 3 | ODE steps for rollout |
| `--ss_start_epoch` | 20 | Scheduled sampling start |
| `--ss_max_prob` | 0.5 | Max scheduled sampling prob |
| **EMA** | | |
| `--ema_decay` | 0.999 | EMA decay |
| **Logging** | | |
| `--log_every` | 50 | TensorBoard log interval |
| `--log_dir` | runs/dynamics | TensorBoard log dir |
| `--checkpoint_dir` | checkpoints | Checkpoint save dir |
| **Resume** | | |
| `--resume` | None | Path to checkpoint to resume |

### Examples

```bash
# Default training
python -m dynamics.train

# Reduce batch if OOM
python -m dynamics.train --batch_size 32

# Multi-step rollout
python -m dynamics.train --rollout_length 2 --rollout_ode_steps 5

# Resume
python -m dynamics.train --resume checkpoints/dynamics_latest.pt
```

### Outputs

- `checkpoints/dynamics_best.pt` — Best validation loss
- `checkpoints/dynamics_latest.pt` — Latest checkpoint

---

## 5. Playback (AI-Driven Game)

Run the fully neural game.

### Command

```bash
python play_world_model.py [OPTIONS]
```

### CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--vqvae_checkpoint` | checkpoints/vqvae_best.pt | VQ-VAE checkpoint |
| `--dynamics_checkpoint` | checkpoints/dynamics_best.pt | Dynamics checkpoint |
| `--seeds_path` | latents/seeds.pt | Seed sequences |
| `--ode_steps` | 10 | ODE integration steps |
| `--solver` | euler | `euler` or `midpoint` |
| `--seed_index` | None | Fixed seed index (None = random) |
| `--width` | 840 | Display width |
| `--height` | 420 | Display height |
| `--no_compile` | — | Disable torch.compile |

### Controls

- **SPACE** — Jump
- **R** — Restart (new seed)
- **ESC** — Quit

### Examples

```bash
# Default
python play_world_model.py

# Higher quality (slower)
python play_world_model.py --ode_steps 15 --solver midpoint

# Specific seed
python play_world_model.py --seed_index 5

# Disable torch.compile (if issues on Windows)
python play_world_model.py --no_compile
```

---

## Additional Scripts

### VQ-VAE Evaluation

```bash
python -m vqvae.evaluate [OPTIONS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | checkpoints/vqvae_best.pt | VQ-VAE checkpoint |
| `--data_dir` | dataset | Dataset root |
| `--output_dir` | eval_output | Output dir |
| `--num_images` | 8 | Number of images |
| `--batch_size` | 16 | Batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--full_res` | — | Also save 840×420 reconstructions |

### Original Game with VQ-VAE Overlay

```bash
python game.py --vqvae
python game.py --vqvae --checkpoint path/to/vqvae.pt
```

---

## Troubleshooting

### Out of Memory (OOM)

- **VQ-VAE:** Reduce `--batch_size` (e.g. 4)
- **Dynamics:** Reduce `--batch_size` (e.g. 32)

### Checkpoint Paths

Ensure paths match: `--data_dir`, `--checkpoint`, `--latents_dir`, `--vqvae_checkpoint`, etc. Defaults assume `dataset/`, `checkpoints/`, `latents/`.

### Windows + torch.compile

On Windows, the dynamics script uses `backend="eager"` to avoid Triton issues. Use `--no_compile` if you see compile errors.

### Early Stopping

Both VQ-VAE and dynamics use early stopping. Adjust `--patience` to train longer or stop sooner.
