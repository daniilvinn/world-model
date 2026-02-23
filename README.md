# World Model

A neural world model that learns to predict game dynamics from visual observations and actions. Train a **VQ-VAE** for image compression and a **flow-matching dynamics model** to predict the next frame—then play the game entirely through neural networks, no physics engine / render engine required.

---

## 🎯 What's the goal?

Build a neural world model that can play a platformer-style game purely from learned representations:

- No physics simulation
- No collision detection  
- No level generation

The model observes a sequence of frames and your action, predicts the next frame in latent space, decodes it to pixels, and repeats. Everything runs through neural network forward passes.

**How it works:** Compress game frames into discrete latent codes (VQ-VAE), then train a conditional generative model (flow matching) to predict the next latent from context + action. At inference, the model generates the game frame by frame.

---

## 📁 Project structure

```
World/
├── game.py                 # Core game engine (GD-style platformer)
│                           #   - Pygame + Pymunk physics
│                           #   - Optional VQ-VAE overlay (--vqvae)
│
├── collect_dataset.py      # Data collection: auto-play, save frame pairs
├── precompute_latents.py   # Encode dataset → VQ-VAE latent indices
├── play_world_model.py     # AI-driven game (no physics, pure NN)
├── quickstart.py           # Workflow guide / next-step checker
│
├── requirements.txt        # Dependencies
│
├── vqvae/                  # VQ-VAE image compression
│   ├── model.py            # VQVAE, Encoder, Decoder, VectorQuantizerEMA
│   ├── train.py            # Training script
│   ├── dataset.py          # GameFrameDataset
│   ├── losses.py           # PerceptualLoss (LPIPS), compute_loss
│   └── evaluate.py         # Reconstruction grid, codebook stats
│
├── dynamics/               # World model (flow matching)
│   ├── model.py            # DynamicsUNet (conditional U-Net)
│   ├── train.py            # Training script
│   ├── dataset.py          # LatentSequenceDataset
│   └── inference.py        # predict_next_frame, quantize_latent, rollout
│
└── (generated at runtime)
    ├── dataset/            # session_YYYYMMDD_HHMMSS/pair_*.npz
    ├── latents/            # session_*.npz, seeds.pt
    ├── checkpoints/       # vqvae_best.pt, dynamics_best.pt
    ├── runs/               # TensorBoard logs
    └── reconstructions/   # VQ-VAE reconstruction grids
```

---

## 🧠 Models at a glance

### VQ-VAE (image compression)

| Property | Value |
|----------|-------|
| **Input** | `[B, 3, 256, 512]` game frames (2:1 aspect) |
| **Latent** | `[B, 16, 32, 32]` quantized via 1024-entry codebook |
| **Output** | `[B, 3, 256, 512]` reconstruction |
| **Architecture** | Encoder (4 down blocks) → VectorQuantizerEMA → Decoder (4 up blocks) |
| **Features** | ResBlocks, self-attention at 32×32, asymmetric strides for 2:1 input |
| **Loss** | L1 + 0.1×LPIPS + 0.25×commitment |

### Dynamics model (flow matching)

| Property | Value |
|----------|-------|
| **Input** | Noisy latent `[B, 16, 32, 32]` + context `[B, ctx_len×16, 32, 32]` |
| **Context** | 4 frames (default) |
| **Conditioning** | Flow time (sinusoidal) + action (embedding) |
| **Output** | Velocity field `[B, 16, 32, 32]` |
| **Architecture** | U-Net with AdaGN blocks, self-attention at 8×8 |
| **Training** | Flow matching on OT path (x_t = (1-t)x_0 + tx_1), velocity MSE loss |
| **Optional** | Diffusion forcing (context corruption), rollout training, scheduled sampling |

---

## 🔄 Pipeline overview

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `collect_dataset.py` | Auto-play headless game, save frame pairs as `.npz` |
| 2 | `python -m vqvae.train` | Compress frames to latent space |
| 3 | `precompute_latents.py` | Encode all frames to indices, create `latents/` and `seeds.pt` |
| 4 | `python -m dynamics.train` | Train flow-matching model on latent sequences |
| 5 | `play_world_model.py` | Autoregressive generation: context → dynamics → decode → display |

For a step-by-step training guide with all CLI options, see **[TRAIN.MD](TRAIN.MD)**.

---

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quickstart to see what to do next
python quickstart.py
```

The quickstart script checks your current progress and suggests the next step (collect data, train VQ-VAE, precompute latents, train dynamics, or play).

---

## 📦 Dependencies

| File | Purpose |
|------|---------|
| `requirements.txt` | Base: `pygame`, `numpy`, `pymunk` (game only) |
| `requirements_vqvae.txt` | Full: `torch`, `torchvision`, `lpips`, `tensorboard`, `matplotlib`, etc. |

---

## License

MIT (see [LICENSE](LICENSE)).
