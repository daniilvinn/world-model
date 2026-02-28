"""
Runtime performance metrics.

Covers: FPS, frame time (ms), VRAM peak and average usage.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import torch

from logger.metric_names import M


class RuntimeProfiler:
    """
    Lightweight profiler for measuring inference FPS and VRAM usage.

    Usage::

        profiler = RuntimeProfiler(device)
        profiler.start()
        for _ in range(N):
            model(input)
            profiler.tick()
        results = profiler.finish()
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._timestamps: list[float] = []
        self._vram_samples: list[float] = []
        self._start_time: Optional[float] = None

    def start(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
            torch.cuda.synchronize(self._device)
        self._timestamps = []
        self._vram_samples = []
        self._start_time = time.perf_counter()

    def tick(self) -> None:
        """Record one frame completion."""
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._timestamps.append(time.perf_counter())

        if self._device.type == "cuda":
            vram_gb = torch.cuda.memory_allocated(self._device) / (1024 ** 3)
            self._vram_samples.append(vram_gb)

    def finish(self) -> Dict[str, float]:
        """
        Compute FPS, frame time, and VRAM stats from collected samples.
        """
        results: Dict[str, float] = {}

        if len(self._timestamps) >= 2:
            total_time = self._timestamps[-1] - self._timestamps[0]
            n_frames = len(self._timestamps) - 1
            fps = n_frames / max(total_time, 1e-9)
            avg_frame_ms = (total_time / n_frames) * 1000
        elif self._start_time is not None and self._timestamps:
            total_time = self._timestamps[-1] - self._start_time
            fps = 1.0 / max(total_time, 1e-9)
            avg_frame_ms = total_time * 1000
        else:
            fps = 0.0
            avg_frame_ms = 0.0

        results[M.fps()] = fps
        results[M.frame_time_ms()] = avg_frame_ms

        if self._vram_samples:
            results[M.vram_avg_gb()] = sum(self._vram_samples) / len(self._vram_samples)

        if self._device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(self._device) / (1024 ** 3)
            results[M.vram_peak_gb()] = peak_gb

        return results


@torch.no_grad()
def profile_inference(
    model: torch.nn.Module,
    generate_fn: Callable,
    num_frames: int,
    device: torch.device,
    warmup: int = 5,
) -> Dict[str, float]:
    """
    Profile model inference over ``num_frames`` generation steps.

    Args:
        model: The dynamics model.
        generate_fn: Callable that runs one generation step (no args).
        num_frames: Number of frames to profile.
        device: Compute device.
        warmup: Warmup iterations (not counted).

    Returns:
        Dict of runtime metrics.
    """
    model.eval()

    for _ in range(warmup):
        generate_fn()

    profiler = RuntimeProfiler(device)
    profiler.start()

    for _ in range(num_frames):
        generate_fn()
        profiler.tick()

    return profiler.finish()
