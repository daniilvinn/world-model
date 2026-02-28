"""
Centralized Weights & Biases logger for Neural Dash.

Wraps all ``wandb`` interactions so that training and evaluation scripts
never call ``wandb`` directly.  Provides:

* Run lifecycle management (init / finish)
* Scalar, image, video, table logging with enforced metric naming
* Gradient statistics logging (global + per-module)
* Model architecture JSON artifact upload
* Checkpoint / artifact management
* Runtime statistics logging

Identity fields (project, entity, group, tags) are read from environment
variables; evaluation cadence comes from ``wandb_config.json``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

try:
    import wandb
    from wandb.sdk.wandb_run import Run as WandbRun
except ImportError:
    wandb = None  # type: ignore[assignment]
    WandbRun = None  # type: ignore[assignment,misc]

from logger.architecture import save_architecture_json, serialize_model_architecture
from logger.gradient_stats import compute_gradient_stats


class WandbLogger:
    """
    High-level W&B logger.

    Parameters
    ----------
    config : dict
        Merged training / eval configuration to record against the run.
    run_name : str, optional
        Override for the W&B run display name.
    tags : list[str], optional
        Extra tags appended to any from ``WANDB_TAGS`` env var.
    enabled : bool
        Set to *False* to disable W&B entirely (dry-run mode).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[List[str]] = None,
        enabled: bool = True,
        metric_enabled_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._enabled = enabled and (wandb is not None)
        self._run: Optional[Any] = None
        self._metric_enabled_fn = metric_enabled_fn
        self._last_step: Optional[int] = None
        self._step_metric_key = "Epochs"

        if not self._enabled:
            if wandb is None and enabled:
                raise ImportError(
                    "wandb is not installed.  Run `pip install wandb` "
                    "or set enabled=False to skip W&B logging."
                )
            return

        env_tags = os.environ.get("WANDB_TAGS", "")
        all_tags = [t.strip() for t in env_tags.split(",") if t.strip()]
        if tags:
            all_tags.extend(tags)

        self._run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "neural-dash"),
            entity=os.environ.get("WANDB_ENTITY", None),
            group=group if group is not None else os.environ.get("WANDB_RUN_GROUP", None),
            name=run_name,
            config=config,
            tags=all_tags or None,
            reinit=True,
        )
        # Use a dedicated epoch axis for all charts so repeated epoch values
        # across multiple log() calls in the same epoch are valid.
        #self._run.define_metric(self._step_metric_key)
        self._run.define_metric("*", step_metric=self._step_metric_key)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run(self) -> Optional[Any]:
        return self._run

    @property
    def enabled(self) -> bool:
        return self._enabled and self._run is not None

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """Log a dictionary of scalars (or any wandb-compatible values)."""
        if not self.enabled:
            return
        filtered = self._filter_metrics(metrics)
        if not filtered:
            return
        payload = dict(filtered)
        effective_step = self._normalize_step(step)
        if effective_step is not None:
            payload[self._step_metric_key] = effective_step
        self._run.log(payload, commit=commit)

    def log_scalar(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        if not self._metric_enabled(key):
            return
        payload = {key: value}
        effective_step = self._normalize_step(step)
        if effective_step is not None:
            payload[self._step_metric_key] = effective_step
        self._run.log(payload)

    # ------------------------------------------------------------------
    # Media logging
    # ------------------------------------------------------------------

    def log_image(
        self,
        key: str,
        image: Any,
        step: Optional[int] = None,
        caption: Optional[str] = None,
        commit: bool = True,
    ) -> None:
        """Log a single image (PIL, numpy HWC uint8, or torch CHW float)."""
        if not self.enabled:
            return
        if not self._metric_enabled(key):
            return
        img = wandb.Image(image, caption=caption)
        payload = {key: img}
        effective_step = self._normalize_step(step)
        if effective_step is not None:
            payload[self._step_metric_key] = effective_step
        self._run.log(payload, commit=commit)

    def log_images(
        self,
        key: str,
        images: Sequence[Any],
        step: Optional[int] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> None:
        if not self.enabled:
            return
        if not self._metric_enabled(key):
            return
        if captions:
            imgs = [wandb.Image(im, caption=c) for im, c in zip(images, captions)]
        else:
            imgs = [wandb.Image(im) for im in images]
        payload = {key: imgs}
        effective_step = self._normalize_step(step)
        if effective_step is not None:
            payload[self._step_metric_key] = effective_step
        self._run.log(payload)

    def log_video(
        self,
        key: str,
        video_path: str,
        step: Optional[int] = None,
        caption: Optional[str] = None,
        fps: int = 30,
    ) -> None:
        if not self.enabled:
            return
        if not self._metric_enabled(key):
            return
        vid = wandb.Video(video_path, caption=caption, fps=fps)
        payload = {key: vid}
        effective_step = self._normalize_step(step)
        if effective_step is not None:
            payload[self._step_metric_key] = effective_step
        self._run.log(payload)

    def log_table(
        self,
        key: str,
        columns: List[str],
        data: List[List[Any]],
        step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        if not self._metric_enabled(key):
            return
        table = wandb.Table(columns=columns, data=data)
        payload = {key: table}
        effective_step = self._normalize_step(step)
        if effective_step is not None:
            payload[self._step_metric_key] = effective_step
        self._run.log(payload)

    # ------------------------------------------------------------------
    # Gradient statistics
    # ------------------------------------------------------------------

    def log_gradient_stats(
        self,
        model: nn.Module,
        step: int,
        module_names: Optional[List[str]] = None,
    ) -> None:
        """Compute and log gradient norm/mean/std/max (global + per-module)."""
        if not self.enabled:
            return
        stats = compute_gradient_stats(model, module_names)
        if stats:
            self.log(stats, step=step)

    # ------------------------------------------------------------------
    # Architecture JSON artifact
    # ------------------------------------------------------------------

    def log_architecture(
        self,
        model: nn.Module,
        model_name: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model architecture as a JSON artifact (not a plot)."""
        if not self.enabled:
            return

        arch = serialize_model_architecture(model, model_name, extra_metadata)
        artifact_name = f"{model_name.lower().replace(' ', '-')}-architecture"

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / f"{artifact_name}.json"
            json_path.write_text(json.dumps(arch, indent=2), encoding="utf-8")

            artifact = wandb.Artifact(artifact_name, type="model-architecture")
            artifact.add_file(str(json_path))
            self._run.log_artifact(artifact)

    # ------------------------------------------------------------------
    # Checkpoint / artifact management
    # ------------------------------------------------------------------

    def log_checkpoint(
        self,
        path: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upload a checkpoint file as a W&B artifact."""
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type="model-checkpoint", metadata=metadata)
        artifact.add_file(path)
        self._run.log_artifact(artifact)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def update_config(self, updates: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._run.config.update(updates)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        for k, v in summary.items():
            if not self._metric_enabled(k):
                continue
            self._run.summary[k] = v

    # ------------------------------------------------------------------
    # Metric filtering
    # ------------------------------------------------------------------

    def _metric_enabled(self, key: str) -> bool:
        if self._metric_enabled_fn is None:
            return True
        return bool(self._metric_enabled_fn(key))

    def _filter_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        if self._metric_enabled_fn is None:
            return metrics
        return {k: v for k, v in metrics.items() if self._metric_enabled(k)}

    def _normalize_step(self, step: Optional[int]) -> Optional[int]:
        """
        Normalize explicit user-provided steps while avoiding drift from
        W&B's internal run.step (which may advance independently).
        """
        if not self.enabled:
            return step

        if step is None:
            run_step = getattr(self._run, "step", None)
            run_step_int = int(run_step) if isinstance(run_step, int) else None
            if run_step_int is not None:
                self._last_step = run_step_int
                return run_step_int
            return self._last_step

        normalized = int(step)
        # Trust explicit step values (e.g. epoch-based axes), but keep local
        # monotonic tracking for optional step=None callers.
        if self._last_step is None or normalized > self._last_step:
            self._last_step = normalized
        return normalized

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(self) -> None:
        if self.enabled:
            self._run.finish()
            self._run = None
