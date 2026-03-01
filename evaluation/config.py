"""
Evaluation configuration loader and validator.

Reads ``wandb_config.json``, applies defaults for missing keys, and
validates types/ranges so that downstream code can rely on well-formed values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULTS: Dict[str, Any] = {
    "metrics_gather_frequency": 5,
    "metric_overrides": {},
    "rollout": {
        "short_horizons": [1, 4, 8, 16, 32, 64],
        "long_horizons": [64, 128, 192, 256, 512],
        "fvd_clip_lengths": [16, 32, 64, 128, 256, 512],
    },
    "gradient_stats": {
        "enabled": True,
        "frequency": 50,
        "modules": {
            "vqvae": ["encoder", "decoder", "quantizer"],
            "dynamics": [
                "cond_net",
                "encoder_blocks",
                "mid_block1",
                "mid_block2",
                "decoder_blocks",
            ],
        },
    },
    "runtime_profiling": {
        "enabled": True,
        "frequency": 1,
    },
    "artifacts": {
        "checkpoint_frequency": 1,
        "log_best": True,
        "log_latest": True,
    },
    "metric_logging": {
        "default_enabled": True,
        "overrides": {},
        "prefix_overrides": {},
    },
}


class EvalConfig:
    """Typed, validated wrapper around the raw JSON config dict."""

    def __init__(self, raw: Dict[str, Any]) -> None:
        self._raw = raw

    # -- Top-level -----------------------------------------------------------

    @property
    def metrics_gather_frequency(self) -> int:
        return int(self._raw.get("metrics_gather_frequency", _DEFAULTS["metrics_gather_frequency"]))

    # -- Per-metric overrides ------------------------------------------------

    def metric_frequency(self, metric_key: str) -> int:
        """Return the evaluation frequency for *metric_key*, falling back to the global default."""
        overrides = self._raw.get("metric_overrides", {})
        entry = overrides.get(metric_key, {})
        return int(entry.get("frequency", self.metrics_gather_frequency))

    def metric_num_samples(self, metric_key: str, default: int = 5000) -> int:
        overrides = self._raw.get("metric_overrides", {})
        entry = overrides.get(metric_key, {})
        return int(entry.get("num_samples", default))

    def metric_min_samples(self, metric_key: str, default: int = 1) -> int:
        overrides = self._raw.get("metric_overrides", {})
        entry = overrides.get(metric_key, {})
        return int(entry.get("min_samples", default))

    def should_run_heavy_metrics(self, epoch: int) -> bool:
        """True when ``epoch`` is a multiple of the global heavy-metric cadence."""
        freq = self.metrics_gather_frequency
        return freq > 0 and (epoch % freq == 0)

    def should_run_metric(self, metric_key: str, epoch: int) -> bool:
        freq = self.metric_frequency(metric_key)
        return freq > 0 and (epoch % freq == 0)

    # -- Rollout horizons ----------------------------------------------------

    @property
    def short_horizons(self) -> List[int]:
        return list(self._raw.get("rollout", _DEFAULTS["rollout"])["short_horizons"])

    @property
    def long_horizons(self) -> List[int]:
        return list(self._raw.get("rollout", _DEFAULTS["rollout"])["long_horizons"])

    @property
    def fvd_clip_lengths(self) -> List[int]:
        return list(self._raw.get("rollout", _DEFAULTS["rollout"])["fvd_clip_lengths"])

    # -- Gradient stats ------------------------------------------------------

    @property
    def gradient_stats_enabled(self) -> bool:
        return bool(self._raw.get("gradient_stats", _DEFAULTS["gradient_stats"])["enabled"])

    @property
    def gradient_stats_frequency(self) -> int:
        return int(self._raw.get("gradient_stats", _DEFAULTS["gradient_stats"])["frequency"])

    def gradient_modules(self, model_key: str) -> List[str]:
        gs = self._raw.get("gradient_stats", _DEFAULTS["gradient_stats"])
        return list(gs.get("modules", {}).get(model_key, []))

    # -- Runtime profiling ---------------------------------------------------

    @property
    def runtime_profiling_enabled(self) -> bool:
        return bool(self._raw.get("runtime_profiling", _DEFAULTS["runtime_profiling"])["enabled"])

    @property
    def runtime_profiling_frequency(self) -> int:
        return int(self._raw.get("runtime_profiling", _DEFAULTS["runtime_profiling"])["frequency"])

    # -- Artifacts -----------------------------------------------------------

    @property
    def checkpoint_frequency(self) -> int:
        return int(self._raw.get("artifacts", _DEFAULTS["artifacts"])["checkpoint_frequency"])

    @property
    def log_best_checkpoint(self) -> bool:
        return bool(self._raw.get("artifacts", _DEFAULTS["artifacts"])["log_best"])

    @property
    def log_latest_checkpoint(self) -> bool:
        return bool(self._raw.get("artifacts", _DEFAULTS["artifacts"])["log_latest"])

    # -- Metric logging toggles ---------------------------------------------

    @property
    def metric_logging_default_enabled(self) -> bool:
        cfg = self._raw.get("metric_logging", _DEFAULTS["metric_logging"])
        return bool(cfg.get("default_enabled", True))

    def is_metric_enabled(self, metric_key: str) -> bool:
        """
        Return whether a metric/media key should be logged.

        Resolution order:
        1) exact key in ``metric_logging.overrides``
        2) longest matching prefix in ``metric_logging.prefix_overrides``
        3) ``metric_logging.default_enabled``
        """
        cfg = self._raw.get("metric_logging", _DEFAULTS["metric_logging"])
        overrides = cfg.get("overrides", {})
        if metric_key in overrides:
            return bool(overrides[metric_key])

        prefix_overrides = cfg.get("prefix_overrides", {})
        best_prefix = ""
        best_value: Optional[bool] = None
        for prefix, enabled in prefix_overrides.items():
            if metric_key.startswith(prefix) and len(prefix) > len(best_prefix):
                best_prefix = prefix
                best_value = bool(enabled)
        if best_value is not None:
            return best_value

        return bool(cfg.get("default_enabled", True))

    # -- Raw access ----------------------------------------------------------

    @property
    def raw(self) -> Dict[str, Any]:
        return dict(self._raw)


def load_eval_config(path: Optional[str] = None) -> EvalConfig:
    """
    Load and validate evaluation config from a JSON file.

    Falls back to built-in defaults when *path* is ``None`` or the file is
    missing.
    """
    if path is None:
        path = "wandb_config.json"

    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = {}

    merged = _deep_merge(_DEFAULTS, raw)
    _validate(merged)
    return EvalConfig(merged)


def _deep_merge(defaults: Dict, overrides: Dict) -> Dict:
    """Recursively merge *overrides* into *defaults* (overrides win)."""
    result = dict(defaults)
    for key, val in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _validate(cfg: Dict[str, Any]) -> None:
    """Raise ``ValueError`` for obviously invalid config values."""
    freq = cfg.get("metrics_gather_frequency", 5)
    if not isinstance(freq, int) or freq < 0:
        raise ValueError(
            f"metrics_gather_frequency must be a non-negative int, got {freq!r}"
        )

    for key in ("short_horizons", "long_horizons", "fvd_clip_lengths"):
        vals = cfg.get("rollout", {}).get(key, [])
        if not isinstance(vals, list) or not all(isinstance(v, int) and v > 0 for v in vals):
            raise ValueError(f"rollout.{key} must be a list of positive ints, got {vals!r}")

    mo = cfg.get("metric_overrides", {})
    if not isinstance(mo, dict):
        raise ValueError("metric_overrides must be a dict")
    for metric_key, entry in mo.items():
        if not isinstance(metric_key, str) or not isinstance(entry, dict):
            raise ValueError("metric_overrides must map metric names to dicts")
        if "frequency" in entry:
            freq = entry["frequency"]
            if not isinstance(freq, int) or freq < 0:
                raise ValueError(
                    f"metric_overrides.{metric_key}.frequency must be a non-negative int, got {freq!r}"
                )
        if "num_samples" in entry:
            num = entry["num_samples"]
            if not isinstance(num, int) or num < 1:
                raise ValueError(
                    f"metric_overrides.{metric_key}.num_samples must be a positive int, got {num!r}"
                )
        if "min_samples" in entry:
            minimum = entry["min_samples"]
            if not isinstance(minimum, int) or minimum < 1:
                raise ValueError(
                    f"metric_overrides.{metric_key}.min_samples must be a positive int, got {minimum!r}"
                )

    gs = cfg.get("gradient_stats", {})
    gs_freq = gs.get("frequency", 50)
    if not isinstance(gs_freq, int) or gs_freq < 1:
        raise ValueError(f"gradient_stats.frequency must be a positive int, got {gs_freq!r}")

    ml = cfg.get("metric_logging", {})
    default_enabled = ml.get("default_enabled", True)
    if not isinstance(default_enabled, bool):
        raise ValueError(
            f"metric_logging.default_enabled must be a bool, got {default_enabled!r}"
        )

    overrides = ml.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("metric_logging.overrides must be a dict[str, bool]")
    for k, v in overrides.items():
        if not isinstance(k, str) or not isinstance(v, bool):
            raise ValueError(
                "metric_logging.overrides must map string keys to bool values"
            )

    prefix_overrides = ml.get("prefix_overrides", {})
    if not isinstance(prefix_overrides, dict):
        raise ValueError("metric_logging.prefix_overrides must be a dict[str, bool]")
    for k, v in prefix_overrides.items():
        if not isinstance(k, str) or not isinstance(v, bool):
            raise ValueError(
                "metric_logging.prefix_overrides must map string prefixes to bool values"
            )
