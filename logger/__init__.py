"""
Logging package for Neural Dash training and evaluation.

Provides a centralized WandbLogger, metric naming conventions,
gradient statistics collection, and model architecture serialization.
"""

from logger.metric_names import M, Namespace


def __getattr__(name):
    """Lazy imports for modules that depend on torch."""
    if name == "WandbLogger":
        from logger.wandb_logger import WandbLogger
        return WandbLogger
    if name == "compute_gradient_stats":
        from logger.gradient_stats import compute_gradient_stats
        return compute_gradient_stats
    if name == "serialize_model_architecture":
        from logger.architecture import serialize_model_architecture
        return serialize_model_architecture
    raise AttributeError(f"module 'logger' has no attribute {name!r}")


__all__ = [
    "WandbLogger",
    "M",
    "Namespace",
    "compute_gradient_stats",
    "serialize_model_architecture",
]
