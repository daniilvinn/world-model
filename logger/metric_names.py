"""
Central metric-name registry for W&B logging.

Every logged metric MUST go through this module so that:
  - Folder structure (Optimization / Gradients / Validation / Evaluation / Runtime) is consistent.
  - Plot titles are human-readable (e.g. "Loss (LPIPS)", not "lpips_loss").
  - Parametric names (horizons, clip lengths) are formatted uniformly.

Usage::

    from logger.metric_names import M

    logger.log({M.train_loss("Total"): 0.42}, step=step)
    logger.log({M.gradient("Norm", "Global"): 1.23}, step=step)
    logger.log({M.inference_at("PSNR", 16): 28.5}, step=step)
"""


class Namespace:
    OPTIMIZATION = "Optimization"
    GRADIENTS = "Gradients"
    VALIDATION = "Validation"
    EVALUATION = "Evaluation"
    RUNTIME = "Runtime"


class M:
    """Metric name formatters.  All return ``str``."""

    # ----- Optimization / Training ------------------------------------------

    @staticmethod
    def train_loss(name: str) -> str:
        return f"{Namespace.OPTIMIZATION}/Train Loss ({name})"

    @staticmethod
    def val_loss(name: str) -> str:
        return f"{Namespace.OPTIMIZATION}/Val Loss ({name})"

    @staticmethod
    def learning_rate() -> str:
        return f"{Namespace.OPTIMIZATION}/Learning Rate"

    @staticmethod
    def codebook_usage(split: str = "Train") -> str:
        return f"{Namespace.OPTIMIZATION}/Codebook Usage ({split})"

    @staticmethod
    def scheduled_sampling_prob() -> str:
        return f"{Namespace.OPTIMIZATION}/Scheduled Sampling Probability"

    # ----- Gradient statistics ----------------------------------------------

    @staticmethod
    def gradient(stat: str, scope: str, model: str | None = None) -> str:
        """
        ``stat`` in {Norm, Mean, Std, Max}; ``scope`` e.g. 'Global', 'Encoder'.
        ``model`` is accepted for backward-call compatibility but ignored:
        gradients are run-scoped and always grouped under ``Gradients/...``.
        """
        return f"{Namespace.GRADIENTS}/Gradient {stat} ({scope})"

    # ----- Validation (single-step & rollout) -------------------------------

    @staticmethod
    def one_step(metric: str) -> str:
        return f"{Namespace.VALIDATION}/One-Step {metric}"

    @staticmethod
    def inference_at(metric: str, h: int) -> str:
        return f"{Namespace.VALIDATION}/Rollout {metric} @ {h}"

    @staticmethod
    def inference_mean(metric: str, H: int) -> str:
        return f"{Namespace.VALIDATION}/Rollout Mean {metric} (1-{H})"

    @staticmethod
    def token_entropy_at(h: int) -> str:
        return f"{Namespace.VALIDATION}/Token Entropy @ {h}"

    @staticmethod
    def codebook_perplexity_at(h: int) -> str:
        return f"{Namespace.VALIDATION}/Codebook Perplexity @ {h}"

    @staticmethod
    def nll_over(H: int) -> str:
        return f"{Namespace.VALIDATION}/NLL (1-{H})"

    # ----- Evaluation (heavy / periodic) ------------------------------------

    @staticmethod
    def fid() -> str:
        return f"{Namespace.EVALUATION}/FID"

    @staticmethod
    def kid() -> str:
        return f"{Namespace.EVALUATION}/KID"

    @staticmethod
    def fvd(n: int) -> str:
        return f"{Namespace.EVALUATION}/FVD-{n}"

    @staticmethod
    def ssim_component(component: str) -> str:
        """``component`` in {l, c, s}."""
        return f"{Namespace.EVALUATION}/SSIM ({component} Component)"

    @staticmethod
    def codebook_perplexity() -> str:
        return f"{Namespace.EVALUATION}/Codebook Perplexity"

    @staticmethod
    def codebook_utilization() -> str:
        return f"{Namespace.EVALUATION}/Codebook Utilization"

    @staticmethod
    def token_accuracy() -> str:
        return f"{Namespace.EVALUATION}/Token Accuracy"

    @staticmethod
    def cross_entropy_latent() -> str:
        return f"{Namespace.EVALUATION}/Cross-Entropy (Latent)"

    @staticmethod
    def kl_divergence_tokens() -> str:
        return f"{Namespace.EVALUATION}/KL Divergence (Token Distribution)"

    @staticmethod
    def js_divergence_tokens() -> str:
        return f"{Namespace.EVALUATION}/JS Divergence (Token Distribution)"

    @staticmethod
    def flicker(stat: str) -> str:
        """``stat`` in {Mean, Std}."""
        return f"{Namespace.EVALUATION}/Flicker LPIPS ({stat})"

    @staticmethod
    def optical_flow_epe() -> str:
        return f"{Namespace.EVALUATION}/Optical Flow EPE"

    @staticmethod
    def motion_magnitude_corr() -> str:
        return f"{Namespace.EVALUATION}/Motion Magnitude Correlation"

    @staticmethod
    def error_growth_rate(metric: str) -> str:
        return f"{Namespace.EVALUATION}/Error Growth Rate ({metric})"

    @staticmethod
    def action_recognition_accuracy() -> str:
        return f"{Namespace.EVALUATION}/Action Recognition Accuracy"

    @staticmethod
    def controllability_score() -> str:
        return f"{Namespace.EVALUATION}/Controllability Score"

    @staticmethod
    def action_response_latency() -> str:
        return f"{Namespace.EVALUATION}/Action Response Latency (frames)"

    @staticmethod
    def action_success_rate() -> str:
        return f"{Namespace.EVALUATION}/Action Success Rate"

    @staticmethod
    def collapse_rate(H: int) -> str:
        return f"{Namespace.EVALUATION}/Collapse Rate @ {H}"

    @staticmethod
    def avg_time_to_collapse() -> str:
        return f"{Namespace.EVALUATION}/Average Time-to-Collapse"

    @staticmethod
    def precision() -> str:
        return f"{Namespace.EVALUATION}/Precision"

    @staticmethod
    def recall() -> str:
        return f"{Namespace.EVALUATION}/Recall"

    @staticmethod
    def codebook_coverage_long() -> str:
        return f"{Namespace.EVALUATION}/Codebook Coverage (Long Rollout)"

    @staticmethod
    def codebook_kl_long() -> str:
        return f"{Namespace.EVALUATION}/Codebook KL (Long Rollout)"

    # ----- Runtime ----------------------------------------------------------

    @staticmethod
    def fps() -> str:
        return f"{Namespace.RUNTIME}/FPS"

    @staticmethod
    def frame_time_ms() -> str:
        return f"{Namespace.RUNTIME}/Frame Time (ms)"

    @staticmethod
    def vram_peak_gb() -> str:
        return f"{Namespace.RUNTIME}/VRAM Peak (GB)"

    @staticmethod
    def vram_avg_gb() -> str:
        return f"{Namespace.RUNTIME}/VRAM Average (GB)"
