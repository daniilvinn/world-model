"""Central metric-name registry for W&B logging."""


class Namespace:
    OPTIMIZATION = "Optimization"
    GRADIENTS = "Gradients"
    VALIDATION = "Validation"
    EVALUATION = "Evaluation"
    RUNTIME = "Runtime"


class M:
    """Metric name formatters."""

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

    @staticmethod
    def gradient(stat: str, scope: str, model: str | None = None) -> str:
        # model is kept for backward-call compatibility.
        return f"{Namespace.GRADIENTS}/Gradient {stat} ({scope})"

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
    def fid() -> str:
        return f"{Namespace.EVALUATION}/FID"

    @staticmethod
    def fvd(n: int) -> str:
        return f"{Namespace.EVALUATION}/FVD-{n}"

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
    def controllability_score() -> str:
        return f"{Namespace.EVALUATION}/Controllability Score"

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
