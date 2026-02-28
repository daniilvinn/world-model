"""
Action-conditioned control metrics.

Covers: action recognition accuracy (from GT labels), controllability
score, action response latency, and action success rate.

All metrics leverage the availability of ground-truth action labels
rather than requiring a separate LAM / classifier.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from logger.metric_names import M


@torch.no_grad()
def compute_action_recognition_accuracy(
    predicted_actions: torch.Tensor,
    gt_actions: torch.Tensor,
) -> float:
    """
    Fraction of frames where the predicted action matches GT.

    Can be used with a simple action-classifier head trained on generated
    video, or by directly comparing GT to a heuristic decoder.
    """
    return (predicted_actions == gt_actions).float().mean().item()


@torch.no_grad()
def compute_controllability_score(
    commanded_actions: torch.Tensor,
    observed_displacements: torch.Tensor,
) -> float:
    """
    Correlation between commanded actions and observed frame-to-frame
    vertical displacement (proxy for jump / no-jump responsiveness).

    Args:
        commanded_actions: [T] int tensor (0 = no-jump, 1 = jump).
        observed_displacements: [T] float tensor, measured vertical
            displacement magnitude in pixel or latent space.

    Returns:
        Pearson correlation coefficient.
    """
    cmd = commanded_actions.float().cpu().numpy()
    obs = observed_displacements.float().cpu().numpy()

    if cmd.std() < 1e-8 or obs.std() < 1e-8:
        return 0.0

    return float(np.corrcoef(cmd, obs)[0, 1])


@torch.no_grad()
def compute_action_response_latency(
    commanded_actions: torch.Tensor,
    frame_diffs: torch.Tensor,
    threshold: float = 0.05,
) -> float:
    """
    Average number of frames between an action command and observable
    visual response (mean absolute frame diff exceeds threshold).

    Args:
        commanded_actions: [T] int tensor of action commands.
        frame_diffs: [T] float tensor of per-frame mean absolute diff
            from the previous frame.
        threshold: Minimum diff to count as a visual response.

    Returns:
        Mean response latency in frames.  Lower is better.
    """
    T = commanded_actions.shape[0]
    actions_np = commanded_actions.cpu().numpy()
    diffs_np = frame_diffs.cpu().numpy()

    latencies = []
    for t in range(T):
        if actions_np[t] == 0:
            continue
        for lag in range(0, min(16, T - t)):
            if diffs_np[t + lag] > threshold:
                latencies.append(lag)
                break
        else:
            latencies.append(16)

    if not latencies:
        return 0.0
    return float(np.mean(latencies))


@torch.no_grad()
def compute_action_success_rate(
    commanded_actions: torch.Tensor,
    frame_diffs: torch.Tensor,
    threshold: float = 0.05,
    max_lag: int = 8,
) -> float:
    """
    Fraction of action commands that produce a detectable visual response
    within ``max_lag`` frames.
    """
    T = commanded_actions.shape[0]
    actions_np = commanded_actions.cpu().numpy()
    diffs_np = frame_diffs.cpu().numpy()

    total_actions = 0
    successes = 0

    for t in range(T):
        if actions_np[t] == 0:
            continue
        total_actions += 1
        for lag in range(0, min(max_lag, T - t)):
            if diffs_np[t + lag] > threshold:
                successes += 1
                break

    if total_actions == 0:
        return 1.0
    return successes / total_actions


@torch.no_grad()
def evaluate_action_metrics(
    commanded_actions: torch.Tensor,
    generated_frames: torch.Tensor,
    gt_frames: Optional[torch.Tensor] = None,
    vqvae_decoder: Optional[torch.nn.Module] = None,
    response_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Full action/control evaluation.

    Args:
        commanded_actions: [T] int action sequence.
        generated_frames: [T, C, H, W] generated frames.
        gt_frames: [T, C, H, W] ground-truth frames (for displacement comparison).
        vqvae_decoder: Decode latents to RGB if needed.
        response_threshold: Pixel-diff threshold for response detection.
    """
    results: Dict[str, float] = {}
    T = generated_frames.shape[0]

    if vqvae_decoder is not None:
        rgb_gen = torch.cat([vqvae_decoder(generated_frames[i:i+1]) for i in range(T)], dim=0)
        rgb_gen = (rgb_gen + 1) / 2
    else:
        rgb_gen = (generated_frames + 1) / 2

    frame_diffs = torch.zeros(T, device=generated_frames.device)
    for i in range(1, T):
        frame_diffs[i] = (rgb_gen[i] - rgb_gen[i - 1]).abs().mean()

    observed_displacements = frame_diffs

    results[M.controllability_score()] = compute_controllability_score(
        commanded_actions, observed_displacements
    )
    results[M.action_response_latency()] = compute_action_response_latency(
        commanded_actions, frame_diffs, threshold=response_threshold
    )
    results[M.action_success_rate()] = compute_action_success_rate(
        commanded_actions, frame_diffs, threshold=response_threshold
    )

    return results
