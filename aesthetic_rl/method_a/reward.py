"""Reward and target-box feature functions shared by demos and the actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import RewardConfig


TARGET_STATE_DIM = 7


@dataclass(frozen=True)
class RewardResult:
    total: float
    aesthetic: float
    aesthetic_absolute: float
    aesthetic_delta: float
    target: float
    motion: float
    smooth: float
    target_present: bool
    target_edge_clear: bool
    target_valid: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "reward": self.total,
            "reward_aesthetic": self.aesthetic,
            "reward_aesthetic_absolute": self.aesthetic_absolute,
            "reward_aesthetic_delta": self.aesthetic_delta,
            "reward_target": self.target,
            "penalty_motion": self.motion,
            "penalty_smooth": self.smooth,
            "target_present": self.target_present,
            "target_edge_clear": self.target_edge_clear,
            # Compatibility for existing logs/analysis: valid means that the
            # detected box also clears the configured edge margin.
            "target_valid": self.target_valid,
        }


def target_state_from_detection(
    detection: dict[str, Any] | None,
    *,
    edge_margin_ratio: float,
) -> np.ndarray:
    """Convert a Grounding DINO response into a normalized compact state."""
    detection = detection or {}
    width = float(detection.get("width") or 0.0)
    height = float(detection.get("height") or 0.0)
    best = detection.get("best")
    if not best or width <= 0.0 or height <= 0.0:
        return np.zeros(TARGET_STATE_DIM, dtype=np.float32)
    try:
        x1, y1, x2, y2 = [float(v) for v in best["xyxy"]]
        confidence = float(best.get("score", 0.0))
    except (KeyError, TypeError, ValueError):
        return np.zeros(TARGET_STATE_DIM, dtype=np.float32)
    x1, x2 = sorted((np.clip(x1, 0.0, width), np.clip(x2, 0.0, width)))
    y1, y2 = sorted((np.clip(y1, 0.0, height), np.clip(y2, 0.0, height)))
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    margin = max(0.0, min(x1, y1, width - x2, height - y2))
    margin_ratio = margin / max(1.0, min(width, height))
    valid = bool(detection.get("target_valid")) and margin_ratio >= edge_margin_ratio
    return np.asarray(
        [
            ((x1 + x2) * 0.5) / width,
            ((y1 + y2) * 0.5) / height,
            box_w / width,
            box_h / height,
            np.clip(confidence, 0.0, 1.0),
            np.clip(margin_ratio, 0.0, 1.0),
            float(valid),
        ],
        dtype=np.float32,
    )


def compute_reward(
    *,
    score: float,
    next_score: float,
    next_target_state: np.ndarray,
    actual_delta_deg: np.ndarray,
    max_step_deg: np.ndarray,
    action: np.ndarray,
    previous_action: np.ndarray,
    config: RewardConfig,
    edge_margin_ratio: float,
) -> RewardResult:
    """Compute the same bounded reward for offline and online transitions."""
    values = np.asarray(
        [score, next_score, *actual_delta_deg, *max_step_deg, *action, *previous_action],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("reward inputs must all be finite")
    target = np.asarray(next_target_state, dtype=np.float32).reshape(-1)
    if target.shape != (TARGET_STATE_DIM,):
        raise ValueError(f"target state must be {TARGET_STATE_DIM}-D, got {target.shape}")
    if np.any(np.asarray(max_step_deg) <= 0.0):
        raise ValueError("max_step_deg must be positive")

    aesthetic_absolute = float(np.clip(
        (float(next_score) - config.aesthetic_absolute_center)
        / config.aesthetic_absolute_scale,
        -1.0,
        1.0,
    ))
    aesthetic_delta = float(np.clip(
        (float(next_score) - float(score)) / config.aesthetic_delta_scale,
        -1.0,
        1.0,
    ))
    aesthetic = (
        config.aesthetic_absolute_weight * aesthetic_absolute
        + config.aesthetic_delta_weight * aesthetic_delta
    )
    # target_state[-1] means edge-clear, not target presence.  A retained box
    # has positive width, height and confidence even when it touches an edge.
    present = bool(target[2] > 0.0 and target[3] > 0.0 and target[4] > 0.0)
    edge_clear = bool(target[-1] >= 0.5)
    if present:
        # Continuous penalty only inside the unsafe margin: zero at/after the
        # configured clearance and -1 exactly at an image boundary.
        margin = float(target[5])
        edge_fraction = max(0.0, 1.0 - margin / max(edge_margin_ratio, 1e-6))
        target_reward = -(edge_fraction**2)
    else:
        target_reward = float(config.missing_target_penalty)
    controlled_delta = np.asarray(actual_delta_deg, dtype=np.float32).reshape(-1)
    scales = np.asarray(max_step_deg, dtype=np.float32).reshape(-1)
    if controlled_delta.shape != scales.shape:
        raise ValueError("actual_delta_deg and max_step_deg shapes must match")
    motion = float(np.mean(np.square(controlled_delta / scales)))
    act = np.asarray(action, dtype=np.float32).reshape(-1)
    prev = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if act.shape != prev.shape:
        raise ValueError("action and previous_action shapes must match")
    smooth = float(np.mean(np.square(act - prev)))
    total = (
        config.aesthetic_weight * aesthetic
        + config.target_weight * target_reward
        - config.motion_weight * motion
        - config.smooth_weight * smooth
    )
    return RewardResult(
        total=float(total),
        aesthetic=aesthetic,
        aesthetic_absolute=aesthetic_absolute,
        aesthetic_delta=aesthetic_delta,
        target=target_reward,
        motion=motion,
        smooth=smooth,
        target_present=present,
        target_edge_clear=edge_clear,
        target_valid=edge_clear,
    )
