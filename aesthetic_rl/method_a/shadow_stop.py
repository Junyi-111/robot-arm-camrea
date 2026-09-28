"""STOP-policy diagnostics and fixed-pose verification for Method A.

The monitor never commands the robot and never terminates an episode. It only
turns a score/target time series into a hypothetical ``WOULD_STOP`` decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median

from .config import ShadowStopConfig


@dataclass(frozen=True)
class ShadowStopDecision:
    step: int
    raw_score: float
    median_score: float
    running_best_median: float
    improvement_over_patience: float | None
    target_present: bool
    eligible: bool
    stagnant: bool
    quality_ok: bool
    candidate: bool
    confirmation_count: int
    would_stop: bool
    already_triggered: bool
    segment_start_step: int
    best_pose_step: int | None
    best_pose_raw_score: float | None
    best_pose_joint_deg: tuple[float, ...] | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StopVerificationResult:
    scores: tuple[float, ...]
    target_present: tuple[bool, ...]
    median_score: float
    target_present_all: bool
    quality_ok: bool
    confirmed: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_stop_verification(
    *,
    scores: list[float] | tuple[float, ...],
    target_present: list[bool] | tuple[bool, ...],
    expected_frames: int,
    quality_score_threshold: float,
) -> StopVerificationResult:
    """Decide whether a fixed-pose return is good enough to end an episode."""

    score_values = tuple(float(value) for value in scores)
    target_values = tuple(bool(value) for value in target_present)
    if len(score_values) != int(expected_frames) or len(target_values) != int(
        expected_frames
    ):
        raise ValueError("STOP verification did not collect the expected frame count")
    if not score_values or not all(math.isfinite(value) for value in score_values):
        raise ValueError("STOP verification scores must be finite")
    median_score = float(median(score_values))
    target_present_all = all(target_values)
    quality_ok = median_score >= float(quality_score_threshold)
    confirmed = bool(target_present_all and quality_ok)
    if not target_present_all:
        reason = "target_missing"
    elif not quality_ok:
        reason = "below_quality_threshold"
    else:
        reason = "confirmed"
    return StopVerificationResult(
        scores=score_values,
        target_present=target_values,
        median_score=median_score,
        target_present_all=target_present_all,
        quality_ok=quality_ok,
        confirmed=confirmed,
        reason=reason,
    )


class ShadowStopMonitor:
    """Evaluate a STOP rule without changing actor behavior.

    Edge clearance is deliberately absent from this interface. A detected
    target is the only Grounding-DINO gate; composition remains ArtiMuse's job.
    """

    def __init__(self, config: ShadowStopConfig) -> None:
        self.config = config
        self.raw_scores: list[float] = []
        self.median_scores: list[float] = []
        self.running_bests: list[float] = []
        self.segment_start_step = 0
        self.confirmation_count = 0
        self.triggered = False
        self.trigger_step: int | None = None
        self.trigger_score: float | None = None
        self.trigger_median_score: float | None = None
        self.best_pose_step: int | None = None
        self.best_pose_raw_score: float | None = None
        self.best_pose_joint_deg: tuple[float, ...] | None = None
        self.trigger_best_pose_step: int | None = None
        self.trigger_best_pose_raw_score: float | None = None
        self.trigger_best_pose_joint_deg: tuple[float, ...] | None = None
        self.reset_count = 0

    def _reset_segment(self, step: int) -> None:
        self.raw_scores.clear()
        self.median_scores.clear()
        self.running_bests.clear()
        self.segment_start_step = int(step)
        self.confirmation_count = 0
        self.best_pose_step = None
        self.best_pose_raw_score = None
        self.best_pose_joint_deg = None

    def observe(
        self,
        *,
        step: int,
        score: float,
        target_present: bool,
        intervention: bool = False,
        joints_deg: tuple[float, ...] | list[float] | None = None,
    ) -> ShadowStopDecision:
        """Observe one frame and return a passive decision.

        An intervention starts a fresh segment and cannot itself trigger STOP.
        This keeps policy-only stagnation separate from operator corrections.
        """

        step = int(step)
        score = float(score)
        if intervention:
            self._reset_segment(step)
            self.reset_count += 1

        if target_present and (
            self.best_pose_raw_score is None or score > self.best_pose_raw_score
        ):
            self.best_pose_step = step
            self.best_pose_raw_score = score
            self.best_pose_joint_deg = (
                None
                if joints_deg is None
                else tuple(float(value) for value in joints_deg)
            )

        self.raw_scores.append(score)
        window = self.raw_scores[-self.config.median_window_frames :]
        median_score = float(median(window))
        self.median_scores.append(median_score)
        running_best = max(
            self.running_bests[-1] if self.running_bests else median_score,
            median_score,
        )
        self.running_bests.append(float(running_best))

        improvement: float | None = None
        enough_history = len(self.running_bests) > self.config.stagnation_patience_steps
        if enough_history:
            earlier_best = self.running_bests[
                -self.config.stagnation_patience_steps - 1
            ]
            improvement = float(running_best - earlier_best)

        enough_steps = (
            step - self.segment_start_step >= self.config.minimum_search_steps
        )
        eligible = bool(
            not intervention
            and len(self.raw_scores) >= self.config.median_window_frames
            and enough_history
            and enough_steps
        )
        stagnant = bool(
            eligible
            and improvement is not None
            and improvement <= self.config.improvement_epsilon
        )
        quality_ok = bool(median_score >= self.config.quality_score_threshold)
        candidate = bool(eligible and target_present and stagnant and quality_ok)

        if candidate and not self.triggered:
            self.confirmation_count += 1
        elif not self.triggered:
            self.confirmation_count = 0

        would_stop = bool(
            candidate
            and not self.triggered
            and self.confirmation_count >= self.config.confirmation_steps
        )
        already_triggered = self.triggered
        if would_stop:
            self.triggered = True
            self.trigger_step = step
            self.trigger_score = score
            self.trigger_median_score = median_score
            self.trigger_best_pose_step = self.best_pose_step
            self.trigger_best_pose_raw_score = self.best_pose_raw_score
            self.trigger_best_pose_joint_deg = self.best_pose_joint_deg

        if intervention:
            reason = "intervention_reset"
        elif self.triggered and not would_stop:
            reason = "already_triggered"
        elif not target_present:
            reason = "target_missing"
        elif not eligible:
            reason = "collecting_history"
        elif not quality_ok:
            reason = "below_quality_threshold"
        elif not stagnant:
            reason = "meaningful_improvement_recently"
        elif not would_stop:
            reason = "confirming_candidate"
        else:
            reason = "would_stop"

        return ShadowStopDecision(
            step=step,
            raw_score=score,
            median_score=median_score,
            running_best_median=float(running_best),
            improvement_over_patience=improvement,
            target_present=bool(target_present),
            eligible=eligible,
            stagnant=stagnant,
            quality_ok=quality_ok,
            candidate=candidate,
            confirmation_count=self.confirmation_count,
            would_stop=would_stop,
            already_triggered=already_triggered,
            segment_start_step=self.segment_start_step,
            best_pose_step=self.best_pose_step,
            best_pose_raw_score=self.best_pose_raw_score,
            best_pose_joint_deg=self.best_pose_joint_deg,
            reason=reason,
        )

    def summary(self) -> dict[str, object]:
        return {
            "enabled": True,
            "observations_in_current_segment": len(self.raw_scores),
            "intervention_resets": self.reset_count,
            "would_stop": self.triggered,
            "would_stop_step": self.trigger_step,
            "would_stop_raw_score": self.trigger_score,
            "would_stop_median3_score": self.trigger_median_score,
            "would_return_to_best": self.triggered,
            "return_target_step": self.trigger_best_pose_step,
            "return_target_raw_score": self.trigger_best_pose_raw_score,
            "return_target_joint_deg": self.trigger_best_pose_joint_deg,
            "return_executed": False,
            "edge_clear_required": False,
            "target_present_required": True,
        }
