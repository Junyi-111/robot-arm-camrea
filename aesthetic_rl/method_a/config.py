"""Typed configuration and validation for method A."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class VisionConfig:
    analyze_url: str = "http://127.0.0.1:18080/analyze"
    grounding_url: str = "http://127.0.0.1:18081/detect"
    request_timeout_seconds: float = 30.0
    image_size: int = 128
    aesthetic_feature_dim: int = 3584
    target_prompt: str = "flower"
    edge_margin_ratio: float = 0.05


@dataclass(frozen=True)
class RewardConfig:
    aesthetic_absolute_center: float = 32.0
    aesthetic_absolute_scale: float = 8.0
    aesthetic_absolute_weight: float = 0.8
    aesthetic_delta_scale: float = 5.0
    aesthetic_delta_weight: float = 0.2
    aesthetic_weight: float = 1.0
    target_weight: float = 0.25
    motion_weight: float = 0.02
    smooth_weight: float = 0.05
    # Multiplied by target_weight, giving an effective -1 terminal penalty.
    missing_target_penalty: float = -4.0
    lost_target_terminal_frames: int = 1


@dataclass(frozen=True)
class RobotConfig:
    controlled_indices: tuple[int, ...] = (0, 1, 2, 4)
    max_step_deg: tuple[float, ...] = (1.80, 1.08, 1.08, 1.44)
    home_joint_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # Policy workspace learned from the demonstrations. SAC and manual
    # intervention are clipped to these bounds.
    joint_low_deg: tuple[float, ...] = (-25.0, -5.0, -15.0, -5.0, -28.0, -5.0)
    # J3 cannot produce useful positive motion beyond 0 deg on the real arm.
    # Keep the wider startup envelope below so a disabled/sagged arm can still
    # be recovered, but never command SAC/manual motion above this limit.
    joint_high_deg: tuple[float, ...] = (10.0, 60.0, 0.0, 5.0, 10.0, 5.0)
    # A separate guarded envelope admits a gravity-displaced, disabled arm so
    # the first supervised reset can return controlled joints to the policy
    # workspace. It does not expand SAC exploration.
    startup_joint_low_deg: tuple[float, ...] = (-30.0, -10.0, -20.0, -10.0, -35.0, -10.0)
    startup_joint_high_deg: tuple[float, ...] = (15.0, 65.0, 10.0, 10.0, 35.0, 10.0)
    random_initial_deg: float = 5.0
    speed_percent: int = 3
    action_seconds: float = 0.6
    reset_speed_percent: int = 3
    reset_deg_per_second: float = 3.0
    settle_seconds: float = 0.6
    feedback_timeout_seconds: float = 1.0
    # Fixed non-policy gripper opening used by the teammate's demo recorder.
    gripper_opening_m: float = 0.07


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    bc_warmup_steps: int = 200
    training_starts: int = 20
    max_online_steps: int = 2000
    max_updates: int = 2000
    replay_capacity: int = 50_000
    demo_fraction: float = 0.5
    discount: float = 0.97
    tau: float = 0.005
    learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 0.01
    checkpoint_period: int = 50
    publish_period: int = 5
    buffer_save_period: int = 50
    max_episode_steps: int = 80


@dataclass(frozen=True)
class ShadowStopConfig:
    # Calibrated on 2026-09-24 from 30 fixed-pose fresh D435 frames.
    improvement_epsilon: float = 0.8186361694335936
    deterioration_threshold: float = 1.6372723388671873
    quality_score_threshold: float = 40.0
    median_window_frames: int = 3
    minimum_search_steps: int = 10
    # Five steps was too aggressive on the frozen checkpoint-1731 replay.
    stagnation_patience_steps: int = 15
    confirmation_steps: int = 3
    # Conservative active-STOP return/verification settings. The return uses
    # the same guarded interpolator as reset motion, but at a slower rate.
    return_deg_per_second: float = 2.0
    return_joint_tolerance_deg: float = 0.75
    verification_frames: int = 3
    verification_interval_seconds: float = 0.25
    max_return_attempts_per_episode: int = 1


@dataclass(frozen=True)
class RuntimeConfig:
    camera_serial: str = "109622075182"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    learner_host: str = "127.0.0.1"
    trainer_port: int = 5588
    broadcast_port: int = 5589
    disable_countdown_seconds: int = 5
    operator_decision_seconds: float = 0.75
    output_dir: str = "aesthetic_rl/method_a_artifacts/runtime_absolute_v2"
    checkpoint_dir: str = "aesthetic_rl/method_a_artifacts/checkpoints_absolute_v2"
    evaluation_output_dir: str = (
        "aesthetic_rl/method_a_artifacts/evaluation_checkpoint_00001731"
    )
    evaluation_checkpoint: str = (
        "aesthetic_rl/method_a_artifacts/checkpoints_absolute_v2/"
        "checkpoint_00001731.msgpack"
    )
    prepared_demo_path: str = (
        "aesthetic_rl/method_a_artifacts/demos/method_a_demo_absolute_v2.pkl"
    )
    feature_cache_dir: str = "aesthetic_rl/method_a_artifacts/feature_cache"


@dataclass(frozen=True)
class MethodAConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    shadow_stop: ShadowStopConfig = field(default_factory=ShadowStopConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> "MethodAConfig":
        if self.vision.image_size < 32:
            raise ValueError("vision.image_size must be at least 32")
        if self.vision.aesthetic_feature_dim <= 0:
            raise ValueError("vision.aesthetic_feature_dim must be positive")
        if not 0.0 < self.vision.edge_margin_ratio < 0.5:
            raise ValueError("vision.edge_margin_ratio must be in (0, 0.5)")
        if self.reward.aesthetic_absolute_scale <= 0.0:
            raise ValueError("reward.aesthetic_absolute_scale must be positive")
        if self.reward.aesthetic_delta_scale <= 0.0:
            raise ValueError("reward.aesthetic_delta_scale must be positive")
        if min(
            self.reward.aesthetic_absolute_weight,
            self.reward.aesthetic_delta_weight,
            self.reward.aesthetic_weight,
            self.reward.target_weight,
        ) < 0.0:
            raise ValueError("reward weights cannot be negative")
        if abs(
            self.reward.aesthetic_absolute_weight
            + self.reward.aesthetic_delta_weight
            - 1.0
        ) > 1e-6:
            raise ValueError("absolute and delta aesthetic weights must sum to 1")
        if self.reward.missing_target_penalty > 0.0:
            raise ValueError("missing_target_penalty cannot be positive")
        if self.reward.lost_target_terminal_frames < 1:
            raise ValueError("lost_target_terminal_frames must be positive")
        if self.robot.controlled_indices != (0, 1, 2, 4):
            raise ValueError("method A is fixed to physical joints J1/J2/J3/J5")
        if len(self.robot.max_step_deg) != 4:
            raise ValueError("robot.max_step_deg must contain J1/J2/J3/J5")
        if len(self.robot.home_joint_deg) != 6:
            raise ValueError("robot.home_joint_deg must contain six joints")
        if len(self.robot.joint_low_deg) != 6 or len(self.robot.joint_high_deg) != 6:
            raise ValueError("joint policy bounds must contain six joints")
        if (
            len(self.robot.startup_joint_low_deg) != 6
            or len(self.robot.startup_joint_high_deg) != 6
        ):
            raise ValueError("joint startup bounds must contain six joints")
        if any(lo >= hi for lo, hi in zip(self.robot.joint_low_deg, self.robot.joint_high_deg)):
            raise ValueError("every joint policy low bound must be below its high bound")
        if any(
            lo >= hi
            for lo, hi in zip(
                self.robot.startup_joint_low_deg, self.robot.startup_joint_high_deg
            )
        ):
            raise ValueError("every joint startup low bound must be below its high bound")
        if any(
            startup_lo > policy_lo or policy_hi > startup_hi
            for startup_lo, policy_lo, policy_hi, startup_hi in zip(
                self.robot.startup_joint_low_deg,
                self.robot.joint_low_deg,
                self.robot.joint_high_deg,
                self.robot.startup_joint_high_deg,
            )
        ):
            raise ValueError("joint policy bounds must lie inside startup bounds")
        if any(
            not lo <= home <= hi
            for home, lo, hi in zip(
                self.robot.home_joint_deg,
                self.robot.joint_low_deg,
                self.robot.joint_high_deg,
            )
        ):
            raise ValueError("every home joint angle must lie within its safety bounds")
        if self.robot.random_initial_deg < 0.0:
            raise ValueError("robot.random_initial_deg cannot be negative")
        if not 1 <= self.robot.speed_percent <= 10:
            raise ValueError("initial real-robot speed_percent must be in [1, 10]")
        if not 0.0 <= self.robot.gripper_opening_m <= 0.07:
            raise ValueError("robot.gripper_opening_m must be in [0, 0.07]")
        if self.training.batch_size < 2 or self.training.batch_size % 2:
            raise ValueError("training.batch_size must be an even integer >= 2")
        if not 0.0 < self.training.demo_fraction < 1.0:
            raise ValueError("training.demo_fraction must be in (0, 1)")
        if self.training.max_episode_steps < 1:
            raise ValueError("training.max_episode_steps must be positive")
        if self.shadow_stop.improvement_epsilon <= 0.0:
            raise ValueError("shadow_stop.improvement_epsilon must be positive")
        if self.shadow_stop.deterioration_threshold <= 0.0:
            raise ValueError("shadow_stop.deterioration_threshold must be positive")
        if self.shadow_stop.median_window_frames < 1:
            raise ValueError("shadow_stop.median_window_frames must be positive")
        if self.shadow_stop.median_window_frames % 2 == 0:
            raise ValueError("shadow_stop.median_window_frames must be odd")
        if self.shadow_stop.minimum_search_steps < 0:
            raise ValueError("shadow_stop.minimum_search_steps cannot be negative")
        if self.shadow_stop.stagnation_patience_steps < 1:
            raise ValueError("shadow_stop.stagnation_patience_steps must be positive")
        if self.shadow_stop.confirmation_steps < 1:
            raise ValueError("shadow_stop.confirmation_steps must be positive")
        if self.shadow_stop.return_deg_per_second <= 0.0:
            raise ValueError("shadow_stop.return_deg_per_second must be positive")
        if self.shadow_stop.return_joint_tolerance_deg <= 0.0:
            raise ValueError("shadow_stop.return_joint_tolerance_deg must be positive")
        if self.shadow_stop.verification_frames < 1:
            raise ValueError("shadow_stop.verification_frames must be positive")
        if self.shadow_stop.verification_frames % 2 == 0:
            raise ValueError("shadow_stop.verification_frames must be odd")
        if self.shadow_stop.verification_interval_seconds < 0.0:
            raise ValueError(
                "shadow_stop.verification_interval_seconds cannot be negative"
            )
        if self.shadow_stop.max_return_attempts_per_episode < 1:
            raise ValueError(
                "shadow_stop.max_return_attempts_per_episode must be positive"
            )
        if min(
            self.training.checkpoint_period,
            self.training.publish_period,
            self.training.buffer_save_period,
        ) < 1:
            raise ValueError("checkpoint/publish/buffer periods must be positive")
        if self.runtime.disable_countdown_seconds < 1:
            raise ValueError("disable_countdown_seconds must be positive")
        if self.runtime.operator_decision_seconds < 0.0:
            raise ValueError("operator_decision_seconds cannot be negative")
        if Path(self.runtime.evaluation_output_dir) == Path(self.runtime.output_dir):
            raise ValueError("evaluation_output_dir must differ from training output_dir")
        if Path(self.runtime.evaluation_checkpoint).suffix != ".msgpack":
            raise ValueError("evaluation_checkpoint must be a .msgpack file")
        return self


def _construct(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    if cls is RobotConfig:
        for key in (
            "controlled_indices",
            "max_step_deg",
            "home_joint_deg",
            "joint_low_deg",
            "joint_high_deg",
            "startup_joint_low_deg",
            "startup_joint_high_deg",
        ):
            if key in values:
                values[key] = tuple(values[key])
    return cls(**values)


def load_config(path: str | Path) -> MethodAConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    unknown = set(raw) - {
        "vision",
        "reward",
        "robot",
        "training",
        "shadow_stop",
        "runtime",
    }
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")
    config = MethodAConfig(
        vision=_construct(VisionConfig, raw.get("vision")),
        reward=_construct(RewardConfig, raw.get("reward")),
        robot=_construct(RobotConfig, raw.get("robot")),
        training=_construct(TrainingConfig, raw.get("training")),
        shadow_stop=_construct(ShadowStopConfig, raw.get("shadow_stop")),
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
    )
    return config.validate()
