"""Physical Method-A actor with an operator-first safety state machine."""

from __future__ import annotations

import enum
import queue
import random
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera import D435Camera
from .config import MethodAConfig
from .robot_ros2 import PiperRos2, RobotSafetyError
from .runtime import (
    CapturedObservation,
    ObservationBuilder,
    PolicyLink,
    RuntimeLogger,
    make_transition,
)
from .shadow_stop import ShadowStopMonitor, StopVerificationResult, assess_stop_verification
from .vision_client import RemoteVisionClient


class ActorState(str, enum.Enum):
    PREFLIGHT = "PREFLIGHT"
    WAIT_ARM = "WAIT_ARM"
    RESETTING = "RESETTING"
    WAIT_CONFIRM = "WAIT_CONFIRM"
    POSITIONING = "POSITIONING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    INTERVENTION = "INTERVENTION"
    STOP_RETURNING = "STOP_RETURNING"
    STOP_VERIFYING = "STOP_VERIFYING"
    STOPPED_HOLD = "STOPPED_HOLD"
    FAULT_HOLD = "FAULT_HOLD"
    DISABLED = "DISABLED"
    CLOSED = "CLOSED"


@dataclass
class UiSnapshot:
    state: str = ActorState.PREFLIGHT.value
    message: str = "initializing"
    episode: int = 0
    episode_step: int = 0
    online_steps: int = 0
    policy_step: int = 0
    score: float | None = None
    best_score: float | None = None
    reward: float | None = None
    target_present: bool | None = None
    target_valid: bool | None = None
    joints_deg: np.ndarray | None = None
    image: np.ndarray | None = None
    detection: dict[str, Any] | None = None
    disable_ready_at: float | None = None


@dataclass(frozen=True)
class ActiveStopExecution:
    verification: StopVerificationResult
    final_observation: CapturedObservation
    target_joint_deg: tuple[float, ...]
    reached_joint_deg: tuple[float, ...]
    max_joint_error_deg: float
    verification_images: tuple[str, ...]


class ManualKeyGate:
    """Turn OpenCV/OS key-repeat events into deliberate press-release steps."""

    def __init__(self, release_seconds: float = 1.0):
        self.release_seconds = float(release_seconds)
        self.latched_key: int | None = None
        self.last_seen = 0.0

    def update(self, key: int | None, now: float) -> bool:
        if key is None:
            if (
                self.latched_key is not None
                and now - self.last_seen >= self.release_seconds
            ):
                self.latched_key = None
            return False
        if self.latched_key is None:
            self.latched_key = key
            self.last_seen = now
            return True
        # Repeated events, including a different jog key pressed too soon, are
        # ignored and extend the release window.
        self.last_seen = now
        return False

    def reset(self) -> None:
        self.latched_key = None
        self.last_seen = 0.0


class OperatorUI:
    """OpenCV UI with both keyboard shortcuts and clickable controls."""

    WINDOW = "Method A - REAL ROBOT (arm remains enabled on stop)"

    def __init__(self, event_queue: queue.Queue[str], snapshot: UiSnapshot, lock: threading.Lock):
        self.events = event_queue
        self.snapshot = snapshot
        self.lock = lock
        self.buttons: list[tuple[tuple[int, int, int, int], str]] = []
        self.manual_key_gate = ManualKeyGate()
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 1050, 640)
        cv2.setMouseCallback(self.WINDOW, self._mouse)

    def _mouse(self, event, x, y, _flags, _parameter):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for (x1, y1, x2, y2), action in list(self.buttons):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.events.put(action)
                return

    @staticmethod
    def _text(canvas, value, at, *, color=(230, 230, 230), scale=0.55, thickness=1):
        cv2.putText(canvas, str(value), at, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def draw(self) -> None:
        with self.lock:
            snap = UiSnapshot(**vars(self.snapshot))
            if snap.image is not None:
                snap.image = snap.image.copy()
            if snap.joints_deg is not None:
                snap.joints_deg = snap.joints_deg.copy()
        image = snap.image
        if image is None:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (640, 480), cv2.INTER_AREA)
        best = (snap.detection or {}).get("best")
        if best and "xyxy" in best:
            source_w = float((snap.detection or {}).get("width") or image.shape[1])
            source_h = float((snap.detection or {}).get("height") or image.shape[0])
            x1, y1, x2, y2 = best["xyxy"]
            points = [
                int(x1 * image.shape[1] / source_w),
                int(y1 * image.shape[0] / source_h),
                int(x2 * image.shape[1] / source_w),
                int(y2 * image.shape[0] / source_h),
            ]
            edge_clear = bool(snap.target_valid)
            color = (0, 220, 0) if edge_clear else (0, 165, 255)
            cv2.rectangle(image, points[:2], points[2:], color, 2)
        canvas = np.zeros((640, 1050, 3), dtype=np.uint8)
        canvas[:480, :640] = image
        state_color = (0, 220, 255) if "HOLD" not in snap.state else (0, 80, 255)
        self._text(canvas, f"STATE: {snap.state}", (15, 515), color=state_color, scale=0.7, thickness=2)
        self._text(canvas, snap.message[:95], (15, 545), color=(180, 220, 255))
        self._text(
            canvas,
            "Stop X = HOLD ENABLED. Disable never happens automatically.",
            (15, 580),
            color=(0, 180, 255),
            scale=0.58,
            thickness=2,
        )
        self._text(canvas, "Close window/Q also stops and HOLDS.", (15, 610), color=(0, 180, 255))
        x = 665
        lines = [
            f"episode {snap.episode}  step {snap.episode_step}",
            f"online {snap.online_steps}  policy {snap.policy_step}",
            f"score {snap.score if snap.score is not None else '--'}",
            f"best {snap.best_score if snap.best_score is not None else '--'}",
            f"reward {snap.reward if snap.reward is not None else '--'}",
            f"target present {snap.target_present}  edge clear {snap.target_valid}",
        ]
        if snap.joints_deg is not None:
            lines.append("joints " + " ".join(f"{v:.1f}" for v in snap.joints_deg))
        if best:
            confidence = best.get("score")
            margin = best.get("margin_ratio")
            if confidence is not None:
                lines.append(f"box confidence {float(confidence):.3f}")
            if margin is not None:
                lines.append(
                    f"box margin {100.0 * float(margin):.1f}%  cropped {bool(best.get('cropped'))}"
                )
        if snap.disable_ready_at is not None:
            remaining = max(0.0, snap.disable_ready_at - time.monotonic())
            lines.append(f"DISABLE WARNING {remaining:.1f}s")
            if remaining <= 0.0:
                lines.append("Protect arm, press D again")
        for index, line in enumerate(lines):
            self._text(canvas, line, (x, 30 + index * 27), scale=0.52)
        controls = [
            ("SPACE Arm/Start", "space"),
            ("C Recapture/check", "recapture"),
            ("H Set current home", "set_home"),
            ("P Pause", "pause"),
            ("O End episode", "end_episode"),
            ("A Intervention", "intervention"),
            ("B Resume policy", "resume"),
            ("X Stop + HOLD", "stop"),
            ("D Disable confirm", "disable"),
            ("ESC Cancel disable", "cancel_disable"),
        ]
        self.buttons = []
        for index, (label, action) in enumerate(controls):
            y1 = 245 + index * 39
            rect = (670, y1, 1015, y1 + 31)
            cv2.rectangle(canvas, rect[:2], rect[2:], (75, 75, 75), -1)
            cv2.rectangle(canvas, rect[:2], rect[2:], (150, 150, 150), 1)
            self._text(canvas, label, (680, y1 + 21), scale=0.52)
            self.buttons.append((rect, action))
        cv2.imshow(self.WINDOW, canvas)

    def poll(self) -> bool:
        self.draw()
        key = cv2.waitKey(30) & 0xFF
        with self.lock:
            manual_mode = self.snapshot.state in (
                ActorState.INTERVENTION.value,
                ActorState.POSITIONING.value,
            )
        mapping = {
            32: "space",
            ord("p"): "pause",
            ord("o"): "end_episode",
            ord("a"): "intervention",
            ord("b"): "resume",
            ord("c"): "recapture",
            ord("h"): "set_home",
            ord("x"): "stop",
            27: "cancel_disable",
        }
        if key in mapping:
            self.events.put(mapping[key])
        manual = {
            ord("w"): "manual_j1_pos",
            ord("s"): "manual_j1_neg",
            ord("e"): "manual_j2_pos",
            ord("d"): "manual_j2_neg",
            ord("r"): "manual_j3_pos",
            ord("f"): "manual_j3_neg",
            ord("t"): "manual_j5_pos",
            ord("g"): "manual_j5_neg",
        }
        if manual_mode:
            manual_key = key if key in manual else None
            if self.manual_key_gate.update(manual_key, time.monotonic()):
                self.events.put(manual[key])
        else:
            self.manual_key_gate.reset()
            if key == ord("d"):
                self.events.put("disable")
        if key == ord("q"):
            self.events.put("stop")
            return False
        try:
            visible = cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0
        if visible < 1:
            self.events.put("stop")
            return False
        return True

    def close(self):
        cv2.destroyWindow(self.WINDOW)


class MethodAActor:
    def __init__(
        self,
        config: MethodAConfig,
        *,
        arm: bool,
        deterministic: bool = False,
        evaluation_only: bool = False,
        shadow_stop: bool = False,
        active_stop: bool = False,
        image_encoder_type: str = "resnet-pretrained",
        seed: int | None = None,
    ) -> None:
        if not arm:
            raise ValueError("MethodAActor requires --arm; use run_preflight for read-only checks")
        if shadow_stop and active_stop:
            raise ValueError("choose exactly one of shadow STOP or active STOP")
        if (shadow_stop or active_stop) and not evaluation_only:
            raise ValueError("STOP evaluation requires frozen --eval-only mode")
        self.config = config
        self.evaluation_only = bool(evaluation_only)
        self.deterministic = bool(deterministic or self.evaluation_only)
        self.shadow_stop_enabled = bool(shadow_stop or active_stop)
        self.active_stop_enabled = bool(active_stop)
        self.image_encoder_type = image_encoder_type
        self.rng = random.Random(config.training.seed if seed is None else seed)
        self.events: queue.Queue[str] = queue.Queue()
        # Never accumulate manual commands while a slow motion/vision step is
        # running. One deliberate key press corresponds to at most one item.
        self.manual_actions: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.manual_motion_busy = threading.Event()
        self.stop_event = threading.Event()
        self.motion_cancel = threading.Event()
        self.episode_end = threading.Event()
        self.snapshot_lock = threading.Lock()
        self.ui_snapshot = UiSnapshot()
        self._state = ActorState.PREFLIGHT
        self._state_lock = threading.RLock()
        self.camera: D435Camera | None = None
        self.robot: PiperRos2 | None = None
        self.policy: PolicyLink | None = None
        log_dir = (
            config.runtime.evaluation_output_dir
            if self.evaluation_only
            else config.runtime.output_dir
        )
        self.logger = RuntimeLogger(log_dir)
        self.logger.event(
            "actor_mode",
            mode="evaluation" if self.evaluation_only else "training",
            deterministic=self.deterministic,
            replay_uploads_enabled=not self.evaluation_only,
            shadow_stop_enabled=self.shadow_stop_enabled,
            active_stop_enabled=self.active_stop_enabled,
            stop_mode=(
                "active_return_verify_v3"
                if self.active_stop_enabled
                else "shadow_return_v2" if self.shadow_stop_enabled else "off"
            ),
            shadow_stop_version=(
                3 if self.active_stop_enabled else 2 if self.shadow_stop_enabled else None
            ),
            shadow_stop_config=(
                vars(config.shadow_stop) if self.shadow_stop_enabled else None
            ),
            frozen_checkpoint=(
                config.runtime.evaluation_checkpoint if self.evaluation_only else None
            ),
        )
        self.worker: threading.Thread | None = None
        self._disable_ready_at: float | None = None
        self.home_joint_deg = np.asarray(config.robot.home_joint_deg, dtype=np.float64).copy()

    def _set_state(self, state: ActorState, message: str) -> None:
        with self._state_lock:
            self._state = state
        with self.snapshot_lock:
            self.ui_snapshot.state = state.value
            self.ui_snapshot.message = message
        self.logger.event("state", state=state.value, message=message)
        print(f"[ACTOR] {state.value}: {message}", flush=True)

    def state(self) -> ActorState:
        with self._state_lock:
            return self._state

    def _discard_queued_manual_action(self) -> None:
        try:
            while True:
                self.manual_actions.get_nowait()
        except queue.Empty:
            pass

    def _initialize(self) -> None:
        cfg = self.config
        self.camera = D435Camera(
            serial=cfg.runtime.camera_serial,
            width=cfg.runtime.camera_width,
            height=cfg.runtime.camera_height,
            fps=cfg.runtime.camera_fps,
        )
        vision = RemoteVisionClient(
            cfg.vision.analyze_url,
            cfg.vision.grounding_url,
            feature_dim=cfg.vision.aesthetic_feature_dim,
            timeout_seconds=cfg.vision.request_timeout_seconds,
        )
        health = vision.health()
        self.logger.event("vision_health", health=health)
        self.robot = PiperRos2(cfg.robot, allow_writes=True)
        joints = self.robot.wait_for_feedback().positions_rad
        joint_deg = np.rad2deg(joints)
        controlled = np.asarray(cfg.robot.controlled_indices, dtype=np.int64)
        low = np.asarray(cfg.robot.startup_joint_low_deg)[controlled]
        high = np.asarray(cfg.robot.startup_joint_high_deg)[controlled]
        if np.any((joint_deg[controlled] < low) | (joint_deg[controlled] > high)):
            raise RobotSafetyError(
                "controlled joints are outside guarded startup bounds: "
                f"{joint_deg[controlled].round(3).tolist()}"
            )
        built = ObservationBuilder(cfg, vision).build(self.camera.capture_rgb(), joints)
        self.builder = ObservationBuilder(cfg, vision)
        self._show_observation(built)
        self.policy = PolicyLink(
            cfg,
            built.observation,
            image_encoder_type=self.image_encoder_type,
            evaluation_only=self.evaluation_only,
        )
        self.policy.wait_ready()
        with self.snapshot_lock:
            self.ui_snapshot.policy_step = self.policy.update_step
        mode_text = (
            f"FROZEN EVAL checkpoint {self.policy.update_step}; deterministic, no replay upload. "
            if self.evaluation_only
            else ""
        )
        if self.shadow_stop_enabled:
            mode_text += (
                "ACTIVE STOP may return slowly and end the episode after 3-frame verification. "
                if self.active_stop_enabled
                else "SHADOW STOP logs only; it never ends an episode. "
            )
        self._set_state(
            ActorState.WAIT_ARM,
            mode_text + "Preflight passed. Support the arm, then SPACE to enable and reset.",
        )

    def _show_observation(self, observation: CapturedObservation, reward: float | None = None):
        target_present = bool((observation.detection or {}).get("best"))
        target_valid = bool(observation.observation["target_state"][-1] >= 0.5)
        with self.snapshot_lock:
            self.ui_snapshot.image = observation.rgb
            self.ui_snapshot.detection = observation.detection
            self.ui_snapshot.score = round(observation.score, 4)
            self.ui_snapshot.target_present = target_present
            self.ui_snapshot.target_valid = target_valid
            self.ui_snapshot.reward = None if reward is None else round(float(reward), 4)
            if self.ui_snapshot.best_score is None or observation.score > self.ui_snapshot.best_score:
                self.ui_snapshot.best_score = round(observation.score, 4)

    def _capture(self) -> CapturedObservation:
        assert self.camera is not None and self.robot is not None
        joints = self.robot.snapshot().positions_rad
        result = self.builder.build(self.camera.capture_rgb(), joints)
        self._show_observation(result)
        with self.snapshot_lock:
            self.ui_snapshot.joints_deg = np.rad2deg(joints)
        return result

    def _random_target(self) -> np.ndarray:
        cfg = self.config.robot
        # J4/J6 are fixed camera-leveling joints. They use their configured
        # home angles, while randomness is applied only to J1/J2/J3/J5.
        target = self.home_joint_deg.copy()
        for index in cfg.controlled_indices:
            target[index] = self.home_joint_deg[index] + self.rng.uniform(
                -cfg.random_initial_deg, cfg.random_initial_deg
            )
        low = np.asarray(cfg.joint_low_deg)
        high = np.asarray(cfg.joint_high_deg)
        for index in cfg.controlled_indices:
            target[index] = np.clip(target[index], low[index], high[index])
        return np.deg2rad(target)

    def _reset_episode(self) -> None:
        assert self.robot is not None
        cfg = self.config.robot
        self._set_state(
            ActorState.RESETTING,
            f"Moving slowly to home, then random +/-{cfg.random_initial_deg:g} deg start",
        )
        self.motion_cancel.clear()
        # The first reset also levels the camera: all six joints move slowly to
        # home, including fixed J4/J6. Later SAC/manual actions remain 4-D.
        home = np.deg2rad(self.home_joint_deg.copy())
        self.robot.move_interpolated(
            home,
            speed_percent=cfg.reset_speed_percent,
            deg_per_second=cfg.reset_deg_per_second,
            cancel_event=self.motion_cancel,
        )
        self.robot.move_interpolated(
            self._random_target(),
            speed_percent=cfg.reset_speed_percent,
            deg_per_second=cfg.reset_deg_per_second,
            cancel_event=self.motion_cancel,
        )
        observation = self._capture()
        self._show_observation(observation)
        with self.snapshot_lock:
            self.ui_snapshot.episode_step = 0
        self._set_state(ActorState.WAIT_CONFIRM, "Reset complete. Check scene and SPACE to begin episode")

    def _motion(self, normalized_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.robot is not None
        cfg = self.config.robot
        action = np.clip(np.asarray(normalized_action, dtype=np.float64), -1.0, 1.0)
        start = self.robot.snapshot().positions_rad
        target = start.copy()
        delta_deg = action * np.asarray(cfg.max_step_deg)
        for action_index, joint_index in enumerate(cfg.controlled_indices):
            target[joint_index] += np.deg2rad(delta_deg[action_index])
        target_deg = np.rad2deg(target)
        low = np.asarray(cfg.joint_low_deg)
        high = np.asarray(cfg.joint_high_deg)
        for joint_index in cfg.controlled_indices:
            target_deg[joint_index] = np.clip(
                target_deg[joint_index], low[joint_index], high[joint_index]
            )
        target = np.deg2rad(target_deg)
        max_distance = float(np.max(np.abs(np.rad2deg(target - start))))
        degrees_per_second = max(max_distance / max(cfg.action_seconds, 0.05), 0.2)
        end = self.robot.move_interpolated(
            target,
            speed_percent=cfg.speed_percent,
            deg_per_second=degrees_per_second,
            cancel_event=self.motion_cancel,
        )
        controlled = np.asarray(cfg.controlled_indices)
        actual_delta = np.rad2deg(end[controlled] - start[controlled]).astype(np.float32)
        return end, actual_delta

    @staticmethod
    def _terminalize(transition: dict[str, Any]) -> dict[str, Any]:
        transition = dict(transition)
        transition["dones"] = np.bool_(True)
        transition["masks"] = np.float32(0.0)
        return transition

    def _validated_active_stop_target(
        self, joints_deg: tuple[float, ...] | list[float] | None
    ) -> np.ndarray:
        """Validate a recorded enabled-arm pose before commanding a return."""

        if joints_deg is None:
            raise RobotSafetyError("active STOP has no recorded return pose")
        target_deg = np.asarray(joints_deg, dtype=np.float64).reshape(-1)
        if target_deg.shape != (6,) or not np.isfinite(target_deg).all():
            raise RobotSafetyError(
                f"active STOP return pose must contain six finite joints: {target_deg}"
            )
        low = np.asarray(self.config.robot.joint_low_deg, dtype=np.float64)
        high = np.asarray(self.config.robot.joint_high_deg, dtype=np.float64)
        tolerance = float(self.config.shadow_stop.return_joint_tolerance_deg)
        outside = (target_deg < low - tolerance) | (target_deg > high + tolerance)
        if np.any(outside):
            bad = np.flatnonzero(outside)
            details = ", ".join(
                f"J{index + 1}={target_deg[index]:.3f} not near "
                f"[{low[index]:.3f}, {high[index]:.3f}]"
                for index in bad
            )
            raise RobotSafetyError(f"active STOP rejected return pose: {details}")
        # Feedback can differ from an exact configured boundary by a tiny
        # amount. Clip only after the tolerance check, never beyond policy
        # bounds, then let PiperRos2 perform its independent startup guard.
        return np.deg2rad(np.clip(target_deg, low, high))

    def _execute_active_stop(
        self,
        *,
        episode_number: int,
        trigger_step: int,
        return_target_step: int,
        return_target_raw_score: float,
        return_target_joint_deg: tuple[float, ...] | list[float] | None,
        return_target_image: str | None,
    ) -> ActiveStopExecution:
        """Return slowly to a recorded pose and verify it without disabling."""

        assert self.robot is not None
        cfg = self.config
        target_rad = self._validated_active_stop_target(return_target_joint_deg)
        target_deg = np.rad2deg(target_rad)
        start_deg = np.rad2deg(self.robot.snapshot().positions_rad)
        self.motion_cancel.clear()
        self._set_state(
            ActorState.STOP_RETURNING,
            f"STOP triggered; returning slowly to best step {return_target_step}",
        )
        self.logger.event(
            "active_stop_return_started",
            episode=episode_number,
            trigger_step=trigger_step,
            return_target_step=return_target_step,
            return_target_raw_score=return_target_raw_score,
            return_target_image=return_target_image,
            start_joint_deg=start_deg.tolist(),
            target_joint_deg=target_deg.tolist(),
            return_deg_per_second=cfg.shadow_stop.return_deg_per_second,
        )
        reached_rad = self.robot.move_interpolated(
            target_rad,
            speed_percent=cfg.robot.reset_speed_percent,
            deg_per_second=cfg.shadow_stop.return_deg_per_second,
            cancel_event=self.motion_cancel,
        )
        reached_deg = np.rad2deg(reached_rad)
        max_error = float(np.max(np.abs(reached_deg - target_deg)))
        self.robot.hold()
        self.logger.event(
            "active_stop_return_completed",
            episode=episode_number,
            trigger_step=trigger_step,
            target_joint_deg=target_deg.tolist(),
            reached_joint_deg=reached_deg.tolist(),
            max_joint_error_deg=max_error,
        )
        if max_error > cfg.shadow_stop.return_joint_tolerance_deg:
            raise RobotSafetyError(
                "active STOP return feedback error "
                f"{max_error:.3f} deg exceeds "
                f"{cfg.shadow_stop.return_joint_tolerance_deg:.3f} deg"
            )
        if self.stop_event.is_set() or self.episode_end.is_set():
            raise RobotSafetyError("active STOP return cancelled; arm is holding")

        self._set_state(
            ActorState.STOP_VERIFYING,
            f"At best step {return_target_step}; collecting fixed-pose verification 1/"
            f"{cfg.shadow_stop.verification_frames}",
        )
        scores: list[float] = []
        target_presence: list[bool] = []
        image_paths: list[str] = []
        final_observation: CapturedObservation | None = None
        for index in range(cfg.shadow_stop.verification_frames):
            if index and cfg.shadow_stop.verification_interval_seconds > 0.0:
                deadline = (
                    time.monotonic()
                    + cfg.shadow_stop.verification_interval_seconds
                )
                while time.monotonic() < deadline:
                    if self.stop_event.is_set() or self.episode_end.is_set():
                        self.robot.hold()
                        raise RobotSafetyError(
                            "active STOP verification cancelled; arm is holding"
                        )
                    time.sleep(0.02)
            if self.stop_event.is_set() or self.episode_end.is_set():
                self.robot.hold()
                raise RobotSafetyError(
                    "active STOP verification cancelled; arm is holding"
                )
            with self.snapshot_lock:
                self.ui_snapshot.message = (
                    f"Holding best pose; verification {index + 1}/"
                    f"{cfg.shadow_stop.verification_frames}"
                )
            observation = self._capture()
            final_observation = observation
            sample_joint_deg = np.rad2deg(self.robot.snapshot().positions_rad)
            sample_error = float(np.max(np.abs(sample_joint_deg - target_deg)))
            if sample_error > cfg.shadow_stop.return_joint_tolerance_deg:
                self.robot.hold()
                raise RobotSafetyError(
                    "active STOP pose drift "
                    f"{sample_error:.3f} deg exceeds "
                    f"{cfg.shadow_stop.return_joint_tolerance_deg:.3f} deg"
                )
            present = bool((observation.detection or {}).get("best"))
            image_path = self.logger.save_image(
                observation.rgb,
                f"episode_{episode_number:05d}_stop_verify_{index + 1:02d}.jpg",
            )
            scores.append(float(observation.score))
            target_presence.append(present)
            image_paths.append(str(image_path))
            self.logger.event(
                "active_stop_verification_sample",
                episode=episode_number,
                trigger_step=trigger_step,
                sample=index + 1,
                score=observation.score,
                target_present=present,
                target_edge_clear=bool(
                    observation.observation["target_state"][-1] >= 0.5
                ),
                edge_clear_required=False,
                joint_deg=sample_joint_deg.tolist(),
                max_joint_error_deg=sample_error,
                image=str(image_path),
            )

        assert final_observation is not None
        verification = assess_stop_verification(
            scores=scores,
            target_present=target_presence,
            expected_frames=cfg.shadow_stop.verification_frames,
            quality_score_threshold=cfg.shadow_stop.quality_score_threshold,
        )
        self.robot.hold()
        self.logger.event(
            "active_stop_verification_result",
            episode=episode_number,
            trigger_step=trigger_step,
            return_target_step=return_target_step,
            return_target_raw_score=return_target_raw_score,
            return_target_image=return_target_image,
            target_joint_deg=target_deg.tolist(),
            reached_joint_deg=reached_deg.tolist(),
            max_joint_error_deg=max_error,
            verification_images=image_paths,
            **verification.as_dict(),
        )
        return ActiveStopExecution(
            verification=verification,
            final_observation=final_observation,
            target_joint_deg=tuple(float(value) for value in target_deg),
            reached_joint_deg=tuple(float(value) for value in reached_deg),
            max_joint_error_deg=max_error,
            verification_images=tuple(image_paths),
        )

    def _episode(self) -> None:
        assert self.policy is not None
        cfg = self.config
        current = self._capture()
        previous_action = np.zeros(4, dtype=np.float32)
        pending: tuple[dict[str, Any], bool] | None = None
        pending_window_done = True
        lost_frames = 0
        step = 0
        episode_number = self.ui_snapshot.episode
        shadow = ShadowStopMonitor(cfg.shadow_stop) if self.shadow_stop_enabled else None
        shadow_image_paths: dict[int, str] = {}
        shadow_summary_logged = False
        active_stop_attempts = 0
        active_stop_summary: dict[str, Any] = {
            "active_stop_enabled": self.active_stop_enabled,
            "return_attempts": 0,
            "return_executed": False,
            "verification_confirmed": None,
            "verification_reason": None,
            "verification_scores": None,
            "verification_median_score": None,
        }

        def finish_shadow(reason: str) -> None:
            nonlocal shadow_summary_logged
            if shadow is None or shadow_summary_logged:
                return
            summary = shadow.summary()
            if self.active_stop_enabled:
                summary.update(active_stop_summary)
            self.logger.event(
                (
                    "active_stop_episode_summary"
                    if self.active_stop_enabled
                    else "shadow_stop_episode_summary"
                ),
                episode=episode_number,
                episode_end_reason=reason,
                **summary,
            )
            shadow_summary_logged = True

        if shadow is not None:
            initial_target_present = bool((current.detection or {}).get("best"))
            assert self.robot is not None
            initial_joints_deg = np.rad2deg(self.robot.snapshot().positions_rad)
            initial_image_path = self.logger.save_image(
                current.rgb,
                f"episode_{episode_number:05d}_step_0000.jpg",
            )
            shadow_image_paths[0] = str(initial_image_path)
            initial_decision = shadow.observe(
                step=0,
                score=current.score,
                target_present=initial_target_present,
                joints_deg=initial_joints_deg.tolist(),
            )
            self.logger.event(
                "shadow_stop_episode_start",
                episode=episode_number,
                score=current.score,
                target_present=initial_target_present,
                joints_deg=initial_joints_deg.tolist(),
                image=str(initial_image_path),
                decision=initial_decision.as_dict(),
            )
        self.episode_end.clear()
        self.motion_cancel.clear()
        while not self.stop_event.is_set():
            if pending is not None and not pending_window_done:
                deadline = time.monotonic() + cfg.runtime.operator_decision_seconds
                while time.monotonic() < deadline and not (
                    self.stop_event.is_set()
                    or self.episode_end.is_set()
                    or self.state() != ActorState.RUNNING
                ):
                    time.sleep(0.02)
                pending_window_done = True
            if self.episode_end.is_set():
                if pending is not None:
                    self.policy.insert(self._terminalize(pending[0]), intervention=pending[1])
                self.policy.flush()
                self.logger.event("episode_end", reason="operator", episode_step=step)
                finish_shadow("operator")
                return
            state = self.state()
            if state == ActorState.PAUSED:
                time.sleep(0.05)
                continue
            if state == ActorState.INTERVENTION:
                try:
                    action = self.manual_actions.get(timeout=0.1)
                except queue.Empty:
                    continue
                intervention = True
            elif state == ActorState.RUNNING:
                action = self.policy.action(current.observation, deterministic=self.deterministic)
                intervention = False
            else:
                time.sleep(0.05)
                continue
            if (intervention and self.state() != ActorState.INTERVENTION) or (
                not intervention and self.state() != ActorState.RUNNING
            ):
                if intervention:
                    self.manual_motion_busy.clear()
                continue
            if pending is not None:
                self.policy.insert(pending[0], intervention=pending[1])
                pending = None
            try:
                end_joints, actual_delta = self._motion(action)
                next_observation = self._capture()
            except RobotSafetyError:
                if self.stop_event.is_set():
                    finish_shadow("actor_stop")
                    return
                if self.episode_end.is_set():
                    continue
                if self.state() in (ActorState.INTERVENTION, ActorState.PAUSED):
                    current = self._capture()
                    continue
                raise
            finally:
                if intervention:
                    self.manual_motion_busy.clear()
            if intervention and self.state() == ActorState.INTERVENTION:
                with self.snapshot_lock:
                    self.ui_snapshot.message = (
                        "Manual step complete; release the key, then press once for the next step"
                    )
                print(
                    f"[MANUAL] complete action={np.asarray(action).tolist()}; ready for next tap",
                    flush=True,
                )
            step += 1
            target_present = bool((next_observation.detection or {}).get("best"))
            lost_frames = 0 if target_present else lost_frames + 1
            with self.snapshot_lock:
                total_after_step = self.ui_snapshot.online_steps + 1
            automatic_done = (
                step >= cfg.training.max_episode_steps
                or lost_frames >= cfg.reward.lost_target_terminal_frames
                or total_after_step >= cfg.training.max_online_steps
            )
            previous_score = current.score
            transition, reward = make_transition(
                cfg,
                current,
                next_observation,
                action=action,
                previous_action=previous_action,
                actual_delta_deg=actual_delta,
                done=automatic_done,
            )
            pending = (transition, intervention)
            pending_window_done = False
            current = next_observation
            previous_action = np.asarray(action, dtype=np.float32)
            with self.snapshot_lock:
                self.ui_snapshot.episode_step = step
                self.ui_snapshot.online_steps += 1
                self.ui_snapshot.policy_step = self.policy.update_step
            self._show_observation(next_observation, reward.total)
            shadow_decision = None
            if shadow is not None:
                shadow_decision = shadow.observe(
                    step=step,
                    score=next_observation.score,
                    target_present=target_present,
                    intervention=intervention,
                    joints_deg=np.rad2deg(end_joints).tolist(),
                )
                if (
                    shadow_decision.confirmation_count > 0
                    and not shadow_decision.already_triggered
                ):
                    gain_text = (
                        "--"
                        if shadow_decision.improvement_over_patience is None
                        else f"{shadow_decision.improvement_over_patience:.3f}"
                    )
                    candidate_label = (
                        "ACTIVE STOP CANDIDATE"
                        if self.active_stop_enabled
                        else "SHADOW STOP"
                    )
                    print(
                        f"[{candidate_label}] episode={episode_number} step={step} "
                        f"median3={shadow_decision.median_score:.3f} "
                        f"best={shadow_decision.running_best_median:.3f} "
                        f"gain15={gain_text} confirm="
                        f"{shadow_decision.confirmation_count}/"
                        f"{cfg.shadow_stop.confirmation_steps}",
                        flush=True,
                    )
            target_state = next_observation.observation["target_state"]
            action_text = ", ".join(f"{float(value):+.2f}" for value in np.asarray(action))
            print(
                f"[STEP] episode={self.ui_snapshot.episode} step={step} "
                f"score={next_observation.score:.4f} reward={reward.total:+.4f} "
                f"abs={reward.aesthetic_absolute:+.3f} "
                f"delta={reward.aesthetic_delta:+.3f} "
                f"edge={cfg.reward.target_weight * reward.target:+.3f} "
                f"present={reward.target_present} edge_clear={reward.target_edge_clear} "
                f"margin={100.0 * float(target_state[5]):.1f}% "
                f"action=[{action_text}]",
                flush=True,
            )
            image_path = self.logger.save_image(
                next_observation.rgb,
                f"episode_{self.ui_snapshot.episode:05d}_step_{step:04d}.jpg",
            )
            if shadow is not None:
                shadow_image_paths[step] = str(image_path)
            active_stop_request: dict[str, Any] | None = None
            if shadow_decision is not None and shadow_decision.would_stop:
                best_step = shadow_decision.best_pose_step
                best_image = (
                    shadow_image_paths.get(best_step) if best_step is not None else None
                )
                if (
                    self.active_stop_enabled
                    and not automatic_done
                    and active_stop_attempts
                    < cfg.shadow_stop.max_return_attempts_per_episode
                ):
                    if best_step is None or shadow_decision.best_pose_raw_score is None:
                        raise RobotSafetyError(
                            "active STOP triggered without a valid recorded best pose"
                        )
                    active_stop_request = {
                        "episode_number": episode_number,
                        "trigger_step": step,
                        "return_target_step": best_step,
                        "return_target_raw_score": float(
                            shadow_decision.best_pose_raw_score
                        ),
                        "return_target_joint_deg": shadow_decision.best_pose_joint_deg,
                        "return_target_image": best_image,
                    }
                    message = (
                        f"ACTIVE STOP: trigger {step} -> returning to best step "
                        f"{best_step}, raw {shadow_decision.best_pose_raw_score:.2f}"
                    )
                    with self.snapshot_lock:
                        self.ui_snapshot.message = message
                    self.logger.event(
                        "active_stop_triggered",
                        episode=episode_number,
                        trigger_step=step,
                        trigger_image=str(image_path),
                        return_target_step=best_step,
                        return_target_image=best_image,
                        return_target_raw_score=shadow_decision.best_pose_raw_score,
                        return_target_joint_deg=shadow_decision.best_pose_joint_deg,
                        decision=shadow_decision.as_dict(),
                        target_edge_clear=reward.target_edge_clear,
                        edge_clear_required=False,
                        action_taken=True,
                    )
                    print(f"[ACTIVE STOP] {message}", flush=True)
                elif not self.active_stop_enabled:
                    message = (
                        f"SHADOW WOULD_RETURN_TO_BEST: trigger {step} -> best step "
                        f"{best_step}, raw {shadow_decision.best_pose_raw_score:.2f}; "
                        "episode continues unchanged"
                    )
                    with self.snapshot_lock:
                        self.ui_snapshot.message = message
                    self.logger.event(
                        "shadow_stop_would_return_to_best",
                        episode=episode_number,
                        trigger_step=step,
                        trigger_image=str(image_path),
                        return_target_step=best_step,
                        return_target_image=best_image,
                        return_target_raw_score=shadow_decision.best_pose_raw_score,
                        return_target_joint_deg=shadow_decision.best_pose_joint_deg,
                        decision=shadow_decision.as_dict(),
                        target_edge_clear=reward.target_edge_clear,
                        edge_clear_required=False,
                        action_taken=False,
                        return_executed=False,
                        episode_continues=True,
                    )
                    print(f"[SHADOW WOULD_RETURN_TO_BEST] {message}", flush=True)
                else:
                    self.logger.event(
                        "active_stop_skipped",
                        episode=episode_number,
                        trigger_step=step,
                        reason=(
                            "episode_already_terminal"
                            if automatic_done
                            else "return_attempt_limit"
                        ),
                        return_attempts=active_stop_attempts,
                    )
            self.logger.event(
                "transition",
                episode=self.ui_snapshot.episode,
                step=step,
                action=np.asarray(action).tolist(),
                actual_delta_deg=actual_delta.tolist(),
                intervention=intervention,
                terminal=automatic_done,
                previous_score=previous_score,
                score=next_observation.score,
                reward=reward.as_dict(),
                shadow_stop=(
                    shadow_decision.as_dict() if shadow_decision is not None else None
                ),
                image=str(image_path),
            )
            if active_stop_request is not None:
                active_stop_attempts += 1
                active_stop_summary.update(
                    {
                        "return_attempts": active_stop_attempts,
                        "return_executed": True,
                        "return_target_step": active_stop_request[
                            "return_target_step"
                        ],
                        "return_target_raw_score": active_stop_request[
                            "return_target_raw_score"
                        ],
                        "return_target_joint_deg": active_stop_request[
                            "return_target_joint_deg"
                        ],
                    }
                )
                try:
                    execution = self._execute_active_stop(**active_stop_request)
                except RobotSafetyError:
                    if self.stop_event.is_set():
                        if pending is not None:
                            self.policy.insert(
                                self._terminalize(pending[0]),
                                intervention=pending[1],
                            )
                            self.policy.flush()
                            pending = None
                        finish_shadow("actor_stop_during_active_stop")
                        return
                    if self.episode_end.is_set():
                        if pending is not None:
                            self.policy.insert(
                                self._terminalize(pending[0]),
                                intervention=pending[1],
                            )
                            self.policy.flush()
                            pending = None
                        self.logger.event(
                            "episode_end",
                            reason="operator_during_active_stop",
                            episode_step=step,
                        )
                        finish_shadow("operator_during_active_stop")
                        return
                    raise

                verification = execution.verification
                active_stop_summary.update(
                    {
                        "verification_confirmed": verification.confirmed,
                        "verification_reason": verification.reason,
                        "verification_scores": verification.scores,
                        "verification_median_score": verification.median_score,
                        "return_max_joint_error_deg": execution.max_joint_error_deg,
                    }
                )
                if verification.confirmed:
                    assert pending is not None
                    self.policy.insert(
                        self._terminalize(pending[0]), intervention=pending[1]
                    )
                    self.policy.flush()
                    pending = None
                    self.logger.event(
                        "episode_end",
                        reason="active_stop_confirmed",
                        episode_step=step,
                        verification_median_score=verification.median_score,
                        return_target_step=active_stop_request[
                            "return_target_step"
                        ],
                    )
                    finish_shadow("active_stop_confirmed")
                    print(
                        f"[ACTIVE STOP CONFIRMED] episode={episode_number} "
                        f"step={step} fixed_pose_median="
                        f"{verification.median_score:.3f}; episode ends, arm holds",
                        flush=True,
                    )
                    return

                if verification.reason == "target_missing":
                    assert pending is not None
                    self.policy.insert(
                        self._terminalize(pending[0]), intervention=pending[1]
                    )
                    self.policy.flush()
                    pending = None
                    self.logger.event(
                        "episode_end",
                        reason="active_stop_target_missing",
                        episode_step=step,
                        verification_median_score=verification.median_score,
                    )
                    finish_shadow("active_stop_target_missing")
                    print(
                        f"[ACTIVE STOP REJECTED] episode={episode_number} "
                        "target missing during fixed-pose verification; "
                        "episode ends with arm holding",
                        flush=True,
                    )
                    return

                # A low fixed-pose median rejects STOP. Resume from the actual
                # returned observation, but do not make another return attempt
                # in this conservative first real-robot version.
                current = execution.final_observation
                previous_action = np.zeros(4, dtype=np.float32)
                lost_frames = 0
                self.motion_cancel.clear()
                self._set_state(
                    ActorState.RUNNING,
                    f"STOP rejected: fixed-pose median {verification.median_score:.2f} "
                    f"< {cfg.shadow_stop.quality_score_threshold:.2f}; policy resumed",
                )
                self.logger.event(
                    "active_stop_policy_resumed",
                    episode=episode_number,
                    episode_step=step,
                    reason=verification.reason,
                    verification_median_score=verification.median_score,
                    further_return_attempts_allowed=False,
                )
                continue
            if automatic_done:
                self.policy.insert(pending[0], intervention=pending[1])
                pending = None
                self.policy.flush()
                reason = (
                    "target_missing"
                    if lost_frames >= cfg.reward.lost_target_terminal_frames
                    else "limit"
                )
                self.logger.event("episode_end", reason=reason, episode_step=step)
                finish_shadow(reason)
                print(
                    f"[EPISODE END] episode={self.ui_snapshot.episode} "
                    f"step={step} reason={reason}",
                    flush=True,
                )
                if self.ui_snapshot.online_steps >= cfg.training.max_online_steps:
                    self.stop_event.set()
                    self.policy.request_stop("max_online_steps")
                return
        if pending is not None:
            self.policy.insert(self._terminalize(pending[0]), intervention=pending[1])
            self.policy.flush()
        finish_shadow("actor_stop")

    def _worker_loop(self) -> None:
        assert self.robot is not None and self.policy is not None
        try:
            while not self.stop_event.is_set():
                state = self.state()
                if state == ActorState.RESETTING:
                    self._reset_episode()
                elif state == ActorState.RUNNING:
                    with self.snapshot_lock:
                        self.ui_snapshot.episode += 1
                    self._episode()
                    if not self.stop_event.is_set():
                        self._set_state(ActorState.RESETTING, "Episode ended; starting safe reset")
                elif state == ActorState.POSITIONING:
                    try:
                        action = self.manual_actions.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    try:
                        self._motion(action)
                        self._capture()
                        if self.state() == ActorState.POSITIONING:
                            with self.snapshot_lock:
                                self.ui_snapshot.message = (
                                    "Setup step complete; release key before the next press"
                                )
                            print(
                                f"[MANUAL] setup complete action={np.asarray(action).tolist()}; "
                                "ready for next tap",
                                flush=True,
                            )
                    finally:
                        self.manual_motion_busy.clear()
                else:
                    time.sleep(0.05)
            try:
                if self.robot.enabled_by_us:
                    self.robot.hold()
            finally:
                if self.robot.enabled_by_us:
                    message = "Training stopped; arm ENABLED and holding"
                else:
                    message = "Training stopped before this process enabled the arm"
                self._set_state(ActorState.STOPPED_HOLD, message)
        except Exception as exc:
            self.logger.event("fault", error=repr(exc), traceback=traceback.format_exc())
            try:
                self.robot.hold()
            except Exception as hold_exc:
                self.logger.event("hold_fault", error=repr(hold_exc))
            self._set_state(ActorState.FAULT_HOLD, f"FAULT; arm intended to remain enabled: {exc}")

    def _manual_vector(self, action: str) -> np.ndarray | None:
        mapping = {
            "manual_j1_pos": (0, 1.0), "manual_j1_neg": (0, -1.0),
            "manual_j2_pos": (1, 1.0), "manual_j2_neg": (1, -1.0),
            "manual_j3_pos": (2, 1.0), "manual_j3_neg": (2, -1.0),
            "manual_j5_pos": (3, 1.0), "manual_j5_neg": (3, -1.0),
        }
        if action not in mapping:
            return None
        vector = np.zeros(4, dtype=np.float32)
        index, value = mapping[action]
        vector[index] = value
        return vector

    def _queue_manual_step(self, action_name: str, vector: np.ndarray) -> bool:
        if self.manual_motion_busy.is_set():
            with self.snapshot_lock:
                self.ui_snapshot.message = (
                    "Manual step still executing; release the key and wait for completion"
                )
            self.logger.event("manual_step_ignored", action=action_name, reason="busy")
            print(f"[MANUAL] ignored {action_name}: previous step busy", flush=True)
            return False
        self.manual_motion_busy.set()
        try:
            self.manual_actions.put_nowait(vector)
        except queue.Full:
            self.manual_motion_busy.clear()
            self.logger.event("manual_step_ignored", action=action_name, reason="queue_full")
            return False
        with self.snapshot_lock:
            self.ui_snapshot.message = f"Manual step executing: {action_name}"
        self.logger.event("manual_step_accepted", action=action_name)
        print(f"[MANUAL] accepted {action_name}: executing one step", flush=True)
        return True

    def _dispatch(self, action: str) -> None:
        state = self.state()
        if action == "space":
            if state == ActorState.WAIT_ARM:
                assert self.robot is not None
                self.robot.enable(True)
                self.robot.hold()
                self._set_state(ActorState.RESETTING, "Arm enabled; beginning initial reset")
            elif state == ActorState.WAIT_CONFIRM:
                with self.snapshot_lock:
                    target_present = bool(self.ui_snapshot.target_present)
                if target_present:
                    self._set_state(ActorState.RUNNING, "Policy active; A intervenes, O ends episode")
                else:
                    with self.snapshot_lock:
                        self.ui_snapshot.message = (
                            "No target box: adjust flower, then C to recapture; episode remains stopped"
                        )
        elif action == "recapture" and state == ActorState.WAIT_CONFIRM:
            observation = self._capture()
            target_present = bool((observation.detection or {}).get("best"))
            edge_clear = bool(observation.observation["target_state"][-1] >= 0.5)
            with self.snapshot_lock:
                if edge_clear:
                    self.ui_snapshot.message = "Target box clear of edge; SPACE may start episode"
                elif target_present:
                    self.ui_snapshot.message = (
                        "Target box is near edge (penalty active); SPACE may start or adjust it"
                    )
                else:
                    self.ui_snapshot.message = "No target box; adjust flower, then press C again"
        elif action == "pause":
            if state in (ActorState.RUNNING, ActorState.INTERVENTION):
                self.motion_cancel.set()
                self._discard_queued_manual_action()
                self._set_state(ActorState.PAUSED, "Paused; arm enabled and holding. P resumes policy")
            elif state == ActorState.PAUSED:
                self.motion_cancel.clear()
                self._set_state(ActorState.RUNNING, "Policy resumed")
        elif action == "intervention" and state == ActorState.WAIT_CONFIRM:
            self._discard_queued_manual_action()
            self.manual_motion_busy.clear()
            self.motion_cancel.clear()
            self._set_state(
                ActorState.POSITIONING,
                "Discrete setup: tap W/S J1, E/D J2, R/F J3, T/G J5; release + wait",
            )
        elif action == "intervention" and state in (ActorState.RUNNING, ActorState.PAUSED):
            self._discard_queued_manual_action()
            self.manual_motion_busy.clear()
            self.motion_cancel.set()
            self._set_state(
                ActorState.INTERVENTION,
                "Discrete jog: W/S J1, E/D J2, R/F J3, T/G J5; tap, release, wait",
            )
        elif action == "resume" and state == ActorState.INTERVENTION:
            if self.manual_motion_busy.is_set():
                with self.snapshot_lock:
                    self.ui_snapshot.message = (
                        "Current manual step is still executing; wait, then press B again"
                    )
            else:
                self._discard_queued_manual_action()
                self.motion_cancel.clear()
                self._set_state(ActorState.RUNNING, "Intervention ended; policy resumed")
        elif action == "resume" and state == ActorState.POSITIONING:
            if self.manual_motion_busy.is_set():
                with self.snapshot_lock:
                    self.ui_snapshot.message = (
                        "Current setup step is still executing; wait, then press B again"
                    )
            else:
                self._discard_queued_manual_action()
                self._set_state(ActorState.WAIT_CONFIRM, "Setup jog ended; C checks target, SPACE starts")
        elif action == "set_home" and state in (ActorState.WAIT_CONFIRM, ActorState.POSITIONING):
            assert self.robot is not None
            with self.snapshot_lock:
                target_valid = bool(self.ui_snapshot.target_valid)
            if not target_valid:
                with self.snapshot_lock:
                    self.ui_snapshot.message = "Cannot set home: target is invalid; jog/adjust then try H"
            else:
                current = np.rad2deg(self.robot.snapshot().positions_rad)
                controlled = np.asarray(self.config.robot.controlled_indices)
                low = np.asarray(self.config.robot.joint_low_deg)[controlled]
                high = np.asarray(self.config.robot.joint_high_deg)[controlled]
                if np.any((current[controlled] < low) | (current[controlled] > high)):
                    raise RobotSafetyError("cannot set runtime home outside policy workspace")
                self.home_joint_deg[controlled] = current[controlled]
                self.logger.event(
                    "runtime_home_set",
                    home_joint_deg=self.home_joint_deg.tolist(),
                    controlled_indices=controlled.tolist(),
                )
                with self.snapshot_lock:
                    self.ui_snapshot.message = (
                        "Current valid view is runtime home; B exits setup, then SPACE starts"
                        if state == ActorState.POSITIONING
                        else "Current valid view is runtime home; SPACE starts"
                    )
        elif action == "end_episode" and state in (
            ActorState.RUNNING,
            ActorState.PAUSED,
            ActorState.INTERVENTION,
            ActorState.STOP_RETURNING,
            ActorState.STOP_VERIFYING,
        ):
            self.episode_end.set()
            self.motion_cancel.set()
            self._discard_queued_manual_action()
            with self.snapshot_lock:
                self.ui_snapshot.message = "Ending episode, then resetting; arm remains enabled"
        elif action == "end_episode" and state == ActorState.WAIT_CONFIRM:
            self.motion_cancel.clear()
            self._set_state(
                ActorState.RESETTING,
                "Trying another random start; arm remains enabled",
            )
        elif action == "stop" and state not in (ActorState.DISABLED, ActorState.CLOSED):
            self.stop_event.set()
            self.motion_cancel.set()
            self._discard_queued_manual_action()
            if self.robot is not None and self.robot.enabled_by_us:
                try:
                    self.robot.hold()
                except Exception as exc:
                    self.logger.event("immediate_hold_fault", error=repr(exc))
            if self.policy is not None:
                self.policy.request_stop("operator_stop")
        elif action == "cancel_disable":
            self._disable_ready_at = None
            with self.snapshot_lock:
                self.ui_snapshot.disable_ready_at = None
                self.ui_snapshot.message = "Disable cancelled; arm remains enabled"
        elif action == "disable" and state in (ActorState.STOPPED_HOLD, ActorState.FAULT_HOLD):
            assert self.robot is not None
            if not self.robot.enabled_by_us:
                self._set_state(ActorState.DISABLED, "Arm was not enabled by this process")
                return
            now = time.monotonic()
            if self._disable_ready_at is None:
                self._disable_ready_at = now + self.config.runtime.disable_countdown_seconds
                with self.snapshot_lock:
                    self.ui_snapshot.disable_ready_at = self._disable_ready_at
                    self.ui_snapshot.message = "WARNING: arm will sag. Protect it; after countdown press D again"
            elif now >= self._disable_ready_at:
                self.robot.enable(False)
                self._disable_ready_at = None
                with self.snapshot_lock:
                    self.ui_snapshot.disable_ready_at = None
                self._set_state(ActorState.DISABLED, "Arm deliberately disabled after two-step confirmation")
        else:
            vector = self._manual_vector(action)
            if vector is not None and state in (ActorState.INTERVENTION, ActorState.POSITIONING):
                self.motion_cancel.clear()
                self._queue_manual_step(action, vector)

    def run(self) -> None:
        ui: OperatorUI | None = None
        try:
            self._initialize()
            ui = OperatorUI(self.events, self.ui_snapshot, self.snapshot_lock)
            self.worker = threading.Thread(target=self._worker_loop, name="method-a-worker", daemon=True)
            self.worker.start()
            visible = True
            while visible and self.state() not in (ActorState.DISABLED, ActorState.CLOSED):
                visible = ui.poll()
                while True:
                    try:
                        action = self.events.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        self._dispatch(action)
                    except Exception as exc:
                        self.logger.event("operator_action_fault", action=action, error=repr(exc))
                        if action == "disable":
                            self._set_state(ActorState.FAULT_HOLD, f"Disable failed; assume ENABLED: {exc}")
                        else:
                            self.stop_event.set()
                            self.motion_cancel.set()
                            self._set_state(ActorState.FAULT_HOLD, f"Action failed; assume ENABLED: {exc}")
            if not self.stop_event.is_set() and self.state() != ActorState.DISABLED:
                self._dispatch("stop")
        finally:
            if self.worker is not None:
                # Do not tear ROS down underneath a still-running motion/capture
                # worker. HTTP calls have a bounded timeout from configuration.
                self.worker.join()
            # Never call disable here. Closing preserves the last enable state.
            if self.policy is not None:
                self.policy.close()
            if self.camera is not None:
                self.camera.close()
            if self.robot is not None:
                self.robot.close()
            if ui is not None:
                ui.close()


def run_preflight(config: MethodAConfig, *, check_robot_feedback: bool) -> dict[str, Any]:
    """Read-only check: no control publisher call, no enable/disable call."""
    result: dict[str, Any] = {}
    result["fixed_gripper_opening_m"] = config.robot.gripper_opening_m
    vision = RemoteVisionClient(
        config.vision.analyze_url,
        config.vision.grounding_url,
        feature_dim=config.vision.aesthetic_feature_dim,
        timeout_seconds=config.vision.request_timeout_seconds,
    )
    result["vision"] = vision.health()
    camera = D435Camera(
        serial=config.runtime.camera_serial,
        width=config.runtime.camera_width,
        height=config.runtime.camera_height,
        fps=config.runtime.camera_fps,
    )
    robot = None
    try:
        rgb = camera.capture_rgb()
        result["camera"] = {"shape": list(rgb.shape), "dtype": str(rgb.dtype)}
        if check_robot_feedback:
            robot = PiperRos2(config.robot, allow_writes=False)
            joints = robot.wait_for_feedback().positions_rad
            result["joint_feedback_deg"] = np.rad2deg(joints).round(4).tolist()
            degrees = np.rad2deg(joints)
            controlled = np.asarray(config.robot.controlled_indices, dtype=np.int64)
            startup_low = np.asarray(config.robot.startup_joint_low_deg)[controlled]
            startup_high = np.asarray(config.robot.startup_joint_high_deg)[controlled]
            within_startup = bool(
                np.all(
                    (degrees[controlled] >= startup_low)
                    & (degrees[controlled] <= startup_high)
                )
            )
            policy_low = np.asarray(config.robot.joint_low_deg)[controlled]
            policy_high = np.asarray(config.robot.joint_high_deg)[controlled]
            within_policy = bool(
                np.all(
                    (degrees[controlled] >= policy_low)
                    & (degrees[controlled] <= policy_high)
                )
            )
            result["controlled_joints_within_startup_bounds"] = within_startup
            # Compatibility with older checklists. Startup bounds apply only to
            # J1/J2/J3/J5; J4/J6 are never changed and retain live feedback.
            result["joint_feedback_within_startup_bounds"] = within_startup
            result["controlled_joints_within_policy_workspace"] = within_policy
            # Compatibility with the existing startup checklist.
            result["joint_feedback_within_configured_bounds"] = within_startup
        else:
            joints = np.deg2rad(np.asarray(config.robot.home_joint_deg))
        analyzed = ObservationBuilder(config, vision).build(rgb, joints)
        target_present = bool((analyzed.detection or {}).get("best"))
        target_edge_clear = bool(analyzed.observation["target_state"][-1])
        result["analysis"] = {
            "score": analyzed.score,
            "feature_shape": list(analyzed.observation["aesthetic_feature"].shape),
            "target_present": target_present,
            "target_edge_clear": target_edge_clear,
            "target_valid": target_edge_clear,
            "artimuse_seconds": analyzed.analyze_seconds,
            "grounding_seconds": analyzed.detection_seconds,
        }
        return result
    finally:
        camera.close()
        if robot is not None:
            robot.close()
