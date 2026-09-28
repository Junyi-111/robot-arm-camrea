"""Safety-gated ROS 2 adapter for the PiPER arm.

Importing this module never initializes ROS and never writes to the robot.
Motion/enable/disable calls require an explicit process-level opt-in.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import RobotConfig


ARM_WRITE_ENV = "PIPER_ARM_WRITES"
ARM_WRITE_VALUE = "I_ACCEPT_REAL_MOTION"


class RobotSafetyError(RuntimeError):
    """A robot operation failed a precondition or safety check."""


@dataclass(frozen=True)
class JointSnapshot:
    positions_rad: np.ndarray
    received_at: float


class PiperRos2:
    JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))

    def __init__(self, config: RobotConfig, *, allow_writes: bool = False) -> None:
        if allow_writes and os.environ.get(ARM_WRITE_ENV) != ARM_WRITE_VALUE:
            raise RobotSafetyError(
                f"real-arm writes require {ARM_WRITE_ENV}={ARM_WRITE_VALUE}"
            )
        try:
            import rclpy
            from piper_msgs.srv import Enable
            from sensor_msgs.msg import JointState
        except ImportError as exc:
            raise RobotSafetyError(
                "ROS 2/PiPER Python packages unavailable; source Humble and handeye setup"
            ) from exc
        self._rclpy = rclpy
        self._Enable = Enable
        self._JointState = JointState
        self.config = config
        self.allow_writes = bool(allow_writes)
        self._lock = threading.RLock()
        self._latest: JointSnapshot | None = None
        self._enabled_by_us = False
        self._closed = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True
        else:
            self._owns_rclpy = False
        self._node = rclpy.create_node("method_a_piper_actor")
        self._publisher = self._node.create_publisher(JointState, "/joint_states", 1)
        self._subscription = self._node.create_subscription(
            JointState, "/joint_states_feedback", self._feedback, 10
        )
        self._enable_client = self._node.create_client(Enable, "/enable_srv")
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), name="method-a-ros-spin", daemon=True
        )
        self._spin_thread.start()

    def _require_write(self) -> None:
        if not self.allow_writes or os.environ.get(ARM_WRITE_ENV) != ARM_WRITE_VALUE:
            raise RobotSafetyError("real-arm write gate is closed")
        if self._closed:
            raise RobotSafetyError("robot adapter is closed")

    @property
    def enabled_by_us(self) -> bool:
        return self._enabled_by_us

    def _feedback(self, message) -> None:
        names = list(message.name)
        positions = list(message.position)
        mapping = {name: positions[i] for i, name in enumerate(names) if i < len(positions)}
        if all(name in mapping for name in self.JOINT_NAMES):
            values = np.asarray([mapping[name] for name in self.JOINT_NAMES], dtype=np.float64)
        elif len(positions) >= 6:
            values = np.asarray(positions[:6], dtype=np.float64)
        else:
            return
        if np.isfinite(values).all():
            with self._lock:
                self._latest = JointSnapshot(values.copy(), time.monotonic())

    def wait_for_feedback(self, timeout_seconds: float = 5.0) -> JointSnapshot:
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            try:
                return self.snapshot()
            except RobotSafetyError:
                time.sleep(0.02)
        raise RobotSafetyError("no fresh /joint_states_feedback received")

    def snapshot(self) -> JointSnapshot:
        with self._lock:
            latest = self._latest
        if latest is None:
            raise RobotSafetyError("joint feedback has not arrived")
        age = time.monotonic() - latest.received_at
        if age > self.config.feedback_timeout_seconds:
            raise RobotSafetyError(f"joint feedback is stale ({age:.2f}s)")
        return JointSnapshot(latest.positions_rad.copy(), latest.received_at)

    def _validate_target(self, positions_rad: np.ndarray) -> np.ndarray:
        target = np.asarray(positions_rad, dtype=np.float64).reshape(-1)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise RobotSafetyError(f"joint target must be six finite radians, got {target}")
        degrees = np.rad2deg(target)
        # Only J1/J2/J3/J5 are learned/action-controlled. A disabled PiPER can
        # sag at J4/J6, so that measured pose must not block startup. The actor
        # subsequently brings J4/J6 slowly to their fixed home angles to level
        # the camera. Keep this guard on the four learned joints.
        controlled = np.asarray(self.config.controlled_indices, dtype=np.int64)
        low = np.asarray(self.config.startup_joint_low_deg)[controlled]
        high = np.asarray(self.config.startup_joint_high_deg)[controlled]
        outside = (degrees[controlled] < low) | (degrees[controlled] > high)
        bad = controlled[np.flatnonzero(outside)]
        if bad.size:
            details = ", ".join(
                f"J{i + 1}={degrees[i]:.2f} not in "
                f"[{self.config.startup_joint_low_deg[i]:.2f},"
                f"{self.config.startup_joint_high_deg[i]:.2f}]"
                for i in bad
            )
            raise RobotSafetyError(f"joint safety bound rejected command: {details}")
        return target

    def enable(self, value: bool, timeout_seconds: float = 8.0) -> None:
        self._require_write()
        if not self._enable_client.wait_for_service(timeout_sec=timeout_seconds):
            raise RobotSafetyError("/enable_srv is unavailable")
        request = self._Enable.Request()
        request.enable_request = bool(value)
        future = self._enable_client.call_async(request)
        deadline = time.monotonic() + timeout_seconds
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise RobotSafetyError("/enable_srv timed out")
        response = future.result()
        if response is None or not response.enable_response:
            raise RobotSafetyError(f"driver did not confirm {'enable' if value else 'disable'}")
        self._enabled_by_us = bool(value)

    def command(self, positions_rad: np.ndarray, *, speed_percent: int) -> None:
        self._require_write()
        if not self._enabled_by_us:
            raise RobotSafetyError("refusing motion before this process enabled the arm")
        if not 1 <= int(speed_percent) <= 10:
            raise RobotSafetyError("method A speed must remain in [1, 10] percent")
        target = self._validate_target(positions_rad)
        message = self._JointState()
        message.header.stamp = self._node.get_clock().now().to_msg()
        message.name = list(self.JOINT_NAMES) + ["gripper"]
        # Gripper is fixed, never part of the SAC action. The installed enable
        # service sends a zero gripper command, so every hold/motion command
        # immediately restores the camera-clear opening used for demo capture.
        message.position = target.tolist() + [float(self.config.gripper_opening_m)]
        # The installed PiPER driver reads velocity[6] as a shared percentage.
        message.velocity = [0.0] * 6 + [float(speed_percent)]
        message.effort = [0.0] * 6 + [0.5]
        self._publisher.publish(message)

    def hold(self) -> None:
        """Command the latest measured pose; this never disables the arm."""
        self.command(self.snapshot().positions_rad, speed_percent=self.config.speed_percent)

    def move_interpolated(
        self,
        target_rad: np.ndarray,
        *,
        speed_percent: int,
        deg_per_second: float,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        self._require_write()
        start = self.snapshot().positions_rad
        target = self._validate_target(target_rad)
        distance = float(np.max(np.abs(np.rad2deg(target - start))))
        duration = distance / max(float(deg_per_second), 1e-6)
        steps = max(1, int(np.ceil(duration / 0.05)))
        for index in range(1, steps + 1):
            if cancel_event is not None and cancel_event.is_set():
                self.hold()
                raise RobotSafetyError("motion cancelled; arm remains enabled and holding")
            alpha = index / steps
            self.command(start + alpha * (target - start), speed_percent=speed_percent)
            time.sleep(max(duration / steps, 0.02))
        time.sleep(self.config.settle_seconds)
        return self.snapshot().positions_rad

    def close(self) -> None:
        """Tear down communications. Deliberately never disables the arm."""
        if self._closed:
            return
        if self.allow_writes and self._enabled_by_us:
            try:
                self.hold()
            except Exception:
                pass
        self._closed = True
        self._node.destroy_node()
        if self._owns_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
        self._spin_thread.join(timeout=2.0)
