"""Hardware-free regression tests for Method A."""

from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from aesthetic_rl.method_a.config import MethodAConfig, load_config
from aesthetic_rl.method_a.actor import ActorState, ManualKeyGate, MethodAActor
from aesthetic_rl.method_a.robot_ros2 import PiperRos2, RobotSafetyError
from aesthetic_rl.method_a.replay import (
    ReplayStore,
    mixed_batch,
    mixed_batch_with_interventions,
    validate_transition,
)
from aesthetic_rl.method_a.reward import compute_reward, target_state_from_detection
from aesthetic_rl.method_a.runtime import CapturedObservation, ObservationBuilder, make_transition
from aesthetic_rl.method_a.shadow_stop import (
    ShadowStopMonitor,
    assess_stop_verification,
)
from aesthetic_rl.method_a.vision_client import VisionAnalysis
from aesthetic_rl.scripts.method_a_calibrate_score_noise import (
    build_summary,
    rolling_median,
    score_statistics,
)


class FakeVision:
    def analyze(self, rgb):
        height, width = rgb.shape[:2]
        detection = {
            "target_valid": True,
            "width": width,
            "height": height,
            "best": {"xyxy": [100, 80, 400, 350], "score": 0.9},
        }
        return VisionAnalysis(37.5, np.ones(3584, np.float32), detection, 0.1, 0.2)


def fake_capture(score=30.0, valid=True, present=True, margin=0.2):
    target = (
        np.asarray([0.5, 0.5, 0.3, 0.3, 0.9, margin, float(valid)], np.float32)
        if present
        else np.zeros(7, np.float32)
    )
    obs = {
        "wrist_view": np.zeros((128, 128, 3), np.uint8),
        "robot_state": np.zeros(6, np.float32),
        "target_state": target,
        "aesthetic_feature": np.zeros(3584, np.float32),
    }
    detection = (
        {
            "target_valid": bool(valid),
            "width": 640,
            "height": 480,
            "best": {"xyxy": [100, 80, 400, 350], "score": 0.9},
        }
        if present
        else {"target_valid": False, "width": 640, "height": 480, "best": None}
    )
    return CapturedObservation(
        obs, np.zeros((480, 640, 3), np.uint8), score, detection, 0.1, 0.1
    )


class MethodATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("aesthetic_rl/configs/method_a.yaml")

    def test_score_noise_statistics_and_threshold_recommendation(self):
        exact = []
        fresh = []
        for index in range(30):
            base = {
                "index": index,
                "joint_deg": [0.0] * 6,
                "target_present": True,
                "target_edge_clear": True,
                "box_geometry": [0.5, 0.5, 0.3, 0.4],
            }
            exact.append({**base, "score": 40.0})
            fresh.append({**base, "score": 40.0 + 0.1 * (index % 3 - 1)})
        stats = score_statistics([record["score"] for record in fresh])
        self.assertEqual(stats["count"], 30)
        self.assertGreater(stats["consecutive_abs_diff_p95"], 0.0)
        self.assertEqual(rolling_median([1.0, 2.0, 3.0, 4.0], 3).tolist(), [2.0, 3.0])
        summary = build_summary(
            exact,
            fresh,
            np.zeros(6),
            self.config,
            drift_limit_deg=0.1,
        )
        self.assertTrue(summary["measurement_valid"])
        self.assertGreaterEqual(summary["recommended"]["improvement_epsilon"], 0.1)
        self.assertEqual(summary["target"]["present_rate"], 1.0)

    def test_config_and_fixed_action_space(self):
        self.assertEqual(self.config.robot.controlled_indices, (0, 1, 2, 4))
        self.assertEqual(len(self.config.robot.max_step_deg), 4)
        self.assertEqual(self.config.vision.aesthetic_feature_dim, 3584)
        self.assertEqual(self.config.reward.lost_target_terminal_frames, 1)
        self.assertEqual(self.config.robot.joint_high_deg[2], 0.0)
        self.assertGreater(self.config.robot.startup_joint_high_deg[2], 0.0)
        self.assertTrue(
            self.config.runtime.evaluation_checkpoint.endswith(
                "checkpoint_00001731.msgpack"
            )
        )
        self.assertNotEqual(
            self.config.runtime.evaluation_output_dir, self.config.runtime.output_dir
        )
        self.assertAlmostEqual(self.config.reward.aesthetic_absolute_weight, 0.8)
        self.assertAlmostEqual(self.config.reward.aesthetic_delta_weight, 0.2)
        self.assertAlmostEqual(
            self.config.shadow_stop.improvement_epsilon, 0.8186361694335936
        )
        self.assertEqual(self.config.shadow_stop.stagnation_patience_steps, 15)
        self.assertEqual(self.config.shadow_stop.verification_frames, 3)
        self.assertAlmostEqual(self.config.shadow_stop.return_deg_per_second, 2.0)
        for startup_lo, policy_lo, policy_hi, startup_hi in zip(
            self.config.robot.startup_joint_low_deg,
            self.config.robot.joint_low_deg,
            self.config.robot.joint_high_deg,
            self.config.robot.startup_joint_high_deg,
        ):
            self.assertLessEqual(startup_lo, policy_lo)
            self.assertLessEqual(policy_hi, startup_hi)

    def test_shadow_stop_is_passive_and_uses_target_presence_only(self):
        monitor = ShadowStopMonitor(self.config.shadow_stop)
        decision = None
        for step in range(18):
            # Edge clearance is intentionally not an input to the monitor.
            decision = monitor.observe(
                step=step,
                score=41.0,
                target_present=True,
                joints_deg=[float(step), 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        assert decision is not None
        self.assertTrue(decision.would_stop)
        self.assertEqual(decision.step, 17)
        self.assertEqual(monitor.trigger_step, 17)
        self.assertFalse(monitor.summary()["edge_clear_required"])
        self.assertFalse(monitor.summary()["return_executed"])

    def test_shadow_stop_v2_selects_exact_best_frame_and_joint_pose(self):
        config = replace(
            self.config.shadow_stop,
            median_window_frames=1,
            minimum_search_steps=0,
            stagnation_patience_steps=1,
            confirmation_steps=4,
        )
        monitor = ShadowStopMonitor(config)
        monitor.observe(
            step=0,
            score=41.0,
            target_present=True,
            joints_deg=[0.0] * 6,
        )
        monitor.observe(
            step=1,
            score=43.0,
            target_present=True,
            joints_deg=[1.0] * 6,
        )
        monitor.observe(
            step=2,
            score=41.0,
            target_present=True,
            joints_deg=[2.0] * 6,
        )
        monitor.observe(
            step=3,
            score=41.0,
            target_present=True,
            joints_deg=[3.0] * 6,
        )
        first_candidate = monitor.observe(
            step=4,
            score=41.0,
            target_present=True,
            joints_deg=[4.0] * 6,
        )
        decision = monitor.observe(
            step=5,
            score=41.0,
            target_present=True,
            joints_deg=[5.0] * 6,
        )
        self.assertTrue(first_candidate.candidate)
        self.assertTrue(decision.would_stop)
        self.assertEqual(decision.best_pose_step, 1)
        self.assertEqual(decision.best_pose_raw_score, 43.0)
        self.assertEqual(decision.best_pose_joint_deg, (1.0,) * 6)
        summary = monitor.summary()
        self.assertTrue(summary["would_return_to_best"])
        self.assertEqual(summary["return_target_step"], 1)
        self.assertEqual(summary["return_target_joint_deg"], (1.0,) * 6)

    def test_shadow_stop_missing_target_and_intervention_reset_confirmation(self):
        config = replace(
            self.config.shadow_stop,
            median_window_frames=1,
            minimum_search_steps=0,
            stagnation_patience_steps=1,
            confirmation_steps=2,
        )
        monitor = ShadowStopMonitor(config)
        monitor.observe(step=0, score=41.0, target_present=True)
        first = monitor.observe(step=1, score=41.0, target_present=True)
        self.assertEqual(first.confirmation_count, 1)
        missing = monitor.observe(step=2, score=41.0, target_present=False)
        self.assertFalse(missing.candidate)
        self.assertEqual(missing.confirmation_count, 0)
        reset = monitor.observe(
            step=3,
            score=41.0,
            target_present=True,
            intervention=True,
        )
        self.assertEqual(reset.reason, "intervention_reset")
        self.assertEqual(reset.confirmation_count, 0)
        self.assertEqual(reset.segment_start_step, 3)
        self.assertEqual(monitor.reset_count, 1)

    def test_stop_modes_require_frozen_evaluation_and_are_exclusive(self):
        with self.assertRaisesRegex(ValueError, "requires frozen --eval-only mode"):
            MethodAActor(
                self.config,
                arm=True,
                evaluation_only=False,
                shadow_stop=True,
                image_encoder_type="tiny",
            )
        with self.assertRaisesRegex(ValueError, "requires frozen --eval-only mode"):
            MethodAActor(
                self.config,
                arm=True,
                evaluation_only=False,
                active_stop=True,
                image_encoder_type="tiny",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MethodAActor(
                self.config,
                arm=True,
                evaluation_only=True,
                shadow_stop=True,
                active_stop=True,
                image_encoder_type="tiny",
            )

    def test_active_stop_verification_requires_all_targets_and_quality(self):
        confirmed = assess_stop_verification(
            scores=[40.5, 41.0, 39.8],
            target_present=[True, True, True],
            expected_frames=3,
            quality_score_threshold=40.0,
        )
        self.assertTrue(confirmed.confirmed)
        self.assertEqual(confirmed.median_score, 40.5)
        missing = assess_stop_verification(
            scores=[42.0, 42.0, 42.0],
            target_present=[True, False, True],
            expected_frames=3,
            quality_score_threshold=40.0,
        )
        self.assertFalse(missing.confirmed)
        self.assertEqual(missing.reason, "target_missing")
        low = assess_stop_verification(
            scores=[39.0, 39.5, 40.5],
            target_present=[True, True, True],
            expected_frames=3,
            quality_score_threshold=40.0,
        )
        self.assertFalse(low.confirmed)
        self.assertEqual(low.reason, "below_quality_threshold")

    def test_active_stop_executes_guarded_return_and_three_frame_verification(self):
        class Snapshot:
            def __init__(self, positions):
                self.positions_rad = np.asarray(positions, dtype=np.float64)

        class FakeRobot:
            def __init__(self):
                self.positions = np.deg2rad([-4.0, 16.0, 0.0, 0.0, 5.0, 0.0])
                self.move_calls = []
                self.holds = 0

            def snapshot(self):
                return Snapshot(self.positions.copy())

            def move_interpolated(self, target, **kwargs):
                self.move_calls.append((np.asarray(target).copy(), kwargs))
                self.positions = np.asarray(target).copy()
                return self.positions.copy()

            def hold(self):
                self.holds += 1

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                shadow_stop=replace(
                    self.config.shadow_stop,
                    verification_interval_seconds=0.0,
                ),
                runtime=replace(
                    self.config.runtime,
                    evaluation_output_dir=directory,
                ),
            )
            actor = MethodAActor(
                config,
                arm=True,
                evaluation_only=True,
                active_stop=True,
                image_encoder_type="tiny",
            )
            actor.robot = FakeRobot()
            observations = iter(
                [fake_capture(40.5), fake_capture(41.0), fake_capture(42.0)]
            )
            actor._capture = lambda: next(observations)
            target = [-7.0, 23.0, -1.0, 0.0, 0.0, 0.0]
            result = actor._execute_active_stop(
                episode_number=1,
                trigger_step=25,
                return_target_step=8,
                return_target_raw_score=42.5,
                return_target_joint_deg=target,
                return_target_image="best.jpg",
            )
            self.assertTrue(result.verification.confirmed)
            self.assertEqual(result.verification.median_score, 41.0)
            self.assertEqual(len(result.verification_images), 3)
            self.assertEqual(len(actor.robot.move_calls), 1)
            self.assertEqual(actor.robot.move_calls[0][1]["deg_per_second"], 2.0)
            self.assertGreaterEqual(actor.robot.holds, 2)
            np.testing.assert_allclose(result.reached_joint_deg, target)

    def test_active_stop_rejects_recorded_pose_outside_policy_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(
                    self.config.runtime,
                    evaluation_output_dir=directory,
                ),
            )
            actor = MethodAActor(
                config,
                arm=True,
                evaluation_only=True,
                active_stop=True,
                image_encoder_type="tiny",
            )
            with self.assertRaisesRegex(RobotSafetyError, "rejected return pose"):
                actor._validated_active_stop_target(
                    [-7.0, 23.0, 5.0, 0.0, 0.0, 0.0]
                )

    def test_evaluation_actor_is_deterministic_and_uses_isolated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation_dir = Path(directory) / "evaluation"
            config = replace(
                self.config,
                runtime=replace(
                    self.config.runtime,
                    output_dir=str(Path(directory) / "training"),
                    evaluation_output_dir=str(evaluation_dir),
                ),
            )
            actor = MethodAActor(
                config,
                arm=True,
                deterministic=False,
                evaluation_only=True,
                image_encoder_type="tiny",
            )
            self.assertTrue(actor.deterministic)
            self.assertTrue(actor.evaluation_only)
            self.assertEqual(actor.logger.root.parent, evaluation_dir.resolve())

    def test_j3_positive_motion_is_clipped_at_real_zero_degree_limit(self):
        class Snapshot:
            def __init__(self, positions):
                self.positions_rad = positions

        class FakeRobot:
            def __init__(self):
                self.positions = np.deg2rad([-7.0, 23.0, -0.1, 0.0, 0.0, 0.0])
                self.last_target = None

            def snapshot(self):
                return Snapshot(self.positions.copy())

            def move_interpolated(self, target, **_kwargs):
                self.last_target = np.asarray(target).copy()
                self.positions = self.last_target.copy()
                return self.positions.copy()

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(self.config.runtime, output_dir=directory),
            )
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            actor.robot = FakeRobot()
            end, actual_delta = actor._motion(
                np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            )
            self.assertAlmostEqual(float(np.rad2deg(end[2])), 0.0, places=5)
            self.assertAlmostEqual(float(actual_delta[2]), 0.1, places=4)
            end, actual_delta = actor._motion(
                np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            )
            self.assertAlmostEqual(float(np.rad2deg(end[2])), 0.0, places=5)
            self.assertAlmostEqual(float(actual_delta[2]), 0.0, places=5)

    def test_manual_key_gate_requires_release_and_suppresses_repeat(self):
        gate = ManualKeyGate(release_seconds=1.0)
        self.assertTrue(gate.update(ord("w"), 0.0))
        self.assertFalse(gate.update(ord("w"), 0.5))
        self.assertFalse(gate.update(None, 1.2))
        self.assertFalse(gate.update(None, 1.51))
        self.assertTrue(gate.update(ord("w"), 1.52))

    def test_disabled_start_pose_guard_does_not_expand_policy_workspace(self):
        robot = PiperRos2.__new__(PiperRos2)
        robot.config = self.config.robot
        # J4/J6 can sag outside their unused configuration entries. They are
        # passed through from feedback and must not block the J1/J2/J3/J5 reset.
        displaced = np.deg2rad([-8.532, -2.988, -5.980, 13.505, 18.079, -10.334])
        np.testing.assert_allclose(robot._validate_target(displaced), displaced)
        with self.assertRaises(RobotSafetyError):
            robot._validate_target(np.deg2rad([-8.532, -2.988, -5.980, 13.505, 36.0, -10.334]))

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(self.config.runtime, output_dir=directory),
            )
            # Construction is hardware-free; devices are opened only by run().
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            random_target = np.rad2deg(actor._random_target())
            # J4/J6 are fixed at configured zero to level the wrist camera.
            self.assertAlmostEqual(random_target[3], 0.0, places=2)
            self.assertAlmostEqual(random_target[5], 0.0, places=2)
            for index in config.robot.controlled_indices:
                self.assertGreaterEqual(random_target[index], config.robot.joint_low_deg[index])
                self.assertLessEqual(random_target[index], config.robot.joint_high_deg[index])

    def test_target_box_keeps_geometry_but_rejects_edge(self):
        detection = {
            "target_valid": False,
            "width": 640,
            "height": 480,
            "best": {"xyxy": [0, 50, 200, 300], "score": 0.8},
        }
        state = target_state_from_detection(detection, edge_margin_ratio=0.05)
        self.assertEqual(state.shape, (7,))
        self.assertEqual(state[-1], 0.0)
        self.assertGreater(state[2], 0.0)

    def test_observation_builder(self):
        built = ObservationBuilder(self.config, FakeVision()).build(
            np.zeros((480, 640, 3), np.uint8), np.zeros(6)
        )
        self.assertEqual(built.observation["wrist_view"].shape, (128, 128, 3))
        self.assertEqual(built.observation["aesthetic_feature"].shape, (3584,))
        self.assertTrue(bool(built.observation["target_state"][-1]))

    def test_reward_is_absolute_dominant_and_terminal_transition_validates(self):
        current = fake_capture(30.0)
        future = fake_capture(32.5)
        transition, reward = make_transition(
            self.config,
            current,
            future,
            action=np.zeros(4, np.float32),
            previous_action=np.zeros(4, np.float32),
            actual_delta_deg=np.zeros(4, np.float32),
            done=True,
        )
        self.assertAlmostEqual(reward.aesthetic_absolute, 0.0625)
        self.assertAlmostEqual(reward.aesthetic_delta, 0.5)
        self.assertAlmostEqual(reward.aesthetic, 0.15)
        self.assertEqual(float(transition["masks"]), 0.0)
        validate_transition(transition)

    def test_absolute_score_prevents_low_score_loitering(self):
        _, low = make_transition(
            self.config,
            fake_capture(20.0),
            fake_capture(24.0),
            action=np.zeros(4, np.float32),
            previous_action=np.zeros(4, np.float32),
            actual_delta_deg=np.zeros(4, np.float32),
            done=False,
        )
        _, high = make_transition(
            self.config,
            fake_capture(44.0),
            fake_capture(40.0),
            action=np.zeros(4, np.float32),
            previous_action=np.zeros(4, np.float32),
            actual_delta_deg=np.zeros(4, np.float32),
            done=False,
        )
        self.assertLess(low.total, 0.0)
        self.assertGreater(high.total, 0.0)

    def test_edge_box_is_present_and_penalized_but_missing_box_is_distinct(self):
        edge = fake_capture(30.0, valid=False, present=True, margin=0.025)
        _, edge_reward = make_transition(
            self.config,
            fake_capture(30.0),
            edge,
            action=np.zeros(4, np.float32),
            previous_action=np.zeros(4, np.float32),
            actual_delta_deg=np.zeros(4, np.float32),
            done=False,
        )
        self.assertTrue(edge_reward.target_present)
        self.assertFalse(edge_reward.target_edge_clear)
        self.assertAlmostEqual(edge_reward.target, -0.25, places=5)

        _, missing_reward = make_transition(
            self.config,
            fake_capture(30.0),
            fake_capture(30.0, valid=False, present=False),
            action=np.zeros(4, np.float32),
            previous_action=np.zeros(4, np.float32),
            actual_delta_deg=np.zeros(4, np.float32),
            done=True,
        )
        self.assertFalse(missing_reward.target_present)
        self.assertEqual(missing_reward.target, self.config.reward.missing_target_penalty)
        self.assertAlmostEqual(
            self.config.reward.target_weight * missing_reward.target, -1.0
        )

    def test_replay_mixing_preserves_batch_size(self):
        transition, _ = make_transition(
            self.config,
            fake_capture(),
            fake_capture(31.0),
            action=np.zeros(4, np.float32),
            previous_action=np.zeros(4, np.float32),
            actual_delta_deg=np.zeros(4, np.float32),
            done=False,
        )
        online, demos, interventions = (ReplayStore(20, i) for i in range(3))
        for _ in range(8):
            online.insert(transition)
            demos.insert(transition)
            interventions.insert(transition)
        self.assertEqual(mixed_batch(online, demos, 8, 0.5)["actions"].shape, (8, 4))
        self.assertEqual(
            mixed_batch_with_interventions(online, demos, interventions, 8, 0.5)["actions"].shape,
            (8, 4),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = online.save(Path(directory) / "replay.pkl")
            loaded = ReplayStore(20)
            self.assertEqual(loaded.load_file(path), 8)

    def test_disable_requires_countdown_and_second_confirmation(self):
        class FakeRobot:
            def __init__(self):
                self.enable_calls = []
                self.enabled_by_us = True

            def enable(self, value):
                self.enable_calls.append(value)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(
                    self.config.runtime,
                    output_dir=directory,
                    disable_countdown_seconds=2,
                ),
            )
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            actor.robot = FakeRobot()
            actor._set_state(ActorState.STOPPED_HOLD, "test")
            actor._dispatch("disable")
            self.assertEqual(actor.robot.enable_calls, [])
            actor._disable_ready_at = time.monotonic() - 0.1
            actor._dispatch("disable")
            self.assertEqual(actor.robot.enable_calls, [False])
            self.assertEqual(actor.state(), ActorState.DISABLED)

    def test_stop_holds_and_never_disables(self):
        class FakeRobot:
            def __init__(self):
                self.holds = 0
                self.enabled_by_us = True

            def hold(self):
                self.holds += 1

        class FakePolicy:
            def __init__(self):
                self.reasons = []

            def request_stop(self, reason):
                self.reasons.append(reason)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(self.config.runtime, output_dir=directory),
            )
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            actor.robot = FakeRobot()
            actor.policy = FakePolicy()
            actor._set_state(ActorState.RUNNING, "test")
            actor._dispatch("stop")
            self.assertEqual(actor.robot.holds, 1)
            self.assertEqual(actor.policy.reasons, ["operator_stop"])
            self.assertTrue(actor.stop_event.is_set())

    def test_episode_start_requires_box_but_not_edge_clearance(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(self.config.runtime, output_dir=directory),
            )
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            actor._set_state(ActorState.WAIT_CONFIRM, "test")
            actor.ui_snapshot.target_present = False
            actor._dispatch("space")
            self.assertEqual(actor.state(), ActorState.WAIT_CONFIRM)
            actor.ui_snapshot.target_present = True
            actor.ui_snapshot.target_valid = False
            actor._dispatch("space")
            self.assertEqual(actor.state(), ActorState.RUNNING)

    def test_wait_confirm_can_request_another_random_start(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(self.config.runtime, output_dir=directory),
            )
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            actor._set_state(ActorState.WAIT_CONFIRM, "target invalid")
            actor._dispatch("end_episode")
            self.assertEqual(actor.state(), ActorState.RESETTING)

    def test_pre_episode_positioning_and_runtime_home(self):
        class FakeRobot:
            def snapshot(self):
                class Snapshot:
                    positions_rad = np.deg2rad([-10.0, 24.0, -1.0, 0.0, 1.0, 3.0])

                return Snapshot()

        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                self.config,
                runtime=replace(self.config.runtime, output_dir=directory),
            )
            actor = MethodAActor(config, arm=True, image_encoder_type="tiny")
            actor.robot = FakeRobot()
            actor._set_state(ActorState.WAIT_CONFIRM, "target invalid")
            actor._dispatch("intervention")
            self.assertEqual(actor.state(), ActorState.POSITIONING)
            actor._dispatch("manual_j2_pos")
            actor._dispatch("manual_j2_pos")
            self.assertEqual(
                actor.manual_actions.get_nowait().tolist(), [0.0, 1.0, 0.0, 0.0]
            )
            self.assertTrue(actor.manual_motion_busy.is_set())
            self.assertTrue(actor.manual_actions.empty())
            actor._dispatch("resume")
            self.assertEqual(actor.state(), ActorState.POSITIONING)
            actor.manual_motion_busy.clear()
            actor.ui_snapshot.target_valid = True
            actor._dispatch("set_home")
            np.testing.assert_allclose(
                actor.home_joint_deg[[0, 1, 2, 4]], [-10.0, 24.0, -1.0, 1.0]
            )
            actor._dispatch("resume")
            self.assertEqual(actor.state(), ActorState.WAIT_CONFIRM)


if __name__ == "__main__":
    unittest.main()
