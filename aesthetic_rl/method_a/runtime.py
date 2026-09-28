"""Actor-side observation, policy transport, and transition bookkeeping."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerConfig

from .agent import AestheticSACAgent
from .config import MethodAConfig
from .reward import RewardResult, compute_reward, target_state_from_detection
from .vision_client import RemoteVisionClient, VisionAnalysis


@dataclass(frozen=True)
class CapturedObservation:
    observation: dict[str, np.ndarray]
    rgb: np.ndarray
    score: float
    detection: dict[str, Any]
    analyze_seconds: float
    detection_seconds: float


class ObservationBuilder:
    def __init__(self, config: MethodAConfig, vision: RemoteVisionClient) -> None:
        self.config = config
        self.vision = vision

    def build(self, rgb: np.ndarray, joint_positions_rad: np.ndarray) -> CapturedObservation:
        joints = np.asarray(joint_positions_rad, dtype=np.float32).reshape(-1)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("robot state must be six finite joint angles")
        analysis: VisionAnalysis = self.vision.analyze(rgb)
        size = self.config.vision.image_size
        resized = cv2.resize(np.asarray(rgb, dtype=np.uint8), (size, size), cv2.INTER_AREA)
        target = target_state_from_detection(
            analysis.detection,
            edge_margin_ratio=self.config.vision.edge_margin_ratio,
        )
        observation = {
            "wrist_view": resized,
            "robot_state": joints,
            "target_state": target,
            "aesthetic_feature": analysis.feature.astype(np.float32, copy=False),
        }
        return CapturedObservation(
            observation=observation,
            rgb=np.asarray(rgb, dtype=np.uint8),
            score=analysis.score,
            detection=analysis.detection,
            analyze_seconds=analysis.analyze_seconds,
            detection_seconds=analysis.detection_seconds,
        )


class PolicyLink:
    """Bidirectional Agentlace link with synchronized policy replacement."""

    def __init__(
        self,
        config: MethodAConfig,
        sample_observation: dict[str, np.ndarray],
        *,
        image_encoder_type: str = "resnet-pretrained",
        evaluation_only: bool = False,
    ) -> None:
        self.config = config
        self.evaluation_only = bool(evaluation_only)
        self.online_queue = QueuedDataStore(config.training.replay_capacity)
        self.intervention_queue = QueuedDataStore(config.training.replay_capacity)
        workspace = Path(__file__).resolve().parents[2]
        resnet_path = workspace / "hil-serl/examples/experiments/resnet10_params.pkl"
        self._agent = AestheticSACAgent.create(
            config.training.seed,
            sample_observation,
            image_encoder_type=image_encoder_type,
            aesthetic_feature_dim=config.vision.aesthetic_feature_dim,
            action_dim=4,
            learning_rate=config.training.learning_rate,
            temperature_learning_rate=config.training.temperature_learning_rate,
            initial_temperature=config.training.initial_temperature,
            discount=config.training.discount,
            tau=config.training.tau,
            pretrained_resnet_path=(
                resnet_path if image_encoder_type == "resnet-pretrained" else None
            ),
        )
        self._lock = threading.RLock()
        self._client_lock = threading.RLock()
        self._network_ready = threading.Event()
        self._network_error: Exception | None = None
        trainer_config = TrainerConfig(
            port_number=config.runtime.trainer_port,
            broadcast_port=config.runtime.broadcast_port,
            request_types=["send-stats", "stop-training", "get-network"],
        )
        self.client = TrainerClient(
            "method-a-actor",
            config.runtime.learner_host,
            trainer_config,
            data_stores={
                "actor_env": self.online_queue,
                "actor_env_intvn": self.intervention_queue,
            },
            wait_for_server=True,
            timeout_ms=5000,
        )
        self.client.recv_network_callback(self._receive_network)
        initial = self.client.request("get-network", {})
        if initial:
            self._receive_network(initial)

    def _receive_network(self, payload: dict[str, Any]) -> None:
        try:
            server_mode = str(payload.get("server_mode", "training"))
            if self.evaluation_only and server_mode != "evaluation":
                raise RuntimeError(
                    "evaluation actor refused a non-evaluation learner; start the frozen "
                    "evaluation learner"
                )
            if not self.evaluation_only and server_mode == "evaluation":
                raise RuntimeError("training actor refused a frozen evaluation learner")
            if self.evaluation_only:
                expected = Path(self.config.runtime.evaluation_checkpoint).expanduser().resolve()
                received_text = payload.get("checkpoint_path")
                if not received_text:
                    raise RuntimeError("evaluation learner did not identify its checkpoint")
                received = Path(str(received_text)).expanduser().resolve()
                if received != expected:
                    raise RuntimeError(
                        f"evaluation checkpoint mismatch: expected {expected}, received {received}"
                    )
            with self._lock:
                self._agent = self._agent.with_policy_payload(payload)
            self._network_ready.set()
        except Exception as exc:
            self._network_error = exc
            self._network_ready.set()
            print(f"[ACTOR] rejected network payload: {exc}", flush=True)

    def wait_ready(self, timeout_seconds: float = 120.0) -> None:
        if not self._network_ready.wait(timeout_seconds):
            raise RuntimeError("learner connected but no policy payload arrived")
        if self._network_error is not None:
            raise RuntimeError(f"learner/actor mode check failed: {self._network_error}")

    @property
    def update_step(self) -> int:
        with self._lock:
            return int(np.asarray(self._agent.update_step))

    def action(self, observation: dict[str, np.ndarray], *, deterministic: bool = False) -> np.ndarray:
        with self._lock:
            agent = self._agent
        seed = jax.random.PRNGKey(time.monotonic_ns() & 0xFFFFFFFF)
        values = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[None], observation)
        action = agent.sample_actions(values, seed=seed, argmax=deterministic)
        result = np.asarray(jax.device_get(action))[0].astype(np.float32)
        if result.shape != (4,) or not np.isfinite(result).all():
            raise RuntimeError(f"policy returned invalid action {result}")
        return np.clip(result, -1.0, 1.0)

    def insert(self, transition: dict[str, Any], *, intervention: bool) -> None:
        if self.evaluation_only:
            return
        self.online_queue.insert(transition)
        if intervention:
            self.intervention_queue.insert(transition)
        # The robot must not continue collecting with an unreachable learner.
        # Queues retain the transition so a later flush can still recover it.
        with self._client_lock:
            sent = self.client.update()
        if not sent:
            raise RuntimeError("learner connection lost while sending transition")

    def flush(self) -> bool:
        if self.evaluation_only:
            return True
        with self._client_lock:
            return bool(self.client.update())

    def send_stats(self, payload: dict[str, Any]) -> None:
        if self.evaluation_only:
            return
        with self._client_lock:
            self.client.request("send-stats", payload)

    def request_stop(self, reason: str) -> None:
        with self._client_lock:
            self.client.request("stop-training", {"reason": reason})

    def close(self) -> None:
        try:
            self.flush()
        finally:
            with self._client_lock:
                self.client.stop()


def make_transition(
    config: MethodAConfig,
    current: CapturedObservation,
    next_observation: CapturedObservation,
    *,
    action: np.ndarray,
    previous_action: np.ndarray,
    actual_delta_deg: np.ndarray,
    done: bool,
) -> tuple[dict[str, Any], RewardResult]:
    reward = compute_reward(
        score=current.score,
        next_score=next_observation.score,
        next_target_state=next_observation.observation["target_state"],
        actual_delta_deg=actual_delta_deg,
        max_step_deg=np.asarray(config.robot.max_step_deg),
        action=action,
        previous_action=previous_action,
        config=config.reward,
        edge_margin_ratio=config.vision.edge_margin_ratio,
    )
    transition = {
        "observations": current.observation,
        "actions": np.asarray(action, dtype=np.float32),
        "next_observations": next_observation.observation,
        "rewards": np.float32(reward.total),
        "masks": np.float32(0.0 if done else 1.0),
        "dones": np.bool_(done),
    }
    return transition, reward


class RuntimeLogger:
    def __init__(self, output_dir: str | Path) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.root = Path(output_dir).expanduser().resolve() / stamp
        self.root.mkdir(parents=True, exist_ok=False)
        self.images = self.root / "images"
        self.images.mkdir()
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.Lock()

    def event(self, kind: str, **payload: Any) -> None:
        record = {"time": time.time(), "kind": kind, **payload}
        with self._lock, self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def save_image(self, rgb: np.ndarray, name: str) -> Path:
        path = self.images / name
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return path
