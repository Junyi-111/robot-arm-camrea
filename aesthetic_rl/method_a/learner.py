"""Local BC warmup and RLPD-style SAC learner served through Agentlace."""

from __future__ import annotations

import json
import signal
import threading
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from agentlace.trainer import TrainerConfig, TrainerServer

from .agent import AestheticSACAgent, load_checkpoint, save_checkpoint
from .config import MethodAConfig
from .demo_data import load_pickle
from .replay import ReplayStore, mixed_batch, mixed_batch_with_interventions


def _to_jax(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


class MethodALearner:
    def __init__(
        self,
        config: MethodAConfig,
        *,
        image_encoder_type: str = "resnet-pretrained",
        resume: bool = False,
        evaluation_only: bool = False,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.image_encoder_type = image_encoder_type
        self.evaluation_only = bool(evaluation_only)
        self.loaded_checkpoint: Path | None = None
        self.stop_event = threading.Event()
        self.online = ReplayStore(config.training.replay_capacity, config.training.seed)
        self.interventions = ReplayStore(
            config.training.replay_capacity, config.training.seed + 1
        )
        self.demos = ReplayStore(
            max(config.training.replay_capacity, 1000), config.training.seed + 2
        )
        demo_path = Path(config.runtime.prepared_demo_path).expanduser().resolve()
        transitions = load_pickle(demo_path)
        for transition in transitions:
            self.demos.insert(transition)
        if not transitions:
            raise ValueError(f"prepared demo is empty: {demo_path}")
        sample_observation = transitions[0]["observations"]
        workspace = Path(__file__).resolve().parents[2]
        resnet_path = workspace / "hil-serl/examples/experiments/resnet10_params.pkl"
        self.agent = AestheticSACAgent.create(
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
            pretrained_resnet_path=resnet_path if image_encoder_type == "resnet-pretrained" else None,
        )
        self.checkpoint_dir = Path(config.runtime.checkpoint_dir).expanduser().resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = (
            config.runtime.evaluation_output_dir
            if self.evaluation_only
            else config.runtime.output_dir
        )
        self.runtime_dir = Path(runtime_path).expanduser().resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.runtime_dir / "learner_metrics.jsonl"
        self.online_replay_path = self.runtime_dir / "online_replay.pkl"
        self.intervention_replay_path = self.runtime_dir / "intervention_replay.pkl"
        self.sac_updates = 0
        if self.evaluation_only:
            selected = Path(
                checkpoint_path or config.runtime.evaluation_checkpoint
            ).expanduser().resolve()
            if not selected.is_file():
                raise FileNotFoundError(f"frozen evaluation checkpoint not found: {selected}")
            self.agent = load_checkpoint(self.agent, selected)
            metadata = self._checkpoint_metadata(selected)
            loaded_step = int(self.agent.update_step)
            try:
                filename_step = int(selected.stem.rsplit("_", 1)[-1])
            except ValueError as exc:
                raise ValueError(
                    f"evaluation checkpoint must use checkpoint_NNNNNNNN naming: {selected}"
                ) from exc
            if loaded_step != filename_step:
                raise ValueError(
                    f"checkpoint filename/weights step mismatch: "
                    f"filename={filename_step}, weights={loaded_step}"
                )
            metadata_step = int(metadata.get("update_step", loaded_step))
            if loaded_step != metadata_step:
                raise ValueError(
                    f"checkpoint/metadata step mismatch: weights={loaded_step}, "
                    f"metadata={metadata_step}"
                )
            self.loaded_checkpoint = selected
            self.sac_updates = int(metadata.get("sac_updates", 0))
            print(
                f"[LEARNER] FROZEN EVALUATION checkpoint={selected} "
                f"update_step={loaded_step}; gradients/replay/checkpoint writes disabled",
                flush=True,
            )
        elif resume:
            latest = self.latest_checkpoint()
            if latest is None:
                raise FileNotFoundError(f"--resume requested but no checkpoint in {self.checkpoint_dir}")
            self.agent = load_checkpoint(self.agent, latest)
            self.loaded_checkpoint = latest
            self.sac_updates = self._checkpoint_metadata(latest).get("sac_updates", 0)
            if self.online_replay_path.exists():
                self.online.load_file(self.online_replay_path)
            if self.intervention_replay_path.exists():
                self.interventions.load_file(self.intervention_replay_path)
            print(f"[LEARNER] resumed {latest} sac_updates={self.sac_updates}", flush=True)
        trainer_config = TrainerConfig(
            port_number=config.runtime.trainer_port,
            broadcast_port=config.runtime.broadcast_port,
            request_types=["send-stats", "stop-training", "get-network"],
        )
        self.server = TrainerServer(trainer_config, request_callback=self._request)
        self.server.register_data_store("actor_env", self.online)
        self.server.register_data_store("actor_env_intvn", self.interventions)

    def _policy_payload(self) -> dict[str, Any]:
        payload = self.agent.policy_payload()
        payload["server_mode"] = "evaluation" if self.evaluation_only else "training"
        if self.evaluation_only:
            assert self.loaded_checkpoint is not None
            payload["checkpoint_path"] = str(self.loaded_checkpoint)
            payload["frozen"] = True
        return payload

    def _request(self, request_type: str, payload: dict) -> dict:
        if request_type == "send-stats":
            if self.evaluation_only:
                return {
                    "accepted": False,
                    "reason": "frozen evaluation does not record training stats",
                }
            self._write_metric({"kind": "actor", **payload})
            return {"accepted": True}
        if request_type == "stop-training":
            self.stop_event.set()
            return {"accepted": True, "holding_enabled": True}
        if request_type == "get-network":
            # A request/reply bootstrap avoids the PUB/SUB slow-joiner race.
            return self._policy_payload()
        raise ValueError(f"unknown learner request: {request_type}")

    def _write_metric(self, payload: dict[str, Any]) -> None:
        if self.evaluation_only:
            raise RuntimeError("frozen evaluation cannot write learner metrics")
        record = {"time": time.time(), **payload}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def latest_checkpoint(self) -> Path | None:
        paths = sorted(self.checkpoint_dir.glob("checkpoint_*.msgpack"))
        return paths[-1] if paths else None

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_suffix(".json")

    def _checkpoint_metadata(self, path: Path) -> dict:
        metadata = self._metadata_path(path)
        return json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}

    def checkpoint(self, label_step: int | None = None) -> Path:
        if self.evaluation_only:
            raise RuntimeError("frozen evaluation cannot write checkpoints")
        step = int(self.agent.update_step) if label_step is None else int(label_step)
        path = self.checkpoint_dir / f"checkpoint_{step:08d}.msgpack"
        save_checkpoint(self.agent, path)
        metadata = {
            "update_step": int(self.agent.update_step),
            "sac_updates": int(self.sac_updates),
            "online_transitions": len(self.online),
            "intervention_transitions": len(self.interventions),
            "demo_transitions": len(self.demos),
            "image_encoder_type": self.image_encoder_type,
            "holding_enabled_on_stop": True,
        }
        self._metadata_path(path).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[LEARNER] checkpoint={path}", flush=True)
        return path

    def save_replays(self) -> None:
        if self.evaluation_only:
            raise RuntimeError("frozen evaluation cannot write replay buffers")
        self.online.save(self.online_replay_path)
        self.interventions.save(self.intervention_replay_path)

    def bc_warmup(self) -> None:
        if self.evaluation_only:
            raise RuntimeError("frozen evaluation cannot perform gradient updates")
        steps = self.config.training.bc_warmup_steps
        if int(self.agent.update_step) >= steps:
            print("[LEARNER] BC warmup already present in checkpoint", flush=True)
            return
        for index in range(int(self.agent.update_step), steps):
            batch = _to_jax(self.demos.sample(self.config.training.batch_size))
            self.agent, metrics = self.agent.bc_update(batch)
            if index == 0 or (index + 1) % 10 == 0:
                values = _scalar_metrics(metrics)
                print(
                    f"[BC] {index + 1}/{steps} loss={values['bc_loss']:.5f} "
                    f"mse={values['bc_mse']:.5f}",
                    flush=True,
                )
                self._write_metric({"kind": "bc", "step": index + 1, **values})
        self.checkpoint()

    def run(self, *, bc_only: bool = False) -> None:
        if self.evaluation_only:
            if bc_only:
                raise ValueError("--bc-only is incompatible with frozen evaluation")
            self.server.start(threaded=True)
            self.server.publish_network(self._policy_payload())
            print(
                f"[LEARNER] frozen policy serving ports {self.config.runtime.trainer_port}/"
                f"{self.config.runtime.broadcast_port}; waiting for actor stop",
                flush=True,
            )
            while not self.stop_event.wait(0.2):
                pass
            print(
                "[LEARNER] frozen evaluation server stopped; no training artifacts written",
                flush=True,
            )
            return
        self.bc_warmup()
        if bc_only:
            print("[LEARNER] BC-only run complete", flush=True)
            return
        self.server.start(threaded=True)
        self.server.publish_network(self._policy_payload())
        print(
            f"[LEARNER] serving ports {self.config.runtime.trainer_port}/"
            f"{self.config.runtime.broadcast_port}; waiting for "
            f"{self.config.training.training_starts} online transitions",
            flush=True,
        )
        while (
            len(self.online) < self.config.training.training_starts
            and not self.stop_event.wait(0.2)
        ):
            pass
        while self.sac_updates < self.config.training.max_updates and not self.stop_event.is_set():
            # At most one gradient update per received online transition keeps
            # the policy synchronized with slow real-robot data collection.
            if self.sac_updates >= len(self.online):
                self.stop_event.wait(0.05)
                continue
            if len(self.interventions) >= 4:
                batch = mixed_batch_with_interventions(
                    self.online,
                    self.demos,
                    self.interventions,
                    self.config.training.batch_size,
                    self.config.training.demo_fraction,
                )
            else:
                batch = mixed_batch(
                    self.online,
                    self.demos,
                    self.config.training.batch_size,
                    self.config.training.demo_fraction,
                )
            self.agent, metrics = self.agent.sac_update(_to_jax(batch))
            self.sac_updates += 1
            values = _scalar_metrics(metrics)
            if self.sac_updates == 1 or self.sac_updates % 10 == 0:
                print(
                    f"[SAC] update={self.sac_updates} online={len(self.online)} "
                    f"critic={values['critic_loss']:.5f} actor={values['actor_loss']:.5f}",
                    flush=True,
                )
                self._write_metric(
                    {"kind": "sac", "update": self.sac_updates, "online": len(self.online), **values}
                )
            if self.sac_updates % self.config.training.publish_period == 0:
                self.server.publish_network(self._policy_payload())
            if self.sac_updates % self.config.training.buffer_save_period == 0:
                self.save_replays()
            if self.sac_updates % self.config.training.checkpoint_period == 0:
                self.checkpoint()
        self.save_replays()
        self.checkpoint()
        self.server.publish_network(self._policy_payload())
        print("[LEARNER] stopped; robot actor must remain holding until operator action", flush=True)

    def close(self) -> None:
        self.stop_event.set()
        self.server.stop()


def install_signal_handlers(learner: MethodALearner) -> None:
    def stop(_signum, _frame):
        learner.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
