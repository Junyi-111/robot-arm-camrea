"""Thread-safe replay stores and 50/50 RLPD-style sampling."""

from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np
from agentlace.data.data_store import DataStoreBase


CORE_KEYS = ("observations", "actions", "next_observations", "rewards", "masks", "dones")


def _stack(values):
    first = values[0]
    if isinstance(first, dict):
        return {key: _stack([value[key] for value in values]) for key in first}
    return np.stack(values)


def validate_transition(transition: dict[str, Any]) -> None:
    missing = set(CORE_KEYS) - set(transition)
    if missing:
        raise ValueError(f"transition missing keys: {sorted(missing)}")
    action = np.asarray(transition["actions"], dtype=np.float32)
    if action.shape != (4,) or not np.isfinite(action).all():
        raise ValueError(f"invalid 4-D action: shape={action.shape}")
    for side in ("observations", "next_observations"):
        obs = transition[side]
        expected = {
            "robot_state": (6,),
            "target_state": (7,),
            "aesthetic_feature": (3584,),
        }
        for key, shape in expected.items():
            value = np.asarray(obs[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {side}.{key}: shape={value.shape}")
        image = np.asarray(obs["wrist_view"])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"invalid {side}.wrist_view: {image.shape}/{image.dtype}")


class ReplayStore(DataStoreBase):
    def __init__(self, capacity: int, seed: int = 0):
        super().__init__(capacity)
        self._data: list[dict[str, Any]] = []
        self._next_index = 0
        self._sequence_id = -1
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        with self._lock:
            return len(self._data)

    def latest_data_id(self):
        with self._lock:
            return self._sequence_id

    def get_latest_data(self, from_id):
        del from_id
        raise NotImplementedError("learner replay stores are receive-only")

    def insert(self, data):
        validate_transition(data)
        item = {key: data[key] for key in CORE_KEYS}
        with self._lock:
            if len(self._data) < self.capacity:
                self._data.append(item)
            else:
                self._data[self._next_index] = item
            self._next_index = (self._next_index + 1) % self.capacity
            self._sequence_id += 1

    def sample(self, batch_size: int):
        with self._lock:
            if not self._data:
                raise RuntimeError("cannot sample an empty replay store")
            indices = self._rng.integers(0, len(self._data), size=int(batch_size))
            samples = [self._data[int(index)] for index in indices]
        return _stack(samples)

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = list(self._data)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
        return destination

    def load_file(self, path: str | Path) -> int:
        source = Path(path).expanduser().resolve()
        with source.open("rb") as handle:
            transitions = pickle.load(handle)
        for transition in transitions:
            self.insert(transition)
        return len(transitions)


def mixed_batch(
    online: ReplayStore,
    demos: ReplayStore,
    batch_size: int,
    demo_fraction: float,
):
    demo_count = int(round(batch_size * demo_fraction))
    demo_count = min(max(demo_count, 1), batch_size - 1)
    online_count = batch_size - demo_count
    online_batch = online.sample(online_count)
    demo_batch = demos.sample(demo_count)
    return _concat(online_batch, demo_batch)


def mixed_batch_with_interventions(
    online: ReplayStore,
    demos: ReplayStore,
    interventions: ReplayStore,
    batch_size: int,
    demo_fraction: float,
):
    """Keep online/demo mixing while reserving half the demo side for corrections."""
    offline_count = int(round(batch_size * demo_fraction))
    offline_count = min(max(offline_count, 2), batch_size - 1)
    online_count = batch_size - offline_count
    intervention_count = max(1, offline_count // 2)
    demo_count = offline_count - intervention_count
    return _concat(
        online.sample(online_count),
        _concat(demos.sample(demo_count), interventions.sample(intervention_count)),
    )


def _concat(first, second):
    if isinstance(first, dict):
        return {key: _concat(first[key], second[key]) for key in first}
    return np.concatenate([first, second], axis=0)
