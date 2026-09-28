"""Read-only demo discovery plus prepared method-A dataset construction."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .config import MethodAConfig
from .reward import TARGET_STATE_DIM, compute_reward, target_state_from_detection


def policy_image_hash(rgb: np.ndarray) -> str:
    image = np.ascontiguousarray(rgb, dtype=np.uint8)
    return hashlib.sha256(image.tobytes()).hexdigest()


def load_pickle(path: str | Path):
    """Load a trusted project pickle.

    Pickle can execute code. This helper is intentionally limited to explicit
    local paths and must never be used on an untrusted download.
    """
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        return pickle.load(handle)


def discover_demo_pairs(demo_dir: str | Path) -> list[tuple[Path, Path]]:
    root = Path(demo_dir).expanduser().resolve()
    pairs: list[tuple[Path, Path]] = []
    for raw_path in sorted(root.glob("piper_keyboard_demo_*.raw.pkl")):
        scored_path = raw_path.with_name(raw_path.name.replace(".raw.pkl", ".pkl"))
        if not scored_path.exists():
            raise FileNotFoundError(f"missing scored pair for {raw_path.name}")
        pairs.append((raw_path, scored_path))
    if not pairs:
        raise FileNotFoundError(f"no paired demos found in {root}")
    return pairs


def resolve_rgb_path(recorded_path: str, demo_dir: str | Path) -> Path | None:
    """Map a teammate-machine absolute path to the copied local rgb tree."""
    recorded = Path(recorded_path)
    if recorded.exists():
        return recorded.resolve()
    root = Path(demo_dir).expanduser().resolve()
    candidate = root / "rgb" / recorded.parent.name / recorded.name
    return candidate if candidate.exists() else None


def iter_unique_frames(demo_dir: str | Path):
    """Yield each unique policy frame and its best available source image.

    The recorded high-resolution JPEG corresponds to next_observations. The
    first observation of each trajectory exists only as the embedded 128px
    array; later observation frames reuse the preceding high-resolution JPEG.
    """
    seen: set[str] = set()
    for raw_path, _ in discover_demo_pairs(demo_dir):
        transitions = load_pickle(raw_path)
        previous_high_res: Path | None = None
        for index, transition in enumerate(transitions):
            obs_rgb = np.asarray(transition["observations"]["wrist_view"], dtype=np.uint8)
            obs_hash = policy_image_hash(obs_rgb)
            if obs_hash not in seen:
                seen.add(obs_hash)
                yield {
                    "key": obs_hash,
                    "rgb": obs_rgb,
                    "high_res_path": previous_high_res,
                    "trajectory": raw_path.name,
                    "step": index,
                    "kind": "observation",
                }
            next_rgb = np.asarray(
                transition["next_observations"]["wrist_view"], dtype=np.uint8
            )
            next_hash = policy_image_hash(next_rgb)
            image_path = resolve_rgb_path(
                transition.get("infos", {}).get("rgb_image_path", ""), demo_dir
            )
            if next_hash not in seen:
                seen.add(next_hash)
                yield {
                    "key": next_hash,
                    "rgb": next_rgb,
                    "high_res_path": image_path,
                    "trajectory": raw_path.name,
                    "step": index,
                    "kind": "next_observation",
                }
            previous_high_res = image_path


class FeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    def path_for(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError(f"invalid frame cache key: {key!r}")
        return self.root / f"{key}.npz"

    def contains(self, key: str) -> bool:
        return self.path_for(key).is_file()

    def load(self, key: str) -> dict[str, Any]:
        path = self.path_for(key)
        with np.load(path, allow_pickle=False) as data:
            result = {
                "score": float(data["score"]),
                "feature": np.asarray(data["feature"], dtype=np.float32),
                "target_state": np.asarray(data["target_state"], dtype=np.float32),
                "detection": json.loads(str(data["detection_json"].item())),
                "metadata": json.loads(str(data["metadata_json"].item())),
            }
        return result

    def save(
        self,
        key: str,
        *,
        score: float,
        feature: np.ndarray,
        target_state: np.ndarray,
        detection: dict,
        metadata: dict,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        final_path = self.path_for(key)
        feature = np.asarray(feature, dtype=np.float32)
        target_state = np.asarray(target_state, dtype=np.float32)
        if target_state.shape != (TARGET_STATE_DIM,):
            raise ValueError(f"target_state has shape {target_state.shape}")
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".npz", dir=self.root
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary,
                score=np.asarray(score, dtype=np.float32),
                feature=feature,
                target_state=target_state,
                detection_json=np.asarray(json.dumps(detection, ensure_ascii=False)),
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            os.replace(temporary, final_path)
        finally:
            temporary.unlink(missing_ok=True)
        return final_path


def audit_demos(demo_dir: str | Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "trajectories": 0,
        "transitions": 0,
        "raw_terminal_count": 0,
        "scored_terminal_count": 0,
        "j4_action_transitions": 0,
        "j6_action_transitions": 0,
        "nonfinite_transitions": 0,
        "missing_remapped_images": 0,
        "lengths": [],
    }
    scores: list[float] = []
    final_scores: list[float] = []
    for raw_path, scored_path in discover_demo_pairs(demo_dir):
        raw = load_pickle(raw_path)
        scored = load_pickle(scored_path)
        if len(raw) != len(scored):
            raise ValueError(f"raw/scored length mismatch: {raw_path.name}")
        report["trajectories"] += 1
        report["transitions"] += len(raw)
        report["lengths"].append(len(raw))
        report["raw_terminal_count"] += sum(bool(t["dones"]) for t in raw)
        report["scored_terminal_count"] += sum(bool(t["dones"]) for t in scored)
        for raw_t, scored_t in zip(raw, scored):
            action6 = np.asarray(raw_t["infos"]["keyboard_action_6"], dtype=np.float32)
            report["j4_action_transitions"] += int(abs(float(action6[3])) > 1e-6)
            report["j6_action_transitions"] += int(abs(float(action6[5])) > 1e-6)
            values = np.concatenate(
                [
                    np.asarray(raw_t["observations"]["state"]).reshape(-1),
                    np.asarray(raw_t["next_observations"]["state"]).reshape(-1),
                    np.asarray(raw_t["actions"]).reshape(-1),
                ]
            )
            report["nonfinite_transitions"] += int(not np.isfinite(values).all())
            image_path = resolve_rgb_path(
                raw_t["infos"].get("rgb_image_path", ""), demo_dir
            )
            report["missing_remapped_images"] += int(image_path is None)
            value = scored_t.get("infos", {}).get("aesthetic_score")
            if value is not None:
                scores.append(float(value))
        final = scored[-1].get("infos", {}).get("aesthetic_score")
        if final is not None:
            final_scores.append(float(final))
    if scores:
        report["score"] = {
            "min": float(np.min(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "max": float(np.max(scores)),
        }
    if final_scores:
        report["final_score"] = {
            "min": float(np.min(final_scores)),
            "mean": float(np.mean(final_scores)),
            "max": float(np.max(final_scores)),
        }
    return report


def _prepared_observation(
    source: dict,
    cached: dict,
    config: MethodAConfig,
) -> dict[str, np.ndarray]:
    state = np.asarray(source["state"], dtype=np.float32).reshape(-1)[:6]
    image = np.asarray(source["wrist_view"], dtype=np.uint8)
    feature = np.asarray(cached["feature"], dtype=np.float32)
    target = np.asarray(cached["target_state"], dtype=np.float32)
    expected_image = (config.vision.image_size, config.vision.image_size, 3)
    if state.shape != (6,) or image.shape != expected_image:
        raise ValueError(f"bad observation state/image shape: {state.shape}/{image.shape}")
    if feature.shape != (config.vision.aesthetic_feature_dim,):
        raise ValueError(f"bad aesthetic feature shape: {feature.shape}")
    if target.shape != (TARGET_STATE_DIM,):
        raise ValueError(f"bad target state shape: {target.shape}")
    return {
        "wrist_view": image.copy(),
        "robot_state": state.copy(),
        "target_state": target.copy(),
        "aesthetic_feature": feature.copy(),
    }


def prepare_demos(
    demo_dir: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    config: MethodAConfig,
) -> dict[str, Any]:
    """Build a new dataset without changing either source pickle."""
    cache = FeatureCache(cache_dir)
    prepared: list[dict[str, Any]] = []
    excluded_j4 = 0
    per_trajectory: list[int] = []
    for raw_path, _ in discover_demo_pairs(demo_dir):
        raw = load_pickle(raw_path)
        trajectory: list[dict[str, Any]] = []
        previous_action = np.zeros(4, dtype=np.float32)
        for index, transition in enumerate(raw):
            action6 = np.asarray(
                transition["infos"]["keyboard_action_6"], dtype=np.float32
            ).reshape(6)
            action = action6[list(config.robot.controlled_indices)].copy()
            if abs(float(action6[3])) > 1e-6 or abs(float(action6[5])) > 1e-6:
                excluded_j4 += 1
                previous_action = action
                continue
            obs_source = transition["observations"]
            next_source = transition["next_observations"]
            obs_cached = cache.load(policy_image_hash(obs_source["wrist_view"]))
            next_cached = cache.load(policy_image_hash(next_source["wrist_view"]))
            delta6 = np.asarray(
                transition["infos"]["actual_delta_deg"], dtype=np.float32
            ).reshape(6)
            controlled_delta = delta6[list(config.robot.controlled_indices)]
            reward = compute_reward(
                score=obs_cached["score"],
                next_score=next_cached["score"],
                next_target_state=next_cached["target_state"],
                actual_delta_deg=controlled_delta,
                max_step_deg=np.asarray(config.robot.max_step_deg, dtype=np.float32),
                action=action,
                previous_action=previous_action,
                config=config.reward,
                edge_margin_ratio=config.vision.edge_margin_ratio,
            )
            done = bool(transition["dones"])
            trajectory.append(
                {
                    "observations": _prepared_observation(obs_source, obs_cached, config),
                    "actions": action.astype(np.float32),
                    "next_observations": _prepared_observation(
                        next_source, next_cached, config
                    ),
                    "rewards": np.float32(reward.total),
                    "masks": np.float32(0.0 if done else 1.0),
                    "dones": done,
                    "infos": {
                        **reward.as_dict(),
                        "aesthetic_score": float(obs_cached["score"]),
                        "next_aesthetic_score": float(next_cached["score"]),
                        "is_demo": True,
                        "source_trajectory": raw_path.name,
                        "source_step": index,
                    },
                }
            )
            previous_action = action
        if not trajectory:
            raise ValueError(f"all transitions were excluded from {raw_path.name}")
        for transition in trajectory:
            transition["dones"] = False
            transition["masks"] = np.float32(1.0)
        trajectory[-1]["dones"] = True
        trajectory[-1]["masks"] = np.float32(0.0)
        prepared.extend(trajectory)
        per_trajectory.append(len(trajectory))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite prepared dataset; move it first: {destination}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    summary = {
        "source_demo_dir": str(Path(demo_dir).expanduser().resolve()),
        "output_path": str(destination),
        "trajectories": len(per_trajectory),
        "transitions": len(prepared),
        "excluded_unmodelled_j4_j6_transitions": excluded_j4,
        "trajectory_lengths": per_trajectory,
        "configuration": asdict(config),
    }
    summary_path = destination.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
