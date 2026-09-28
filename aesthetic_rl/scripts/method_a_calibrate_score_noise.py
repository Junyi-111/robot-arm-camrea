#!/usr/bin/env python3
"""Measure Method-A score noise without commanding the robot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aesthetic_rl.method_a.camera import D435Camera
from aesthetic_rl.method_a.config import MethodAConfig, load_config
from aesthetic_rl.method_a.reward import target_state_from_detection
from aesthetic_rl.method_a.robot_ros2 import PiperRos2
from aesthetic_rl.method_a.vision_client import RemoteVisionClient, VisionAnalysis


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def rolling_median(values: list[float], window: int = 3) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if window < 1:
        raise ValueError("rolling median window must be positive")
    if data.size < window:
        return np.asarray([], dtype=np.float64)
    return np.asarray(
        [np.median(data[index - window + 1 : index + 1]) for index in range(window - 1, len(data))],
        dtype=np.float64,
    )


def score_statistics(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("score statistics require finite samples")
    differences = np.abs(np.diff(data))
    median = float(np.median(data))
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "median": median,
        "median_absolute_deviation": float(np.median(np.abs(data - median))),
        "min": float(data.min()),
        "max": float(data.max()),
        "range": float(np.ptp(data)),
        "consecutive_abs_diff_mean": float(differences.mean()) if differences.size else 0.0,
        "consecutive_abs_diff_p95": _percentile(differences, 95.0),
        "consecutive_abs_diff_max": float(differences.max()) if differences.size else 0.0,
    }


def _box_geometry(detection: dict[str, Any]) -> list[float] | None:
    best = (detection or {}).get("best")
    if not best:
        return None
    try:
        width = float(detection["width"])
        height = float(detection["height"])
        x1, y1, x2, y2 = (float(value) for value in best["xyxy"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return [
        (x1 + x2) / (2.0 * width),
        (y1 + y2) / (2.0 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def _box_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [record["box_geometry"] for record in records if record["box_geometry"]]
    if not values:
        return {"count": 0}
    data = np.asarray(values, dtype=np.float64)
    labels = ("center_x", "center_y", "width", "height")
    return {
        "count": int(len(data)),
        "mean": {key: float(value) for key, value in zip(labels, data.mean(axis=0))},
        "std": {
            key: float(value)
            for key, value in zip(
                labels,
                data.std(axis=0, ddof=1) if len(data) > 1 else np.zeros(4),
            )
        },
    }


def _record(
    group: str,
    index: int,
    analysis: VisionAnalysis,
    joints_deg: np.ndarray,
    config: MethodAConfig,
    image_path: str,
) -> dict[str, Any]:
    target = target_state_from_detection(
        analysis.detection, edge_margin_ratio=config.vision.edge_margin_ratio
    )
    best = (analysis.detection or {}).get("best")
    return {
        "group": group,
        "index": int(index),
        "host_time": time.time(),
        "score": float(analysis.score),
        "joint_deg": np.asarray(joints_deg, dtype=np.float64).round(6).tolist(),
        "target_present": bool(best),
        "target_edge_clear": bool(target[-1] >= 0.5),
        "box_geometry": _box_geometry(analysis.detection),
        "box_confidence": float(best.get("score", 0.0)) if best else None,
        "artimuse_seconds": float(analysis.analyze_seconds),
        "grounding_seconds": float(analysis.detection_seconds),
        "image": image_path,
    }


def _drift_deg(records: list[dict[str, Any]], baseline_deg: np.ndarray) -> np.ndarray:
    if not records:
        return np.zeros(6, dtype=np.float64)
    joints = np.asarray([record["joint_deg"] for record in records], dtype=np.float64)
    return np.max(np.abs(joints - baseline_deg[None]), axis=0)


def build_summary(
    exact_records: list[dict[str, Any]],
    fresh_records: list[dict[str, Any]],
    baseline_deg: np.ndarray,
    config: MethodAConfig,
    drift_limit_deg: float,
) -> dict[str, Any]:
    exact_scores = [float(record["score"]) for record in exact_records]
    fresh_scores = [float(record["score"]) for record in fresh_records]
    exact_stats = score_statistics(exact_scores)
    fresh_stats = score_statistics(fresh_scores)
    smoothed = rolling_median(fresh_scores, window=3)
    smoothed_stats = score_statistics(smoothed.tolist())
    smoothed_p95 = float(smoothed_stats["consecutive_abs_diff_p95"])
    recommended_epsilon = max(0.1, 1.2 * smoothed_p95)
    all_records = exact_records + fresh_records
    drift = _drift_deg(all_records, np.asarray(baseline_deg, dtype=np.float64))
    controlled = np.asarray(config.robot.controlled_indices, dtype=np.int64)
    target_present_rate = float(
        np.mean([record["target_present"] for record in fresh_records])
    )
    edge_clear_rate = float(
        np.mean([record["target_edge_clear"] for record in fresh_records])
    )
    warnings: list[str] = []
    if float(drift.max()) > drift_limit_deg:
        warnings.append("joint drift exceeded the configured limit")
    if target_present_rate < 1.0:
        warnings.append("Grounding DINO lost the flower in at least one fresh frame")
    if edge_clear_rate < 1.0:
        warnings.append("the flower box touched the configured edge margin")
    if float(exact_stats["std"]) > 0.1:
        warnings.append("identical-image model inference is unexpectedly variable")
    if float(fresh_stats["std"]) > 1.0:
        warnings.append("fresh-frame score noise is too large for a reliable STOP rule")
    exact_variance = float(exact_stats["std"]) ** 2
    fresh_variance = float(fresh_stats["std"]) ** 2
    return {
        "measurement_valid": not warnings,
        "warnings": warnings,
        "exact_image": exact_stats,
        "fresh_frames": fresh_stats,
        "fresh_frames_median3": smoothed_stats,
        "estimated_camera_environment_std": math.sqrt(
            max(0.0, fresh_variance - exact_variance)
        ),
        "recommended": {
            "score_smoothing": "median of latest 3 scores",
            "improvement_epsilon": float(recommended_epsilon),
            "deterioration_threshold": float(2.0 * recommended_epsilon),
            "stagnation_window_steps": 5,
            "stop_confirmation_steps": 3,
            "provisional_good_score": 40.0,
            "formula": "epsilon=max(0.1, 1.2*P95(abs(diff(median3_score))))",
        },
        "target": {
            "present_rate": target_present_rate,
            "edge_clear_rate": edge_clear_rate,
            "box_geometry": _box_statistics(fresh_records),
        },
        "robot": {
            "baseline_joint_deg": np.asarray(baseline_deg).round(6).tolist(),
            "max_abs_drift_deg": drift.round(6).tolist(),
            "max_controlled_drift_deg": float(drift[controlled].max()),
            "drift_limit_deg": float(drift_limit_deg),
        },
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "group",
        "index",
        "host_time",
        "score",
        "target_present",
        "target_edge_clear",
        "box_confidence",
        "artimuse_seconds",
        "grounding_seconds",
        "image",
        "joint_deg",
        "box_geometry",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["joint_deg"] = json.dumps(row["joint_deg"])
            row["box_geometry"] = json.dumps(row["box_geometry"])
            writer.writerow(row)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _joint_deg(robot: PiperRos2) -> np.ndarray:
    return np.rad2deg(robot.snapshot().positions_rad)


def _check_policy_workspace(joints_deg: np.ndarray, config: MethodAConfig) -> None:
    controlled = np.asarray(config.robot.controlled_indices, dtype=np.int64)
    low = np.asarray(config.robot.joint_low_deg)[controlled]
    high = np.asarray(config.robot.joint_high_deg)[controlled]
    values = joints_deg[controlled]
    if np.any((values < low) | (values > high)):
        raise RuntimeError(
            "fixed-pose calibration requires J1/J2/J3/J5 inside policy bounds: "
            f"{values.round(3).tolist()}"
        )


def _check_drift(
    joints_deg: np.ndarray,
    baseline_deg: np.ndarray,
    limit_deg: float,
) -> None:
    drift = np.abs(joints_deg - baseline_deg)
    if float(drift.max()) > limit_deg:
        raise RuntimeError(
            "robot moved during fixed-pose calibration: "
            f"drift={drift.round(4).tolist()} deg; limit={limit_deg:.4f} deg"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only fixed-pose ArtiMuse/Grounding noise calibration"
    )
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    parser.add_argument("--sample-delay", type=float, default=0.6)
    parser.add_argument("--operator-clear-seconds", type=int, default=5)
    parser.add_argument("--joint-drift-limit-deg", type=float, default=0.1)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.samples < 20:
        raise ValueError("use at least 20 samples; 30 is recommended")
    if args.warmup_samples < 0 or args.settle_seconds < 0.0 or args.sample_delay < 0.0:
        raise ValueError("warmup, settling, and delay values cannot be negative")
    if args.joint_drift_limit_deg <= 0.0:
        raise ValueError("joint drift limit must be positive")
    config = load_config(args.config)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path(
            "aesthetic_rl/method_a_artifacts/score_noise_calibration", stamp
        ).resolve()
    )
    output.mkdir(parents=True, exist_ok=False)
    images = output / "fresh_images"
    images.mkdir()
    raw_path = output / "samples.jsonl"
    client = RemoteVisionClient(
        config.vision.analyze_url,
        config.vision.grounding_url,
        feature_dim=config.vision.aesthetic_feature_dim,
        timeout_seconds=config.vision.request_timeout_seconds,
    )
    health = client.health()
    _atomic_json(output / "services.json", health)
    print("[CALIBRATION] READ ONLY: this process cannot command or enable the arm.", flush=True)
    print("[CALIBRATION] Keep the arm enabled/HOLDING; keep people out of the image.", flush=True)
    robot: PiperRos2 | None = None
    camera: D435Camera | None = None
    all_records: list[dict[str, Any]] = []
    exact_records: list[dict[str, Any]] = []
    fresh_records: list[dict[str, Any]] = []
    try:
        robot = PiperRos2(config.robot, allow_writes=False)
        baseline_deg = np.rad2deg(robot.wait_for_feedback().positions_rad)
        _check_policy_workspace(baseline_deg, config)
        camera = D435Camera(
            serial=config.runtime.camera_serial,
            width=config.runtime.camera_width,
            height=config.runtime.camera_height,
            fps=config.runtime.camera_fps,
        )
        print(
            f"[CALIBRATION] camera settling for {args.settle_seconds:.1f}s; "
            f"baseline joints={baseline_deg.round(3).tolist()}",
            flush=True,
        )
        deadline = time.monotonic() + args.settle_seconds
        while time.monotonic() < deadline:
            camera.capture_rgb()
        for remaining in range(args.operator_clear_seconds, 0, -1):
            print(f"[CALIBRATION] leave the camera view: {remaining}", flush=True)
            time.sleep(1.0)
        print(f"[CALIBRATION] {args.warmup_samples} unrecorded vision warmups", flush=True)
        for _ in range(args.warmup_samples):
            client.analyze(camera.capture_rgb())
            _check_drift(_joint_deg(robot), baseline_deg, args.joint_drift_limit_deg)
        reference_rgb = camera.capture_rgb()
        reference_path = output / "identical_reference.png"
        cv2.imwrite(str(reference_path), cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR))
        print(f"[CALIBRATION] identical-image repeats: {args.samples}", flush=True)
        for index in range(args.samples):
            analysis = client.analyze(reference_rgb)
            joints_deg = _joint_deg(robot)
            record = _record(
                "identical_image",
                index,
                analysis,
                joints_deg,
                config,
                str(reference_path),
            )
            _append_jsonl(raw_path, record)
            exact_records.append(record)
            all_records.append(record)
            _check_drift(joints_deg, baseline_deg, args.joint_drift_limit_deg)
            print(
                f"[IDENTICAL] {index + 1:02d}/{args.samples} "
                f"score={analysis.score:.4f}",
                flush=True,
            )
            time.sleep(args.sample_delay)
        print(f"[CALIBRATION] fresh-frame repeats: {args.samples}", flush=True)
        for index in range(args.samples):
            rgb = camera.capture_rgb()
            image_path = images / f"fresh_{index:03d}.jpg"
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            analysis = client.analyze(rgb)
            joints_deg = _joint_deg(robot)
            record = _record(
                "fresh_frame",
                index,
                analysis,
                joints_deg,
                config,
                str(image_path),
            )
            _append_jsonl(raw_path, record)
            fresh_records.append(record)
            all_records.append(record)
            _check_drift(joints_deg, baseline_deg, args.joint_drift_limit_deg)
            print(
                f"[FRESH] {index + 1:02d}/{args.samples} "
                f"score={analysis.score:.4f} present={record['target_present']} "
                f"edge_clear={record['target_edge_clear']}",
                flush=True,
            )
            time.sleep(args.sample_delay)
        summary = build_summary(
            exact_records,
            fresh_records,
            baseline_deg,
            config,
            args.joint_drift_limit_deg,
        )
        summary["services"] = health
        summary["output"] = str(output)
        _write_csv(output / "samples.csv", all_records)
        _atomic_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"[CALIBRATION COMPLETE] {output}", flush=True)
    except Exception as exc:
        _atomic_json(
            output / "FAILED.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "recorded_samples": len(all_records),
                "safety": "No robot command was sent by this process.",
            },
        )
        raise
    finally:
        if camera is not None:
            camera.close()
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
