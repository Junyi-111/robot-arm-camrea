#!/usr/bin/env python3
"""Non-mutating local prerequisite report for Method A."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import jax

from aesthetic_rl.method_a.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument("--require-learner-gpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    checks = {
        "jax_version": jax.__version__,
        "jax_devices": [str(device) for device in jax.devices()],
        "prepared_demo": Path(config.runtime.prepared_demo_path).expanduser().resolve().is_file(),
        "resnet_weights": (root / "hil-serl/examples/experiments/resnet10_params.pkl").is_file(),
        "pyrealsense2": importlib.util.find_spec("pyrealsense2") is not None,
        "rclpy": importlib.util.find_spec("rclpy") is not None,
        "piper_msgs": importlib.util.find_spec("piper_msgs") is not None,
    }
    checks["learner_gpu"] = any(device.platform == "gpu" for device in jax.devices())
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    required = (checks["prepared_demo"], checks["resnet_weights"], checks["pyrealsense2"])
    if not all(required) or (args.require_learner_gpu and not checks["learner_gpu"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
