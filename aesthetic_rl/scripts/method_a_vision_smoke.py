#!/usr/bin/env python3
"""Write a JSON artifact containing score, ArtiMuse hidden, and target box."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from aesthetic_rl.method_a.config import load_config
from aesthetic_rl.method_a.reward import target_state_from_detection
from aesthetic_rl.method_a.vision_client import RemoteVisionClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    image_path = Path(args.image).expanduser().resolve()
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    client = RemoteVisionClient(
        config.vision.analyze_url,
        config.vision.grounding_url,
        feature_dim=config.vision.aesthetic_feature_dim,
        timeout_seconds=config.vision.request_timeout_seconds,
    )
    health = client.health()
    result = client.analyze(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    target = target_state_from_detection(
        result.detection, edge_margin_ratio=config.vision.edge_margin_ratio
    )
    payload = {
        "image": str(image_path),
        "score": result.score,
        "hidden_dim": int(result.feature.shape[0]),
        "multimodal_hidden": result.feature.tolist(),
        "detection": result.detection,
        "target_state": target.tolist(),
        "roundtrip_seconds": {
            "artimuse": result.analyze_seconds,
            "grounding_dino": result.detection_seconds,
        },
        "services": health,
    }
    destination = (
        Path(args.output).expanduser().resolve()
        if args.output
        else Path("aesthetic_rl/method_a_artifacts/vision_smoke") / f"{image_path.stem}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(destination.resolve())


if __name__ == "__main__":
    main()
