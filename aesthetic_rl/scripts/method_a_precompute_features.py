#!/usr/bin/env python3
"""Resumably precompute score, hidden feature, and target box per demo frame."""

import argparse
import json
import time

import cv2

from aesthetic_rl.method_a.config import load_config
from aesthetic_rl.method_a.demo_data import FeatureCache, iter_unique_frames
from aesthetic_rl.method_a.reward import target_state_from_detection
from aesthetic_rl.method_a.vision_client import RemoteVisionClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument("--demo-dir", default="aesthetic_rl/data/demo/demo_data")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    client = RemoteVisionClient(
        config.vision.analyze_url,
        config.vision.grounding_url,
        feature_dim=config.vision.aesthetic_feature_dim,
        timeout_seconds=config.vision.request_timeout_seconds,
    )
    print(json.dumps(client.health(), ensure_ascii=False, indent=2), flush=True)
    cache = FeatureCache(config.runtime.feature_cache_dir)
    frames = list(iter_unique_frames(args.demo_dir))
    completed = skipped = 0
    started = time.monotonic()
    for index, frame in enumerate(frames, 1):
        if args.limit and completed >= args.limit:
            break
        if cache.contains(frame["key"]):
            skipped += 1
            continue
        rgb = frame["rgb"]
        source = "embedded_128"
        if frame["high_res_path"] is not None:
            bgr = cv2.imread(str(frame["high_res_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"cannot read {frame['high_res_path']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            source = str(frame["high_res_path"])
        analysis = client.analyze(rgb)
        target = target_state_from_detection(
            analysis.detection,
            edge_margin_ratio=config.vision.edge_margin_ratio,
        )
        cache.save(
            frame["key"],
            score=analysis.score,
            feature=analysis.feature,
            target_state=target,
            detection=analysis.detection,
            metadata={
                "trajectory": frame["trajectory"],
                "step": frame["step"],
                "kind": frame["kind"],
                "source": source,
                "analyze_roundtrip_seconds": analysis.analyze_seconds,
                "detection_roundtrip_seconds": analysis.detection_seconds,
            },
        )
        completed += 1
        print(
            f"[{index}/{len(frames)}] score={analysis.score:.3f} "
            f"valid={bool(target[-1])} cached={frame['key'][:10]}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "unique_frames": len(frames),
                "computed": completed,
                "already_cached": skipped,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
