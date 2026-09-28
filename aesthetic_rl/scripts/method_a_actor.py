#!/usr/bin/env python3
"""Run Method A preflight or the real-robot actor."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from aesthetic_rl.method_a.actor import MethodAActor, run_preflight
from aesthetic_rl.method_a.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument("--preflight", action="store_true", help="read-only camera/vision check")
    parser.add_argument(
        "--check-robot-feedback", action="store_true", help="also subscribe to ROS feedback"
    )
    parser.add_argument("--arm", action="store_true", help="permit gated physical actor startup")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="frozen deterministic evaluation; disables training replay upload",
    )
    parser.add_argument(
        "--shadow-stop",
        action="store_true",
        help="log hypothetical STOP decisions without ending episodes (requires --eval-only)",
    )
    parser.add_argument(
        "--active-stop",
        action="store_true",
        help=(
            "return slowly to the recorded best pose and end only after fixed-pose "
            "verification (requires --eval-only)"
        ),
    )
    parser.add_argument(
        "--eval-checkpoint",
        help="override runtime.evaluation_checkpoint in frozen --eval-only mode",
    )
    parser.add_argument(
        "--eval-output-dir",
        help="override runtime.evaluation_output_dir in frozen --eval-only mode",
    )
    parser.add_argument("--encoder", choices=("resnet-pretrained", "tiny"), default="resnet-pretrained")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.eval_checkpoint or args.eval_output_dir:
        if not args.eval_only:
            parser.error("evaluation overrides require --eval-only")
        config = replace(
            config,
            runtime=replace(
                config.runtime,
                evaluation_checkpoint=(
                    args.eval_checkpoint or config.runtime.evaluation_checkpoint
                ),
                evaluation_output_dir=(
                    args.eval_output_dir or config.runtime.evaluation_output_dir
                ),
            ),
        ).validate()
    if args.preflight:
        print(json.dumps(run_preflight(config, check_robot_feedback=args.check_robot_feedback), indent=2))
        return
    if not args.arm:
        parser.error("physical operation requires --arm (use --preflight for read-only checks)")
    if args.shadow_stop and args.active_stop:
        parser.error("choose exactly one of --shadow-stop or --active-stop")
    if (args.shadow_stop or args.active_stop) and not args.eval_only:
        parser.error("STOP evaluation requires --eval-only")
    MethodAActor(
        config,
        arm=True,
        deterministic=args.deterministic,
        evaluation_only=args.eval_only,
        shadow_stop=args.shadow_stop,
        active_stop=args.active_stop,
        image_encoder_type=args.encoder,
    ).run()


if __name__ == "__main__":
    main()
