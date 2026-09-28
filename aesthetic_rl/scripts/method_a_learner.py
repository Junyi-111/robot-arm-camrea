#!/usr/bin/env python3
"""Run the local method-A learner. This process never controls the robot."""

import argparse
from dataclasses import replace

from aesthetic_rl.method_a.config import load_config
from aesthetic_rl.method_a.learner import MethodALearner, install_signal_handlers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument(
        "--image-encoder",
        choices=["resnet-pretrained", "tiny"],
        default="resnet-pretrained",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="serve the configured checkpoint read-only; no gradients or replay writes",
    )
    parser.add_argument(
        "--eval-checkpoint",
        help="override runtime.evaluation_checkpoint in frozen --eval-only mode",
    )
    parser.add_argument(
        "--eval-output-dir",
        help="override runtime.evaluation_output_dir in frozen --eval-only mode",
    )
    parser.add_argument("--bc-only", action="store_true")
    args = parser.parse_args()
    if args.eval_only and (args.resume or args.bc_only):
        parser.error("--eval-only cannot be combined with --resume or --bc-only")
    if (args.eval_checkpoint or args.eval_output_dir) and not args.eval_only:
        parser.error("evaluation overrides require --eval-only")
    config = load_config(args.config)
    if args.eval_checkpoint or args.eval_output_dir:
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
    learner = MethodALearner(
        config,
        image_encoder_type=args.image_encoder,
        resume=args.resume,
        evaluation_only=args.eval_only,
    )
    install_signal_handlers(learner)
    try:
        learner.run(bc_only=args.bc_only)
    finally:
        learner.close()


if __name__ == "__main__":
    main()
