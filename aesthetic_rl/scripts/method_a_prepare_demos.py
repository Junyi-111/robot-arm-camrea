#!/usr/bin/env python3
"""Create a cleaned method-A dataset from immutable raw demonstrations."""

import argparse
import json

from aesthetic_rl.method_a.config import load_config
from aesthetic_rl.method_a.demo_data import prepare_demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="aesthetic_rl/configs/method_a.yaml")
    parser.add_argument("--demo-dir", default="aesthetic_rl/data/demo/demo_data")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output or config.runtime.prepared_demo_path
    summary = prepare_demos(
        args.demo_dir,
        config.runtime.feature_cache_dir,
        output,
        config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
