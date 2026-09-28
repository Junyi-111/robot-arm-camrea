#!/usr/bin/env python3
"""Print a read-only audit of the keyboard demonstrations."""

import argparse
import json

from aesthetic_rl.method_a.demo_data import audit_demos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-dir",
        default="aesthetic_rl/data/demo/demo_data",
    )
    args = parser.parse_args()
    print(json.dumps(audit_demos(args.demo_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
