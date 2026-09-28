"""Locate the local HIL-SERL checkout without modifying it."""

from __future__ import annotations

import os
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def ensure_hil_serl_importable() -> Path:
    root = Path(
        os.environ.get("HIL_SERL_ROOT", WORKSPACE_ROOT / "hil-serl")
    ).expanduser().resolve()
    source = root / "serl_launcher"
    if not (source / "serl_launcher").is_dir():
        raise FileNotFoundError(
            f"HIL-SERL Python source not found under {source}; set HIL_SERL_ROOT"
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return root
