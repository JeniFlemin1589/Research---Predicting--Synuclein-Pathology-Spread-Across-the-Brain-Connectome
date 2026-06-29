"""File IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with consistent formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)

