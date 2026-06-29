"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Resolve a safe training device."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
