"""Shared helpers for visualization scripts."""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_figures_dir(out_dir: str | Path) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_if_exists(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    return p if p.exists() else None
