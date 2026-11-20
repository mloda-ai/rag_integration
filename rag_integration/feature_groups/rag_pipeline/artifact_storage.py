"""Storage location and atomic writes shared by the artifact classes."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path


def default_storage_dir() -> Path:
    """Private per-user dir under $XDG_CACHE_HOME (or ~/.cache), never the shared system temp dir."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    cache = Path(xdg) if os.path.isabs(xdg) else Path.home() / ".cache"
    return cache / "rag_integration" / "artifacts"


def resolve_storage_dir(storage_path: str | None, create: bool) -> Path:
    """Return `storage_path`, or the default dir; with `create`, make it (the default as 0o700)."""
    if storage_path is not None:
        storage_dir = Path(storage_path)
        if create:
            storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir
    storage_dir = default_storage_dir()
    if create:
        storage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return storage_dir


def atomic_write(path: Path, write: Callable[[str], object]) -> None:
    """Write via a hidden temp file in the same dir, then `os.replace` it, so readers never see a partial file."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(str(tmp))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
