"""Storage location and atomic writes shared by the artifact classes."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path


def default_storage_dir() -> Path:
    """Private per-user dir under $XDG_CACHE_HOME (or ~/.cache), never the shared system temp dir."""
    xdg = os.environ.get("XDG_CACHE_HOME", "")
    cache = Path(xdg) if os.path.isabs(xdg) else Path.home() / ".cache"
    return cache / "rag_integration" / "artifacts"


def resolve_storage_dir(storage_path: str | None, create: bool) -> Path:
    """Return `storage_path`, or the default dir checked to be private; with `create`, make it."""
    if storage_path is not None:
        storage_dir = Path(storage_path)
        if create:
            storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir
    storage_dir = default_storage_dir()
    if create:
        storage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if storage_dir.exists():
        info = storage_dir.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError(
                f"Default artifact dir {storage_dir} must be owned by the current user and not shared; "
                "fix its permissions or set artifact_storage_path"
            )
    return storage_dir


def atomic_write(writes: Mapping[Path, Callable[[str], object]]) -> None:
    """Write each file to a hidden temp file, then `os.replace` all of them; nothing is replaced if a write fails."""
    temps = {path: path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp") for path in writes}
    try:
        for path, write in writes.items():
            write(str(temps[path]))
        for path, tmp in temps.items():
            os.replace(tmp, path)
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)
