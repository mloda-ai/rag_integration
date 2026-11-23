"""Every module must import on a plain ``pip install rag-integration``, without any extra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Runs in a fresh interpreter: refuse every top-level import outside the stdlib and the base dependency.
_SCRIPT = """
import sys

# The stdlib plus the base dependency closure (mloda has no dependencies); extend it when a base dependency is added.
ALLOWED =sys.stdlib_module_names | {"mloda", "mloda_plugins", "rag_integration"} | set(sys.modules)


class _BaseOnly:
    @staticmethod
    def find_spec(name, path=None, target=None):
        top = name.partition(".")[0]
        if top not in ALLOWED:
            raise ModuleNotFoundError(f"No module named {top!r} (not a base dependency)", name=top)
        return None


sys.meta_path.insert(0, _BaseOnly)

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    pass
else:
    raise SystemExit("blocker inactive: numpy imported")

import importlib
import pkgutil

import rag_integration

for info in pkgutil.walk_packages(rag_integration.__path__, "rag_integration."):
    importlib.import_module(info.name)
"""


def test_every_module_imports_with_base_dependencies_only() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
