"""Zero-row results that keep their schema; mloda rejects a result with no columns."""

from __future__ import annotations

from typing import Any

from mloda.provider import FeatureSet


def empty_result(data: Any, features: FeatureSet, extra_columns: tuple[str, ...] = ()) -> dict[str, list[Any]]:
    """Return zero rows carrying the input columns, ``extra_columns`` and every requested feature."""
    columns = [*(data if isinstance(data, dict) else ()), *extra_columns, *(str(f.name) for f in features.features)]
    return {column: [] for column in columns}
