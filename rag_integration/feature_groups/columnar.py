"""Columnar helpers for the PythonDict framework.

mloda 0.9.0 made PythonDict's native representation columnar ``dict[str, list]``
(one entry per column, values aligned by row index). The feature groups here read
their upstream input row-wise, so they pivot the columnar input back to rows.
"""

from __future__ import annotations

from typing import Any, Dict, List


def columnar_to_rows(data: Any) -> List[Dict[str, Any]]:
    """Pivot columnar ``dict[str, list]`` input to row-wise ``list[dict]``.

    A ``list`` is returned unchanged (already row-wise); anything else, including
    the schema-less empty dict, yields an empty list.
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict) or not data:
        return []
    columns = list(data.keys())
    n_rows = len(data[columns[0]])
    return [{column: data[column][i] for column in columns} for i in range(n_rows)]


def homogenize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give every row the same key set, backfilling missing keys with ``None``.

    mloda 0.9.0's columnar output contract rejects rows with heterogeneous keys.
    Dataset sources whose row types carry different fields (e.g. query rows add
    ``relevant_doc_ids``) pass through here so the union schema is uniform.
    """
    all_keys: Dict[str, None] = {}
    for row in rows:
        for key in row:
            all_keys.setdefault(key, None)
    return [{key: row.get(key) for key in all_keys} for row in rows]
