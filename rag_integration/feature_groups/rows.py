"""Row view over the data mloda hands a feature group.

Since mloda 0.9.0 PythonDict's native representation is columnar ``dict[str, list]``, but every
feature group here is row-wise (a chunk, a passage, a redacted text is a row). Feature groups may
still return ``list[dict]`` with homogeneous keys: mloda normalizes that back to columnar.
"""

from typing import Any, Dict, List

from mloda.user import columnar_to_rows, is_columnar

__all__ = ["as_rows"]


def as_rows(data: Any) -> List[Dict[str, Any]]:
    """Rows for a row-wise feature group, from what mloda passes to ``calculate_feature``.

    Only for that data: a columnar frame, or an already row-wise list. Not a general converter,
    because ``is_columnar`` cannot tell a columnar frame from a single row whose cells all hold
    equal-length lists, and a connector row (``{"retrieved_passages": [...]}``) is exactly that.

    Anything else is a contract violation and raises. Returning no rows instead would turn a
    broken upstream into an empty result, which reads downstream as a real (and wrong) answer.
    """
    if isinstance(data, list):
        return data
    if is_columnar(data):
        return columnar_to_rows(data)
    raise TypeError(f"expected a columnar dict or a list of rows from mloda, got {type(data).__name__}: {data!r}")
