"""Row view over PythonDict data.

Since mloda 0.9.0 PythonDict's native representation is columnar ``dict[str, list]``, but every
feature group here is row-wise (a chunk, a passage, a redacted text is a row). Feature groups may
still return ``list[dict]`` with homogeneous keys: mloda normalizes that back to columnar.
"""

from typing import Any, Dict, List

from mloda.user import columnar_to_rows, is_columnar

__all__ = ["as_rows"]


def as_rows(data: Any) -> List[Dict[str, Any]]:
    """Rows for a row-wise feature group. Anything not row-wise or columnar yields no rows.

    ``columnar_to_rows`` is strict by design (it raises on a non-columnar dict), so the leniency
    lives here: see the "Columnar helpers" section of the mloda compute-frameworks guide.
    """
    if isinstance(data, list):
        return data
    if is_columnar(data):
        return columnar_to_rows(data)
    return []
