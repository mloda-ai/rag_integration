"""The row view every row-wise feature group depends on."""

from __future__ import annotations

from typing import Any

import pytest

from rag_integration.feature_groups.rows import as_rows


def test_columnar_frame_pivots_to_rows() -> None:
    assert as_rows({"doc_id": ["d1", "d2"], "text": ["a", "b"]}) == [
        {"doc_id": "d1", "text": "a"},
        {"doc_id": "d2", "text": "b"},
    ]


def test_empty_frame_is_no_rows() -> None:
    """A schema-bearing frame with zero rows is a valid result, not an error."""
    assert as_rows({"doc_id": [], "text": []}) == []


def test_list_passes_through() -> None:
    rows = [{"doc_id": "d1"}]
    assert as_rows(rows) is rows


@pytest.mark.parametrize("data", [None, "text", 5, {"doc_id": "d1", "text": "a"}, {"a": [1, 2], "b": [1]}])
def test_contract_violation_raises(data: Any) -> None:
    """Never degrade a broken upstream to an empty result: downstream would read it as an answer."""
    with pytest.raises(TypeError):
        as_rows(data)
