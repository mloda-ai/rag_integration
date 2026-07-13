"""Tests for the ``flatten_result`` shape dispatch.

mloda 0.10.0's ``columnar_to_rows`` raises on non-columnar input, so the tolerant
shape dispatch (row-wise passthrough, empty fallback) lives in ``flatten_result``
itself. This suite pins that dispatch.
"""

from __future__ import annotations

from tests.integration.helpers import flatten_result


class TestFlattenResult:
    def test_bare_columnar_dict_pivots_to_rows(self) -> None:
        data = {"doc_id": ["a", "b"], "text": ["one", "two"]}
        assert flatten_result(data) == [
            {"doc_id": "a", "text": "one"},
            {"doc_id": "b", "text": "two"},
        ]

    def test_wrapped_columnar_partition_pivots_to_rows(self) -> None:
        assert flatten_result([{"doc_id": ["a"]}]) == [{"doc_id": "a"}]

    def test_wrapped_row_list_is_kept_row_wise(self) -> None:
        rows = [{"doc_id": "a"}, {"doc_id": "b"}]
        assert flatten_result([rows]) == rows

    def test_scalar_valued_leading_dict_passes_through_as_rows(self) -> None:
        rows = [{"doc_id": "a", "score": 0.5}, {"doc_id": "b", "score": 0.7}]
        assert flatten_result(rows) == rows

    def test_empty_leading_dict_yields_empty_list(self) -> None:
        assert flatten_result([{}]) == []

    def test_empty_inputs_yield_empty_list(self) -> None:
        assert flatten_result([]) == []
        assert flatten_result({}) == []
        assert flatten_result(None) == []
