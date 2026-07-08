"""Tests for the columnar helpers backing every PythonDict feature group.

mloda 0.9.0 made PythonDict columnar ``dict[str, list]``; ``columnar_to_rows`` is the
row-wise read path in each ``calculate_feature`` and ``homogenize_rows`` keeps source
rows on the uniform key schema the columnar output contract requires. Both are
hand-rolled here (mloda ships ``rows_to_columnar`` but no public inverse), so this
suite pins their edge cases directly.
"""

from __future__ import annotations

from typing import Any

from rag_integration.feature_groups.columnar import columnar_to_rows, homogenize_rows


class TestColumnarToRows:
    def test_pivots_columnar_dict_to_rows(self) -> None:
        data = {"doc_id": ["a", "b"], "text": ["one", "two"]}
        assert columnar_to_rows(data) == [
            {"doc_id": "a", "text": "one"},
            {"doc_id": "b", "text": "two"},
        ]

    def test_preserves_column_order_in_rows(self) -> None:
        data = {"z": [1], "a": [2]}
        assert list(columnar_to_rows(data)[0].keys()) == ["z", "a"]

    def test_list_passes_through_unchanged(self) -> None:
        rows = [{"doc_id": "a"}]
        assert columnar_to_rows(rows) is rows

    def test_schemaless_empty_dict_yields_empty_list(self) -> None:
        assert columnar_to_rows({}) == []

    def test_schema_bearing_zero_row_dict_yields_empty_list(self) -> None:
        assert columnar_to_rows({"doc_id": []}) == []

    def test_non_dict_non_list_yields_empty_list(self) -> None:
        assert columnar_to_rows(None) == []
        assert columnar_to_rows("text") == []

    def test_none_cell_values_survive_the_pivot(self) -> None:
        data = {"doc_id": ["a", "b"], "author": [None, "jane"]}
        assert columnar_to_rows(data) == [
            {"doc_id": "a", "author": None},
            {"doc_id": "b", "author": "jane"},
        ]


class TestHomogenizeRows:
    def test_backfills_missing_keys_with_none(self) -> None:
        rows = [{"doc_id": "a"}, {"doc_id": "b", "author": "jane"}]
        assert homogenize_rows(rows) == [
            {"doc_id": "a", "author": None},
            {"doc_id": "b", "author": "jane"},
        ]

    def test_key_order_follows_first_occurrence(self) -> None:
        rows = [{"b": 1}, {"a": 2}, {"c": 3}]
        assert all(list(row.keys()) == ["b", "a", "c"] for row in homogenize_rows(rows))

    def test_uniform_rows_are_copied_unchanged(self) -> None:
        rows = [{"doc_id": "a", "text": "one"}]
        result = homogenize_rows(rows)
        assert result == rows
        assert result[0] is not rows[0]

    def test_empty_input_yields_empty_list(self) -> None:
        assert homogenize_rows([]) == []

    def test_explicit_none_values_are_kept(self) -> None:
        rows: list[dict[str, Any]] = [{"doc_id": "a", "author": None}, {"doc_id": "b"}]
        assert homogenize_rows(rows) == [
            {"doc_id": "a", "author": None},
            {"doc_id": "b", "author": None},
        ]
