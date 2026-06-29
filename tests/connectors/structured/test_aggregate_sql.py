"""Contract test for :class:`AggregateSql` (zero-download CI anchor).

Inherits the whole structured contract suite (count/filter/safety), then adds a
backend-specific proof: an aggregation question runs a real aggregate query and
returns a known computed value (avg of ``[2, 3, 5, 2]`` = ``3.0``), which the
count/filter-only sibling cannot answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Type

from rag_integration.feature_groups.connectors.structured.aggregate_sql import AggregateSql
from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector
from tests.connectors.structured.structured_contract import StructuredConnectorContractBase


class TestAggregateSql(StructuredConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseStructuredConnector]:
        return AggregateSql

    @classmethod
    def backend_value(cls) -> str:
        return "aggregate"

    @classmethod
    def table_name(cls) -> str:
        return "pets"

    @classmethod
    def columns(cls) -> List[str]:
        return ["name", "species", "age"]

    @classmethod
    def rows(cls) -> List[Dict[str, Any]]:
        return [
            {"name": "Whiskers", "species": "cat", "age": 2},
            {"name": "Rex", "species": "dog", "age": 3},
            {"name": "Felix", "species": "cat", "age": 5},
            {"name": "Tom", "species": "cat", "age": 2},
        ]

    @classmethod
    def key_column(cls) -> str:
        return "name"

    @classmethod
    def count_question(cls) -> str:
        return "how many pets are there"

    @classmethod
    def filter_question(cls) -> str:
        return "which pets have species cat"

    @classmethod
    def expected_filter_keys(cls) -> Set[str]:
        return {"Whiskers", "Felix", "Tom"}

    @classmethod
    def filter_value(cls) -> str:
        return "cat"

    # -- Backend-specific proof: aggregation ----------------------------------

    def test_average_question_returns_computed_value(self) -> None:
        """Not-a-stub proof for this backend: an aggregation question runs a real
        AVG over the column and returns the known value (avg of [2,3,5,2] = 3.0)."""
        result = self._query("what is the average age")
        assert 'AVG("age")' in result["sql"]
        assert len(result["rows"]) == 1
        (only_value,) = result["rows"][0].values()
        assert only_value == 3.0

    def test_max_question_returns_computed_value(self) -> None:
        """A second aggregate intent: MAX over the column returns the known max."""
        result = self._query("what is the maximum age")
        assert 'MAX("age")' in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 5

    def test_aggregate_prefers_column_after_the_cue(self) -> None:
        """A non-target column named before the cue must not win: "the species with
        the highest age" aggregates age, not species."""
        result = self._query("what is the species with the highest age")
        assert 'MAX("age")' in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 5

    def test_aggregate_falls_back_to_column_before_the_cue(self) -> None:
        """If no column follows the cue, the any-position match still aggregates."""
        result = self._query("what is the age maximum")
        assert 'MAX("age")' in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 5

    def test_count_with_filter_counts_only_matching_rows(self) -> None:
        """Count intent must keep the filter: COUNT over cats only, not all rows."""
        result = self._query("how many pets have species cat")
        assert "?" in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 3

    def test_snake_case_column_aggregates(self) -> None:
        """The tokenizer keeps underscores, so snake_case columns are recognised."""
        rows = [{"pet_name": "Rex", "unit_price": 2}, {"pet_name": "Felix", "unit_price": 4}]
        result = AggregateSql._query("what is the average unit_price", "items", ["pet_name", "unit_price"], rows)
        assert 'AVG("unit_price")' in result["sql"]
        (only_value,) = result["rows"][0].values()
        assert only_value == 3.0

    def test_average_over_empty_table_returns_none(self) -> None:
        """Pin behavior on an empty table: AVG returns a single row holding None."""
        result = AggregateSql._query("what is the average age", self.table_name(), self.columns(), [])
        (only_value,) = result["rows"][0].values()
        assert only_value is None

    # -- Honest surface: narrowing --------------------------------------------

    def test_ordering_intent_not_supported_lists_all(self) -> None:
        """Honest surface: ordering/grouping is not supported. An ORDER-BY-style
        question with no aggregation cue falls back to list-all rather than
        emitting an ORDER BY clause it does not implement."""
        sql, params = AggregateSql._to_sql("list the pets ordered by age", self.table_name(), self.columns())
        assert sql == 'SELECT * FROM "pets"'
        assert "ORDER BY" not in sql
        assert params == []

    def test_negative_filter_value_sign_dropped(self) -> None:
        """Honest surface: negative filter values are unsupported. The tokenizer
        drops the leading sign, so "-5" binds as "5" (pinned, not a different
        signed match)."""
        sql, params = AggregateSql._to_sql("which pets have age -5", "pets", ["name", "age"])
        assert 'LOWER("age") = ?' in sql
        assert params == ["5"]
