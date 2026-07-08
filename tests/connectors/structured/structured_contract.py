"""Inheritable contract-test suite for the ``structured`` connector family.

Beyond matching/shape, this suite proves the SQL actually ran (a count returns
the real row count) and the filter is precise (it returns exactly the matching
subset, not all rows), and that the base's SQL safety holds (non-SELECT and bad
identifiers are rejected).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Type

import pytest

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.columnar import columnar_to_rows
from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector


class StructuredConnectorContractBase(ABC):
    """Contract every structured (text-to-SQL) backend must satisfy."""

    # -- Adapter methods ------------------------------------------------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> Type[BaseStructuredConnector]:
        """Return the concrete ``BaseStructuredConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``structured_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def table_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def columns(cls) -> List[str]: ...

    @classmethod
    @abstractmethod
    def rows(cls) -> List[Dict[str, Any]]: ...

    @classmethod
    @abstractmethod
    def key_column(cls) -> str:
        """A column whose values uniquely identify rows (used to check filter results)."""

    @classmethod
    @abstractmethod
    def count_question(cls) -> str:
        """A natural-language 'how many' question over the whole table."""

    @classmethod
    @abstractmethod
    def filter_question(cls) -> str:
        """A question that should filter to a strict, non-empty subset of rows."""

    @classmethod
    @abstractmethod
    def expected_filter_keys(cls) -> Set[str]:
        """The ``key_column`` values expected from ``filter_question`` (a strict subset)."""

    @classmethod
    @abstractmethod
    def filter_value(cls) -> str:
        """The literal value that ``filter_question`` filters on (used to prove binding)."""

    # -- Helpers --------------------------------------------------------------

    @classmethod
    def _query(cls, question: str) -> Dict[str, Any]:
        connector = cls.connector_class()
        return connector._query(question, cls.table_name(), cls.columns(), cls.rows())

    @classmethod
    def _run_all(cls, question: str) -> Dict[str, Any]:
        connector = cls.connector_class()
        feature = Feature(
            connector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    connector.STRUCTURED_BACKEND: cls.backend_value(),
                    connector.QUESTION: question,
                    connector.TABLE: cls.table_name(),
                    connector.COLUMNS: cls.columns(),
                    connector.ROWS: cls.rows(),
                }
            ),
        )
        result = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        for partition in result:
            for row in columnar_to_rows(partition):
                if connector.ROOT_FEATURE_NAME in row:
                    answer: Dict[str, Any] = row[connector.ROOT_FEATURE_NAME]
                    return answer
        raise AssertionError(f"run_all returned no '{connector.ROOT_FEATURE_NAME}' row: {result!r}")

    # -- Matching / honest surface --------------------------------------------

    def test_matches_root_feature_for_declared_backend(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.STRUCTURED_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is True

    def test_does_not_match_other_feature_name(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.STRUCTURED_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria("docs", opts) is False

    def test_unknown_backend_does_not_match(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.STRUCTURED_BACKEND: "definitely_not_a_backend_xyz"})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is False

    def test_backend_declared_in_supported_set(self) -> None:
        connector = self.connector_class()
        assert self.backend_value() in connector.STRUCTURED_BACKENDS

    # -- Output contract ------------------------------------------------------

    def test_result_shape(self) -> None:
        result = self._query(self.count_question())
        assert set(result) >= {"sql", "rows"}
        assert isinstance(result["sql"], str)
        assert isinstance(result["rows"], list)
        assert all(isinstance(row, dict) for row in result["rows"])

    def test_count_question_returns_true_row_count(self) -> None:
        """Not-a-stub proof: the SQL actually ran against the data."""
        result = self._query(self.count_question())
        assert len(result["rows"]) == 1
        (only_value,) = result["rows"][0].values()
        assert only_value == len(self.rows())

    def test_filter_question_is_precise(self) -> None:
        """Not-a-stub proof: the filter returns exactly the matching subset, not all rows."""
        result = self._query(self.filter_question())
        keys = {row[self.key_column()] for row in result["rows"]}
        assert keys == self.expected_filter_keys()
        assert len(result["rows"]) < len(self.rows()), "filter did not narrow the result"

    def test_filter_value_is_bound_not_interpolated(self) -> None:
        """Safety: the filter value reaches SQL as a ``?`` placeholder, never as a literal."""
        result = self._query(self.filter_question())
        assert "?" in result["sql"]
        assert self.filter_value() not in result["sql"]

    def test_unmatched_question_lists_all_rows(self) -> None:
        """Pin the list-all fallback: a question matching no intent returns every row."""
        result = self._query("")
        keys = {row[self.key_column()] for row in result["rows"]}
        assert keys == {row[self.key_column()] for row in self.rows()}

    def test_count_on_empty_table_returns_zero(self) -> None:
        """Pin behavior on an empty table: a count question returns 0, not an error."""
        connector = self.connector_class()
        result = connector._query(self.count_question(), self.table_name(), self.columns(), [])
        (only_value,) = result["rows"][0].values()
        assert only_value == 0

    def test_rejects_non_select_sql(self) -> None:
        """Safety: the base rejects any generated statement that is not a SELECT."""
        connector = self.connector_class()
        with pytest.raises(ValueError):
            connector._validate_select("DELETE FROM whatever")

    def test_rejects_bad_identifier(self) -> None:
        """Safety: a non-identifier table/column is rejected (injection guard)."""
        connector = self.connector_class()
        with pytest.raises(ValueError):
            connector._validate_identifier("a; DROP TABLE x", "column")

    def test_query_rejects_bad_table_identifier_end_to_end(self) -> None:
        """Safety through the production path: a malicious table name is rejected
        by ``_query`` itself, not only by the isolated validator."""
        connector = self.connector_class()
        with pytest.raises(ValueError):
            connector._query("anything", "pets; DROP TABLE pets", self.columns(), self.rows())

    def test_query_rejects_bad_column_identifier_end_to_end(self) -> None:
        """Safety through the production path: a malicious column name is rejected
        by ``_query`` itself, not only by the isolated validator."""
        connector = self.connector_class()
        bad_columns = [*self.columns(), 'evil" FROM x; --']
        with pytest.raises(ValueError):
            connector._query("anything", self.table_name(), bad_columns, self.rows())

    def test_idempotent(self) -> None:
        first = self._query(self.filter_question())
        second = self._query(self.filter_question())
        assert first == second

    # -- End to end -----------------------------------------------------------

    def test_end_to_end_run_all(self) -> None:
        result = self._run_all(self.count_question())
        (only_value,) = result["rows"][0].values()
        assert only_value == len(self.rows())
