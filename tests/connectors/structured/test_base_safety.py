"""Base-level safety tests for the structured family, through the production path.

Pins that ``BaseStructuredConnector._query`` itself rejects a non-SELECT or
stacked statement produced by a backend (not just the isolated
``_validate_select``), so a regression that dropped the guard from ``_query``
would fail here. Also pins identifier handling: reserved-word identifiers work
(double-quoted interpolation) and case-insensitive duplicate columns are
rejected, plus the ``calculate_feature`` option-type guards.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector
from rag_integration.feature_groups.connectors.structured.rule_based_sql import RuleBasedSql


class _DeleteBackend(BaseStructuredConnector):
    """A deliberately malicious backend that emits a non-SELECT statement."""

    STRUCTURED_BACKENDS: ClassVar = {"_delete_stub": "test-only stub"}

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: list[str]) -> tuple[str, list[Any]]:
        return "DELETE FROM pets", []


class _StackedBackend(BaseStructuredConnector):
    """A deliberately malicious backend that stacks a write after a SELECT."""

    STRUCTURED_BACKENDS: ClassVar = {"_stacked_stub": "test-only stub"}

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: list[str]) -> tuple[str, list[Any]]:
        return "SELECT 1; DROP TABLE pets", []


class _UnterminatedBackend(BaseStructuredConnector):
    """A broken backend whose SQL fails sqlglot tokenization (unterminated string)."""

    STRUCTURED_BACKENDS: ClassVar = {"_unterminated_stub": "test-only stub"}

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: list[str]) -> tuple[str, list[Any]]:
        return "SELECT 'unterminated FROM pets", []


def test_query_rejects_non_select_through_production_path() -> None:
    with pytest.raises(ValueError):
        _DeleteBackend._query("delete everything", "pets", ["name"], [{"name": "Rex"}])


def test_query_rejects_stacked_statements_through_production_path() -> None:
    """A stacked query ("SELECT 1; DROP TABLE pets") is rejected by ``_query`` itself."""
    with pytest.raises(ValueError):
        _StackedBackend._query("anything", "pets", ["name"], [{"name": "Rex"}])


def test_validate_select_rejects_stacked_statements() -> None:
    """Regression: older sqlglot ``parse_one`` silently accepted stacked queries."""
    with pytest.raises(ValueError):
        BaseStructuredConnector._validate_select("SELECT 1; DROP TABLE pets")


def test_validate_select_rejects_tokenize_errors() -> None:
    """A tokenizer failure (unterminated string literal) surfaces as ValueError, not raw."""
    with pytest.raises(ValueError):
        BaseStructuredConnector._validate_select("SELECT 'unterminated FROM pets")


def test_query_rejects_tokenize_errors_through_production_path() -> None:
    with pytest.raises(ValueError):
        _UnterminatedBackend._query("anything", "pets", ["name"], [{"name": "Rex"}])


def test_validate_select_accepts_quoted_identifiers() -> None:
    """The double-quoted identifiers the family generates pass the SQL guard."""
    BaseStructuredConnector._validate_select('SELECT * FROM "order" WHERE LOWER("from") = ?')


def test_reserved_word_identifiers_work_end_to_end() -> None:
    """A table named "order" with a column named "from" works (quoted interpolation)."""
    rows = [{"from": "berlin", "value": 1}, {"from": "munich", "value": 2}]
    result = RuleBasedSql._query("which entries have from berlin", "order", ["from", "value"], rows)
    assert [row["value"] for row in result["rows"]] == [1]


def test_query_rejects_duplicate_columns_case_insensitively() -> None:
    """SQLite column names are case-insensitive, so ["Name", "name"] must be rejected."""
    with pytest.raises(ValueError):
        RuleBasedSql._query("anything", "pets", ["Name", "name"], [{"Name": "Rex"}])


def _make_features(context: dict[str, Any]) -> Any:
    """Build a minimal FeatureSet mock whose options resolve from ``context``."""
    feature = MagicMock()
    feature.options.get.side_effect = context.get
    features = MagicMock()
    features.features = [feature]
    return features


def test_calculate_feature_rejects_non_list_columns() -> None:
    """A plain-string COLUMNS must not be iterated into characters."""
    features = _make_features(
        {
            BaseStructuredConnector.QUESTION: "how many",
            BaseStructuredConnector.TABLE: "pets",
            BaseStructuredConnector.COLUMNS: "name",
            BaseStructuredConnector.ROWS: [{"name": "Rex"}],
        }
    )
    with pytest.raises(ValueError, match="columns"):
        RuleBasedSql.calculate_feature(None, features)


def test_calculate_feature_rejects_non_dict_rows() -> None:
    features = _make_features(
        {
            BaseStructuredConnector.QUESTION: "how many",
            BaseStructuredConnector.TABLE: "pets",
            BaseStructuredConnector.COLUMNS: ["name"],
            BaseStructuredConnector.ROWS: ["Rex"],
        }
    )
    with pytest.raises(ValueError, match="rows"):
        RuleBasedSql.calculate_feature(None, features)
