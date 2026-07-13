"""Rule-based text-to-SQL backend.

Canonical concrete for the ``structured`` family: zero-download, deterministic,
no LLM. Translates a small set of natural-language intents (count, equality
filter, list-all) into parameterised SQL. There is no mature deterministic
NL->SQL library (every real one needs an LLM), so a transparent rule set is the
right CI anchor; LLM-backed translators are pedigree backends for later.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector

# Tokens keep underscores (snake_case columns) and decimals ("2.5"). Negative
# numbers are not supported as filter values (the leading "-" is dropped).
_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)?")


class RuleBasedSql(BaseStructuredConnector):
    """Rule-based NL->SQL (``structured_backend="rule_based"``).

    Intents, in priority order:

    1. Count (``how many`` / ``count``): ``SELECT COUNT(*) AS cnt FROM <table>``;
       if the question also names a column followed by a value token, the count
       is filtered: ``SELECT COUNT(*) AS cnt FROM <table> WHERE LOWER(<col>) = ?``.
    2. Equality filter (a column name followed by a value token):
       ``SELECT * FROM <table> WHERE LOWER(<col>) = ?`` (case-insensitive).
    3. Otherwise list all rows: ``SELECT * FROM <table>``.

    Table and column names are validated identifiers (by the base) and are
    interpolated double-quoted, so reserved words work; values are always bound
    parameters, never interpolated. Negative numbers are not supported as
    filter values (the tokenizer drops the sign).
    """

    STRUCTURED_BACKENDS = {
        "rule_based": "Rule-based natural-language-to-SQL (no LLM)",
    }

    PROPERTY_MAPPING = {
        BaseStructuredConnector.STRUCTURED_BACKEND: property_spec(
            "Use 'rule_based' for rule-based text-to-SQL", context=False
        ),
        BaseStructuredConnector.QUESTION: property_spec(
            "Natural-language question to answer over the table", context=False
        ),
        BaseStructuredConnector.TABLE: property_spec("Table name (a simple SQL identifier)", context=False),
        BaseStructuredConnector.COLUMNS: property_spec("Column names (simple SQL identifiers)", context=False),
        BaseStructuredConnector.ROWS: property_spec("Table rows: a list of {column: value} dicts", context=False),
    }

    @classmethod
    def _find_filter(cls, tokens: List[str], columns: List[str]) -> Optional[Tuple[str, str]]:
        """Return ``(column, value)`` for the first column (in declaration
        order) named in the question and followed by a value token, or None."""
        for column in columns:
            lowered = column.lower()
            if lowered in tokens:
                position = tokens.index(lowered)
                if position + 1 < len(tokens):
                    return column, tokens[position + 1]
        return None

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: List[str]) -> Tuple[str, List[Any]]:
        tokens = _TOKEN_RE.findall(question.lower())
        token_set = set(tokens)

        # table and column are validated identifiers (by the base, quotes
        # excluded) and interpolated double-quoted; the filter value is always
        # returned as a bound parameter, never interpolated.
        filter_match = cls._find_filter(tokens, columns)

        if "count" in token_set or ("how" in token_set and "many" in token_set):
            if filter_match is not None:
                column, value = filter_match
                return f'SELECT COUNT(*) AS cnt FROM "{table}" WHERE LOWER("{column}") = ?', [value]  # nosec B608
            return f'SELECT COUNT(*) AS cnt FROM "{table}"', []  # nosec B608

        if filter_match is not None:
            column, value = filter_match
            return f'SELECT * FROM "{table}" WHERE LOWER("{column}") = ?', [value]  # nosec B608

        return f'SELECT * FROM "{table}"', []  # nosec B608
