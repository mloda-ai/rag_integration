"""Aggregation-aware rule-based text-to-SQL backend.

Second concrete for the ``structured`` family: zero-download, deterministic, no
LLM. Where :class:`RuleBasedSql` covers count/filter/list intents, this backend
adds *aggregation* (avg/min/max/sum over a column named in the question),
translating it to a parameterised aggregate ``SELECT``. Numericness is not
validated: SQLite coerces non-numeric values, so e.g. ``AVG``/``SUM`` over a
text column silently returns ``0.0``. It reuses the base's identifier
whitelist, sqlglot single-SELECT guard, and sqlite execution. The
count/filter/list intents are reimplemented here too (rather than inherited
from ``RuleBasedSql``) so the backend subclasses the family base directly and
satisfies the shared contract suite on its own.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector

# Tokens keep underscores (snake_case columns) and decimals ("2.5"). Negative
# numbers are not supported as filter values (the leading "-" is dropped).
_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)?")

# Natural-language aggregation cues -> SQL aggregate function. Each function is
# a fixed literal (never user text), so interpolating it is injection-safe.
_AGGREGATIONS = {
    "average": "AVG",
    "avg": "AVG",
    "mean": "AVG",
    "minimum": "MIN",
    "min": "MIN",
    "lowest": "MIN",
    "smallest": "MIN",
    "maximum": "MAX",
    "max": "MAX",
    "highest": "MAX",
    "largest": "MAX",
    "sum": "SUM",
    "total": "SUM",
}


class AggregateSql(BaseStructuredConnector):
    """Aggregation-aware rule-based NL->SQL (``structured_backend="aggregate"``).

    Intents, in priority order:

    1. Aggregate (an aggregation cue such as ``average``/``min``/``max``/``sum``
       plus a column named in the question, preferring the first column
       mentioned after the cue):
       ``SELECT <FUNC>(<col>) AS result FROM <table>``.
    2. Count (``how many`` / ``count``): ``SELECT COUNT(*) AS cnt FROM <table>``;
       if the question also names a column followed by a value token, the count
       is filtered: ``SELECT COUNT(*) AS cnt FROM <table> WHERE LOWER(<col>) = ?``.
    3. Equality filter (a column name followed by a value token):
       ``SELECT * FROM <table> WHERE LOWER(<col>) = ?`` (case-insensitive).
    4. Otherwise list all rows: ``SELECT * FROM <table>``.

    Table and column names are validated identifiers (by the base) and are
    interpolated double-quoted, so reserved words work; the aggregate function
    comes from a fixed whitelist; values are always bound parameters, never
    interpolated. Negative numbers are not supported as filter values (the
    tokenizer drops the sign).
    """

    STRUCTURED_BACKENDS = {
        "aggregate": "Aggregation-aware rule-based natural-language-to-SQL (no LLM)",
    }

    PROPERTY_MAPPING = {
        BaseStructuredConnector.STRUCTURED_BACKEND: property_spec(
            "Use 'aggregate' for aggregation-aware text-to-SQL", context=False
        ),
        BaseStructuredConnector.QUESTION: property_spec(
            "Natural-language question to answer over the table", context=False
        ),
        BaseStructuredConnector.TABLE: property_spec("Table name (a simple SQL identifier)", context=False),
        BaseStructuredConnector.COLUMNS: property_spec("Column names (simple SQL identifiers)", context=False),
        BaseStructuredConnector.ROWS: property_spec("Table rows: a list of {column: value} dicts", context=False),
    }

    @classmethod
    def _find_column(cls, tokens: List[str], columns: List[str]) -> Optional[str]:
        """Return the first column named in ``tokens`` (in token order), or None."""
        lowered = {column.lower(): column for column in columns}
        for token in tokens:
            if token in lowered:
                return lowered[token]
        return None

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

        # 1. Aggregation: an aggregation cue plus a column named in the question.
        # Checked first by design, so "the average age" aggregates rather than
        # being read as a filter; the trade-off is that a filter whose *value*
        # token is itself a cue word (e.g. "... status max") would aggregate
        # instead. The column mentioned after the cue is preferred ("the species
        # with the highest age" aggregates age, not species); a column named
        # only before the cue is the fallback. table, column, and the aggregate
        # function are all whitelisted (the function is a fixed literal) and
        # identifiers are interpolated double-quoted, so the f-string is
        # injection-safe.
        for index, token in enumerate(tokens):
            function = _AGGREGATIONS.get(token)
            if function is not None:
                column = cls._find_column(tokens[index + 1 :], columns)
                if column is None:
                    column = cls._find_column(tokens, columns)
                if column is not None:
                    return f'SELECT {function}("{column}") AS result FROM "{table}"', []  # nosec B608
                break

        filter_match = cls._find_filter(tokens, columns)

        # 2. Count (filtered when the question also names a column + value).
        if "count" in token_set or ("how" in token_set and "many" in token_set):
            if filter_match is not None:
                column, value = filter_match
                return f'SELECT COUNT(*) AS cnt FROM "{table}" WHERE LOWER("{column}") = ?', [value]  # nosec B608
            return f'SELECT COUNT(*) AS cnt FROM "{table}"', []  # nosec B608

        # 3. Equality filter: a column name followed by a value token. The value
        # is always returned as a bound parameter, never interpolated.
        if filter_match is not None:
            column, value = filter_match
            return f'SELECT * FROM "{table}" WHERE LOWER("{column}") = ?', [value]  # nosec B608

        # 4. List all.
        return f'SELECT * FROM "{table}"', []  # nosec B608
