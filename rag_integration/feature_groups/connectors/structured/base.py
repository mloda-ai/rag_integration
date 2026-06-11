"""Base class for the ``structured`` connector family.

Contract: ``question + table -> SQL -> typed rows``.

A structured connector answers a natural-language question over a relational
table by translating it to SQL and executing it. It is a ROOT FeatureGroup: the
table (name, columns, rows) is passed inline through ``Options`` and executed in
an in-memory SQLite database, so the family is self-contained and
contract-testable without an external database.

Output (single row, keyed by the root feature name)::

    {"structured_rows": {"sql": "SELECT ...", "rows": [{"col": value, ...}, ...]}}

The base owns identifier validation, SQL safety (it parses the generated SQL
with sqlglot and rejects anything that is not a single top-level ``SELECT``
statement, with no set operations and no stacked statements), the
in-memory SQLite execution, and row typing. A backend implements only
:meth:`_to_sql` (the natural-language-to-SQL translation), which returns a
parameterised statement so values never reach SQL by string interpolation.
"""

from __future__ import annotations

import re
import sqlite3
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.errors import InvalidOptionError, SqlSafetyError
from rag_integration.feature_groups.connectors.mixins import OptionsMixin, SingleQueryPerRunMixin

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class BaseStructuredConnector(SingleQueryPerRunMixin, OptionsMixin, FeatureGroup):
    """Root FeatureGroup for structured (text-to-SQL) connector backends.

    A concrete backend declares its selector value in ``STRUCTURED_BACKENDS`` and
    implements :meth:`_to_sql`; selection is via
    :meth:`match_feature_group_criteria`, gating on
    ``structured_backend in cls.STRUCTURED_BACKENDS``.
    """

    ROOT_FEATURE_NAME = "structured_rows"

    # Option keys.
    STRUCTURED_BACKEND = "structured_backend"
    QUESTION = "question"
    TABLE = "table_name"
    COLUMNS = "columns"
    ROWS = "rows"

    STRUCTURED_BACKENDS: Dict[str, str] = {}

    PROPERTY_MAPPING = {
        STRUCTURED_BACKEND: {"explanation": "Which structured (text-to-SQL) backend to use"},
        QUESTION: {"explanation": "Natural-language question to answer over the table"},
        TABLE: {"explanation": "Table name (a simple SQL identifier)"},
        COLUMNS: {"explanation": "Column names (simple SQL identifiers)"},
        ROWS: {"explanation": "Table rows: a list of {column: value} dicts"},
    }

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.ROOT_FEATURE_NAME})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match the root feature name only for a backend this concrete declares."""
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.STRUCTURED_BACKEND)
        return backend in cls.STRUCTURED_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (the table arrives via Options)."""
        return None

    @classmethod
    def _validate_identifier(cls, name: str, kind: str) -> str:
        """Reject any table/column name that is not a simple SQL identifier.

        Identifiers cannot be parameterised, so this whitelist is what keeps the
        generated SQL injection-safe (values, by contrast, are always bound)."""
        if not _IDENT_RE.fullmatch(name):
            raise InvalidOptionError(
                f"{cls.__name__}: invalid {kind} identifier {name!r}; expected a simple SQL identifier."
            )
        return name

    @classmethod
    @abstractmethod
    def _to_sql(cls, question: str, table: str, columns: List[str]) -> Tuple[str, List[Any]]:
        """Translate ``question`` into a read-only SQL statement.

        Returns ``(sql, params)`` where ``sql`` is a single ``SELECT`` over
        ``table`` using ``?`` placeholders for any values, and ``params`` are the
        bound values in order. ``table`` and ``columns`` are already validated
        identifiers. The base parses the result with sqlglot and rejects
        anything but a single top-level bare ``SELECT`` statement (no set
        operations, no stacked statements).
        """
        ...

    @classmethod
    def _validate_select(cls, sql: str) -> None:
        """Require ``sql`` to be a single top-level ``SELECT`` statement
        (no set operations, no stacked statements)."""
        import sqlglot
        import sqlglot.expressions as exp
        from sqlglot.errors import SqlglotError

        try:
            statements = sqlglot.parse(sql, read="sqlite")
        except SqlglotError as error:
            raise SqlSafetyError(f"{cls.__name__}._to_sql produced unparseable SQL: {sql!r}") from error
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            raise SqlSafetyError(
                f"{cls.__name__}._to_sql must produce a single top-level SELECT statement, got: {sql!r}"
            )

    @classmethod
    def _query(
        cls,
        question: str,
        table: str,
        columns: List[str],
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Translate, validate, and execute the query over an in-memory SQLite table."""
        table = cls._validate_identifier(table, "table")
        columns = [cls._validate_identifier(c, "column") for c in columns]
        if not columns:
            raise InvalidOptionError(f"{cls.__name__}: at least one column is required.")
        if len({c.lower() for c in columns}) != len(columns):
            raise InvalidOptionError(
                f"{cls.__name__}: duplicate column names (SQLite is case-insensitive) are not allowed: {columns}."
            )

        sql, params = cls._to_sql(question, table, columns)
        cls._validate_select(sql)

        connection = sqlite3.connect(":memory:")
        try:
            # table and columns are whitelisted identifiers (validated above,
            # quotes excluded by the whitelist) and double-quoted so reserved
            # words work; all row values are bound parameters, never interpolated.
            column_ddl = ", ".join(f'"{c}"' for c in columns)
            connection.execute(f'CREATE TABLE "{table}" ({column_ddl})')
            placeholders = ", ".join("?" for _ in columns)
            insert_sql = f'INSERT INTO "{table}" ({column_ddl}) VALUES ({placeholders})'  # nosec B608
            connection.executemany(insert_sql, [[row.get(c) for c in columns] for row in rows])

            # Defense-in-depth: make the connection read-only at the SQLite
            # level before running backend SQL, so any write attempt fails
            # regardless of sqlglot version or validation behavior.
            connection.execute("PRAGMA query_only = ON")

            cursor = connection.execute(sql, params)
            result_columns = [description[0] for description in cursor.description]
            result_rows = [dict(zip(result_columns, record)) for record in cursor.fetchall()]
        finally:
            connection.close()

        return {"sql": sql, "rows": result_rows}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Answer the question over the supplied table, return the SQL and rows."""
        cls._assert_single_feature(features)
        for feature in features.features:
            options = feature.options
            question = str(cls._require_option(options, cls.QUESTION))
            table = str(cls._require_option(options, cls.TABLE))
            raw_columns = cls._require_option(options, cls.COLUMNS)
            if not isinstance(raw_columns, (list, tuple)):
                raise InvalidOptionError(f"{cls.__name__}: '{cls.COLUMNS}' must be a list or tuple of column names.")
            columns = [str(c) for c in raw_columns]
            raw_rows = cls._require_option(options, cls.ROWS)
            if not isinstance(raw_rows, (list, tuple)) or not all(isinstance(row, dict) for row in raw_rows):
                raise InvalidOptionError(f"{cls.__name__}: '{cls.ROWS}' must be a list or tuple of dicts.")
            rows = [dict(row) for row in raw_rows]
            return [{cls.ROOT_FEATURE_NAME: cls._query(question, table, columns, rows)}]
        return []
