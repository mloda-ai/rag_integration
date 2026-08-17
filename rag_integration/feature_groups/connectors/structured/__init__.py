"""The ``structured`` connector family: NL question + table -> SQL -> typed rows."""

from __future__ import annotations

from rag_integration.feature_groups.connectors.structured.aggregate_sql import AggregateSql
from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector
from rag_integration.feature_groups.connectors.structured.rule_based_sql import RuleBasedSql

__all__ = ["AggregateSql", "BaseStructuredConnector", "RuleBasedSql"]
