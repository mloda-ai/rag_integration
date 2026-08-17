"""Cross-family test: every connector base rejects a multi-feature FeatureSet.

When the engine groups two features with the same root feature name into one
FeatureSet (e.g. two ``graph_passages`` requests with different options), the
base must raise ``ValueError`` instead of silently returning only the first
result.  All family bases inherit ``SingleQueryPerRunMixin``; this file pins
that each one wires up ``_assert_single_feature`` in ``calculate_feature``.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest

from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector
from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.connectors.graph_rag.kg_source import BaseKnowledgeGraphSource
from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.structured.base import BaseStructuredConnector


def _two_feature_set() -> Any:
    features = MagicMock()
    features.features = [MagicMock(), MagicMock()]
    return features


class _MinimalRetrieve(BaseRetrieveConnector):
    RETRIEVE_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _rank(cls, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        return []


class _MinimalRerank(BaseRerankConnector):
    RERANK_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _rank(cls, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        return []


class _MinimalGenerate(BaseGenerateConnector):
    GENERATE_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return "", []


class _MinimalGraphRag(BaseGraphRagConnector):
    GRAPH_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _rank(cls, query: str, texts: list[str], edges: list[tuple[int, int]], top_k: int) -> list[tuple[int, float]]:
        return []


class _MinimalKgSource(BaseKnowledgeGraphSource):
    KG_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _build_graph(cls, options: Any) -> dict[str, Any]:
        return {"nodes": [], "edges": []}


class _MinimalStructured(BaseStructuredConnector):
    STRUCTURED_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _to_sql(cls, question: str, table: str, columns: list[str]) -> tuple[str, list[Any]]:
        return f"SELECT * FROM {table}", []  # nosec B608


class _MinimalOrchestrator(BaseOrchestratorConnector):
    ORCHESTRATOR_BACKENDS: ClassVar = {"_stub": "test-only"}

    @classmethod
    def _run(cls, query: str, corpus: list[dict[str, Any]], top_k: int) -> tuple[str, list[dict[str, Any]]]:
        return "", []


@pytest.mark.parametrize(
    "cls",
    [
        _MinimalRetrieve,
        _MinimalRerank,
        _MinimalGenerate,
        _MinimalGraphRag,
        _MinimalKgSource,
        _MinimalStructured,
        _MinimalOrchestrator,
    ],
)
def test_multi_feature_set_raises(cls: Any) -> None:
    with pytest.raises(ValueError, match="one query per run"):
        cls.calculate_feature([], _two_feature_set())
