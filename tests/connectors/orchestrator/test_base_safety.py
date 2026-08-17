"""Base-level safety tests for the orchestrator family, through the production path.

Pins the two guards in ``BaseOrchestratorConnector._answer`` that no real
backend triggers but a future (e.g. server-stub) backend could: a fabricated
document (doc_id not in the corpus) and a non-empty answer with no documents.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector

_CORPUS = [{"doc_id": "d0", "text": "a real document"}]


class _FabricatingBackend(BaseOrchestratorConnector):
    ORCHESTRATOR_BACKENDS: ClassVar = {"_fabricating_stub": "test-only stub"}

    @classmethod
    def _run(cls, query: str, corpus: list[dict[str, Any]], top_k: int) -> tuple[str, list[dict[str, Any]]]:
        return "answer", [{"doc_id": "not_in_corpus", "text": "fabricated", "score": 1.0}]


class _AnswerWithoutDocumentsBackend(BaseOrchestratorConnector):
    ORCHESTRATOR_BACKENDS: ClassVar = {"_answer_no_docs_stub": "test-only stub"}

    @classmethod
    def _run(cls, query: str, corpus: list[dict[str, Any]], top_k: int) -> tuple[str, list[dict[str, Any]]]:
        return "an ungrounded answer", []


def test_query_rejects_fabricated_document() -> None:
    with pytest.raises(ValueError):
        _FabricatingBackend._answer("q", _CORPUS, 5)


def test_query_rejects_nonempty_answer_without_documents() -> None:
    with pytest.raises(ValueError):
        _AnswerWithoutDocumentsBackend._answer("q", _CORPUS, 5)
