"""Contract test for :class:`HaystackOrchestrator`.

Runs a real Haystack in-memory BM25 pipeline (zero-download, so it runs on CI).
Skips cleanly when the ``orchestrator`` extra is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.orchestrator.haystack_orchestrator import HaystackOrchestrator
from tests.connectors.orchestrator.orchestrator_contract import OrchestratorConnectorContractBase

# Clean skip (not an error) when the `orchestrator` extra is not installed.
pytest.importorskip("haystack")


class TestHaystackOrchestrator(OrchestratorConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseOrchestratorConnector]:
        return HaystackOrchestrator

    @classmethod
    def backend_value(cls) -> str:
        return "haystack"

    @classmethod
    def sample_corpus(cls) -> list[dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "A cat is an independent and curious pet."},
            {"doc_id": "d2", "text": "Dogs are loyal and energetic companions."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat pet"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d1"

    @classmethod
    def expected_answer_substring(cls) -> str:
        return "curious pet"
