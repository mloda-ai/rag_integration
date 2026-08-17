"""Contract test for :class:`LexicalReranker` (zero-download CI anchor)."""

from __future__ import annotations

from typing import Any

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.connectors.rerank.lexical_reranker import LexicalReranker
from tests.connectors.rerank.rerank_contract import RerankConnectorContractBase


class TestLexicalReranker(RerankConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseRerankConnector]:
        return LexicalReranker

    @classmethod
    def backend_value(cls) -> str:
        return "lexical"

    @classmethod
    def sample_candidates(cls) -> list[dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "A complete guide to cat care: water, litter, and vet visits."},
            {"doc_id": "d2", "text": "Dogs are loyal and energetic companions."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat care guide"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d1"
