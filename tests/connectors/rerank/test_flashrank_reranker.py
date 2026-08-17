"""Contract test for :class:`FlashRankReranker` (pedigree backend).

Skipped on CI: the FlashRank model (~4 MB) downloads from the network on first
use, which is flaky on CI runners. Runs locally against the cached model. The
zero-download ``LexicalReranker`` is the always-on CI anchor for this family.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.connectors.rerank.flashrank_reranker import FlashRankReranker
from tests.conftest import requires_flashrank_model
from tests.connectors.rerank.rerank_contract import RerankConnectorContractBase

# Clean skip (not an error) when the `rerank` extra is not installed.
pytest.importorskip("flashrank")


@requires_flashrank_model
class TestFlashRankReranker(RerankConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseRerankConnector]:
        return FlashRankReranker

    @classmethod
    def backend_value(cls) -> str:
        return "flashrank"

    @classmethod
    def sample_candidates(cls) -> list[dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "Cats need fresh water, a clean litter box, and regular vet visits."},
            {"doc_id": "d2", "text": "Dogs are loyal and energetic companions."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "how to care for a cat"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d1"
