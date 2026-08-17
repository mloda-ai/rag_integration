"""Contract test for :class:`TfidfRetriever` (zero-download CI anchor).

The whole suite is inherited from :class:`RetrieveConnectorContractBase`; this
class only wires up the five adapter methods. The corpus is crafted so the
TF-IDF vectorizer separates the answer: the query shares both distinctive terms
``cat``/``pet`` only with ``d2`` and just ``pet`` with ``d1`` (the embedder
drops tokens of length <= 2, so short stop-ish words do not muddy the vectors),
so ``d2`` ranks first and ``d1`` is a positively scoring runner-up (the family
drops zero-scored passages, so the score-margin assertion needs one); the
distractors share none.
"""

from __future__ import annotations

from typing import Any

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.tfidf_retriever import TfidfRetriever
from tests.connectors.retrieve.retrieve_contract import RetrieveConnectorContractBase


class TestTfidfRetriever(RetrieveConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseRetrieveConnector]:
        return TfidfRetriever

    @classmethod
    def backend_value(cls) -> str:
        return "tfidf"

    @classmethod
    def sample_corpus(cls) -> list[dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "The mat lay flat on the floor by the window."},
            {"doc_id": "d1", "text": "A dog can be a loyal and energetic pet."},
            {"doc_id": "d2", "text": "A cat is an independent and curious pet."},
            {"doc_id": "d3", "text": "Cars need regular engine oil and maintenance."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat pet"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d2"
