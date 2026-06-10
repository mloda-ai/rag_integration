"""Contract test for :class:`FaissDenseRetriever` (canonical dense backend).

The suite is inherited from :class:`RetrieveConnectorContractBase`. The corpus
uses exact whitespace tokens (the hash embedder does not strip punctuation):
the query shares two tokens with ``d2``, one with ``d1``, none with the
distractors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.faiss_retriever import FaissDenseRetriever
from tests.connectors.retrieve.retrieve_contract import RetrieveConnectorContractBase


class TestFaissDenseRetriever(RetrieveConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseRetrieveConnector]:
        return FaissDenseRetriever

    @classmethod
    def backend_value(cls) -> str:
        return "faiss"

    @classmethod
    def sample_corpus(cls) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "the mat lay flat on the floor by the window"},
            {"doc_id": "d1", "text": "a dog can be a loyal and energetic pet"},
            {"doc_id": "d2", "text": "a cat is an independent and curious pet"},
            {"doc_id": "d3", "text": "cars need regular engine oil and maintenance"},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat pet"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d2"
