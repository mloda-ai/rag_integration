"""Contract test for :class:`HybridRrfRetriever` (RRF-fused lexical + dense).

The suite is inherited from :class:`RetrieveConnectorContractBase`. The corpus
is the dense backend's hash-embedder-friendly fixture (exact whitespace
tokens), on which the lexical and dense components agree on the top doc, so
the fused ranking has a determinate winner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from rag_integration.feature_groups.connectors.fusion import rrf_fuse
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.hybrid_rrf_retriever import HybridRrfRetriever
from tests.connectors.retrieve.retrieve_contract import RetrieveConnectorContractBase


class TestHybridRrfRetriever(RetrieveConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseRetrieveConnector]:
        return HybridRrfRetriever

    @classmethod
    def backend_value(cls) -> str:
        return "hybrid_rrf"

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

    # -- Hybrid-specific assertions -------------------------------------------

    def test_fused_ranking_equals_rrf_of_component_rankings(self) -> None:
        """The backend is exactly RRF over its components' full rankings."""
        corpus = self.sample_corpus()
        texts = [str(doc["text"]) for doc in corpus]
        query = self.sample_query()

        component_rankings = [
            [idx for idx, _ in component._rank(query, texts, len(texts))]
            for component in HybridRrfRetriever._COMPONENTS
        ]
        expected = rrf_fuse(component_rankings, top_k=len(texts))

        assert HybridRrfRetriever._rank(query, texts, len(texts)) == expected

    def test_one_component_empty_other_carries_the_ranking(self) -> None:
        """An all-stopwords corpus blanks the lexical component (empty BM25
        vocabulary); the fused result is the dense ranking alone and still
        honors only-positive scores and best-first order."""
        corpus = [
            {"doc_id": "s0", "text": "the and of"},
            {"doc_id": "s1", "text": "the of by"},
        ]
        texts = [str(doc["text"]) for doc in corpus]
        query = "the of by"

        lexical, dense = HybridRrfRetriever._COMPONENTS
        assert lexical._rank(query, texts, len(texts)) == []
        dense_indices = [idx for idx, _ in dense._rank(query, texts, len(texts))]
        assert dense_indices, "fixture must keep the dense component non-empty"

        fused = HybridRrfRetriever._rank(query, texts, len(texts))
        assert [idx for idx, _ in fused] == dense_indices
        scores = [score for _, score in fused]
        assert all(score > 0.0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_doc_found_by_one_component_still_surfaces(self) -> None:
        """Union semantics: a doc only one component scores positively is kept."""
        corpus = self.sample_corpus()
        texts = [str(doc["text"]) for doc in corpus]
        fused_indices = {idx for idx, _ in HybridRrfRetriever._rank(self.sample_query(), texts, len(texts))}

        union: set[int] = set()
        for component in HybridRrfRetriever._COMPONENTS:
            union.update(idx for idx, _ in component._rank(self.sample_query(), texts, len(texts)))

        assert fused_indices == union
