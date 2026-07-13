"""Inheritable contract-test suite for the ``rerank`` connector family.

Mirrors the ``retrieve`` family's contract suite: a concrete backend's test
implements five adapter methods and inherits every assertion. The base is not
named ``Test*`` so pytest does not collect it directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.rows import as_rows


class RerankConnectorContractBase(ABC):
    """Contract every rerank-connector backend must satisfy."""

    # -- Adapter methods (a concrete test implements these five) --------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> Type[BaseRerankConnector]:
        """Return the concrete ``BaseRerankConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``rerank_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def sample_candidates(cls) -> List[Dict[str, Any]]:
        """Return candidate passages (``{doc_id, text}``) with one determinate best match."""

    @classmethod
    @abstractmethod
    def sample_query(cls) -> str:
        """Return a query whose best match in ``sample_candidates`` is determinate."""

    @classmethod
    @abstractmethod
    def expected_top_doc_id(cls) -> str:
        """Return the ``doc_id`` that must rank first after reranking."""

    # -- Helpers --------------------------------------------------------------

    @classmethod
    def _rerank(cls, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return cls.connector_class()._rerank(query, candidates, top_k)

    @classmethod
    def _run_all(cls, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        connector = cls.connector_class()
        feature = Feature(
            connector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    connector.RERANK_BACKEND: cls.backend_value(),
                    connector.QUERY_TEXT: query,
                    connector.CANDIDATES: candidates,
                    connector.TOP_K: top_k,
                }
            ),
        )
        result = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        for partition in result:
            for row in as_rows(partition):
                if connector.ROOT_FEATURE_NAME in row:
                    passages: List[Dict[str, Any]] = row[connector.ROOT_FEATURE_NAME]
                    return passages
        raise AssertionError(f"run_all returned no '{connector.ROOT_FEATURE_NAME}' row: {result!r}")

    # -- Matching / honest surface --------------------------------------------

    def test_matches_root_feature_for_declared_backend(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.RERANK_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is True

    def test_does_not_match_other_feature_name(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.RERANK_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria("docs", opts) is False

    def test_unknown_backend_does_not_match(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.RERANK_BACKEND: "definitely_not_a_backend_xyz"})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is False

    def test_backend_declared_in_supported_set(self) -> None:
        connector = self.connector_class()
        assert self.backend_value() in connector.RERANK_BACKENDS

    # -- Output contract ------------------------------------------------------

    def test_returns_ranked_passage_shape(self) -> None:
        passages = self._rerank(self.sample_query(), self.sample_candidates(), top_k=len(self.sample_candidates()))
        assert isinstance(passages, list)
        assert passages, "canonical query returned no passages; contract assertions would be vacuous"
        for passage in passages:
            assert set(passage) >= {"doc_id", "text", "score", "rank"}
            assert isinstance(passage["doc_id"], str)
            assert isinstance(passage["text"], str)
            assert isinstance(passage["score"], float)
            assert isinstance(passage["rank"], int)

    def test_scores_non_increasing_and_ranks_ascending(self) -> None:
        passages = self._rerank(self.sample_query(), self.sample_candidates(), top_k=len(self.sample_candidates()))
        ranks = [p["rank"] for p in passages]
        scores = [p["score"] for p in passages]
        assert ranks == list(range(len(passages)))
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranked_first(self) -> None:
        """Not-a-stub proof: the crafted relevant doc must rank #1, by a margin."""
        passages = self._rerank(self.sample_query(), self.sample_candidates(), top_k=len(self.sample_candidates()))
        assert passages[0]["doc_id"] == self.expected_top_doc_id()
        assert len(passages) >= 2, "contract candidates must have >=2 docs to prove score separation"
        assert passages[0]["score"] > passages[1]["score"]

    def test_passage_text_matches_candidates(self) -> None:
        """Guards the base assembly: each passage's text is the candidate text for its doc_id."""
        candidates = self.sample_candidates()
        text_by_doc_id = {str(doc["doc_id"]): str(doc["text"]) for doc in candidates}
        passages = self._rerank(self.sample_query(), candidates, top_k=len(candidates))
        for passage in passages:
            assert passage["text"] == text_by_doc_id[passage["doc_id"]]

    def test_doc_ids_unique_and_cover_candidates(self) -> None:
        """No silent drop or duplicate: with top_k >= candidate count the returned
        doc_ids are unique and cover exactly the candidates."""
        candidates = self.sample_candidates()
        passages = self._rerank(self.sample_query(), candidates, top_k=len(candidates))
        returned = [p["doc_id"] for p in passages]
        assert len(returned) == len(set(returned)), "rerank returned duplicate doc_ids"
        assert set(returned) == {str(doc["doc_id"]) for doc in candidates}

    def test_top_k_respected(self) -> None:
        passages = self._rerank(self.sample_query(), self.sample_candidates(), top_k=1)
        assert len(passages) == 1
        assert passages[0]["doc_id"] == self.expected_top_doc_id()

    def test_top_k_clamped_to_candidates(self) -> None:
        candidates = self.sample_candidates()
        passages = self._rerank(self.sample_query(), candidates, top_k=len(candidates) + 50)
        assert len(passages) == len(candidates)

    def test_empty_candidates_returns_empty(self) -> None:
        assert self._rerank(self.sample_query(), [], top_k=5) == []

    def test_idempotent(self) -> None:
        candidates = self.sample_candidates()
        first = self._rerank(self.sample_query(), candidates, top_k=len(candidates))
        second = self._rerank(self.sample_query(), candidates, top_k=len(candidates))
        assert first == second

    # -- End to end -----------------------------------------------------------

    def test_end_to_end_run_all(self) -> None:
        candidates = self.sample_candidates()
        passages = self._run_all(self.sample_query(), candidates, top_k=len(candidates))
        assert passages, "run_all produced no passages"
        assert passages[0]["doc_id"] == self.expected_top_doc_id()
