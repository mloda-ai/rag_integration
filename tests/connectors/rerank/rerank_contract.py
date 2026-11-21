"""Inheritable contract-test suite for the ``rerank`` connector family.

Mirrors the ``retrieve`` family's contract suite: a concrete backend's test
implements five adapter methods and inherits every assertion. The base is not
named ``Test*`` so pytest does not collect it directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pytest
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, PluginCollector, mloda, mlodaAPI
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import columnar_to_rows

from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from tests.connectors.retrieve.retrieve_contract import ScopedCorpusSource


class RerankConnectorContractBase(ABC):
    """Contract every rerank-connector backend must satisfy."""

    # -- Adapter methods (a concrete test implements these five) --------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> type[BaseRerankConnector]:
        """Return the concrete ``BaseRerankConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``rerank_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def sample_candidates(cls) -> list[dict[str, Any]]:
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
    def _rerank(cls, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        return cls.connector_class()._rerank(query, candidates, top_k)

    @classmethod
    def _run_all(cls, query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        connector = cls.connector_class()
        context = {
            connector.RERANK_BACKEND: cls.backend_value(),
            connector.QUERY_TEXT: query,
            connector.CANDIDATES: candidates,
            connector.TOP_K: top_k,
        }
        return cls._run_options(Options(context=context))

    @classmethod
    def _run_from_source(
        cls, candidates: list[dict[str, Any]], allowed_doc_ids: list[str], category: str
    ) -> list[dict[str, Any]]:
        """Run with the candidates taken from ``ScopedCorpusSource`` via ``candidates_source``."""
        connector = cls.connector_class()
        values: dict[str, Any] = {
            connector.RERANK_BACKEND: cls.backend_value(),
            connector.CANDIDATES_SOURCE: ScopedCorpusSource.ROOT_FEATURE_NAME,
            ScopedCorpusSource.DOCUMENTS: candidates,
            ScopedCorpusSource.ALLOWED_DOC_IDS: allowed_doc_ids,
        }
        runtime = {connector.QUERY_TEXT: cls.sample_query(), connector.TOP_K: len(candidates)}
        if category == "context":
            options = Options(context={**values, **runtime})
        else:
            options = Options(group=values, context=runtime)  # runtime keys are context only
        return cls._run_options(options, ScopedCorpusSource)

    @classmethod
    def _run_options(cls, options: Options, *upstream: type[FeatureGroup]) -> list[dict[str, Any]]:
        connector = cls.connector_class()
        result = mlodaAPI.run_all(
            [Feature(connector.ROOT_FEATURE_NAME, options=options)],
            compute_frameworks=[PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({connector, *upstream}),
        )
        return cls._passages(result)

    @classmethod
    def _passages(cls, result: Any) -> list[dict[str, Any]]:
        connector = cls.connector_class()
        for partition in result:
            for row in columnar_to_rows(partition):
                if connector.ROOT_FEATURE_NAME in row:
                    passages: list[dict[str, Any]] = row[connector.ROOT_FEATURE_NAME]
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

    @pytest.mark.parametrize("category", ["context", "group"])
    def test_end_to_end_candidates_source(self, category: str) -> None:
        """Candidates can come from an upstream feature; a doc it filters out never surfaces."""
        candidates = self.sample_candidates()
        doc_ids = [str(doc["doc_id"]) for doc in candidates]
        top = self.expected_top_doc_id()

        passages = self._run_from_source(candidates, doc_ids, category)
        assert passages == self._rerank(self.sample_query(), candidates, len(candidates))

        passages = self._run_from_source(candidates, [doc_id for doc_id in doc_ids if doc_id != top], category)
        assert passages
        assert top not in {p["doc_id"] for p in passages}

    def test_prepared_session_serves_a_query_per_run(self) -> None:
        connector = self.connector_class()
        candidates = self.sample_candidates()
        options = Options(context={connector.RERANK_BACKEND: self.backend_value(), connector.CANDIDATES: candidates})
        session = mloda.prepare(
            [Feature(connector.ROOT_FEATURE_NAME, options=options)],
            [PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        assert set(session.run_surface.parameters) == {connector.QUERY_TEXT, connector.TOP_K}
        answers = []
        for query in (self.sample_query(), "zzzz qqqq"):
            run = self._passages(session.run(parameters={connector.QUERY_TEXT: query, connector.TOP_K: 2}))
            assert run == self._run_all(query, candidates, top_k=2)
            answers.append(run)
        assert answers[0][0]["doc_id"] == self.expected_top_doc_id()
        assert answers[0] != answers[1]
        with pytest.raises(ValueError, match=connector.QUERY_TEXT):
            session.run()
        grouped = Options(group={**options.context, connector.QUERY_TEXT: self.sample_query()})
        with pytest.raises(ValueError, match=connector.QUERY_TEXT):
            mloda.prepare(
                [Feature(connector.ROOT_FEATURE_NAME, options=grouped)],
                [PythonDictFramework],
                plugin_collector=PluginCollector.enabled_feature_groups({connector}),
            )
