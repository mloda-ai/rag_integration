"""Inheritable contract-test suite for the ``graph_rag`` connector family.

Beyond the shared ranked-passage assertions, this suite adds the graph-specific
not-a-stub proof: a node with zero query-term overlap is surfaced because it
neighbours a relevant node, while an equally non-overlapping but *isolated* node
is not. A plain lexical retriever could not produce that result.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import pytest

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.rows import as_rows


class GraphRagConnectorContractBase(ABC):
    """Contract every graph-RAG backend must satisfy."""

    # -- Adapter methods ------------------------------------------------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> Type[BaseGraphRagConnector]:
        """Return the concrete ``BaseGraphRagConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``graph_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def sample_nodes(cls) -> List[Dict[str, Any]]:
        """Nodes (``{doc_id, text}``). Must include a relevant node, a zero-overlap
        node connected to it, and a zero-overlap isolated node."""

    @classmethod
    @abstractmethod
    def sample_edges(cls) -> List[List[str]]:
        """Edges (``[doc_id_a, doc_id_b]``) connecting the relevant and context nodes."""

    @classmethod
    @abstractmethod
    def sample_query(cls) -> str:
        """A query whose best match is determinate."""

    @classmethod
    @abstractmethod
    def expected_top_doc_id(cls) -> str:
        """The doc_id that must rank first."""

    @classmethod
    @abstractmethod
    def expected_connected_doc_id(cls) -> str:
        """A zero-overlap doc_id that must be retrieved via its edge to the top node."""

    @classmethod
    @abstractmethod
    def expected_isolated_doc_id(cls) -> str:
        """A zero-overlap doc_id with no edge that must NOT be retrieved at top_k=2."""

    # -- Helpers --------------------------------------------------------------

    @classmethod
    def _passages(
        cls, query: str, nodes: List[Dict[str, Any]], edges: List[List[str]], top_k: int
    ) -> List[Dict[str, Any]]:
        connector = cls.connector_class()
        edge_pairs = [(str(a), str(b)) for a, b in edges]
        return connector._retrieve(query, nodes, edge_pairs, top_k)

    @classmethod
    def _run_all(
        cls, query: str, nodes: List[Dict[str, Any]], edges: List[List[str]], top_k: int
    ) -> List[Dict[str, Any]]:
        connector = cls.connector_class()
        feature = Feature(
            connector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    connector.GRAPH_BACKEND: cls.backend_value(),
                    connector.QUERY_TEXT: query,
                    connector.NODES: nodes,
                    connector.EDGES: edges,
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
        opts = Options(context={connector.GRAPH_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is True

    def test_does_not_match_other_feature_name(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.GRAPH_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria("docs", opts) is False

    def test_unknown_backend_does_not_match(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.GRAPH_BACKEND: "definitely_not_a_backend_xyz"})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is False

    def test_backend_declared_in_supported_set(self) -> None:
        connector = self.connector_class()
        assert self.backend_value() in connector.GRAPH_BACKENDS

    # -- Output contract ------------------------------------------------------

    def test_returns_ranked_passage_shape(self) -> None:
        nodes = self.sample_nodes()
        passages = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        assert isinstance(passages, list)
        assert passages, "canonical query returned no passages; assertions would be vacuous"
        for passage in passages:
            assert set(passage) >= {"doc_id", "text", "score", "rank"}
            assert isinstance(passage["doc_id"], str)
            assert isinstance(passage["text"], str)
            assert isinstance(passage["score"], float)
            assert isinstance(passage["rank"], int)

    def test_scores_non_increasing_and_ranks_ascending(self) -> None:
        nodes = self.sample_nodes()
        passages = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        ranks = [p["rank"] for p in passages]
        scores = [p["score"] for p in passages]
        assert ranks == list(range(len(passages)))
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranked_first(self) -> None:
        nodes = self.sample_nodes()
        passages = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        assert passages[0]["doc_id"] == self.expected_top_doc_id()
        assert len(passages) >= 2
        assert passages[0]["score"] > passages[1]["score"]

    def test_connected_context_retrieved(self) -> None:
        """Graph not-a-stub proof: a zero-overlap node neighbouring the relevant
        node must *outscore* an equally non-overlapping isolated node, purely
        because of the edge. A lexical-only backend gives the two equal scores
        and fails this regardless of how it breaks ties.

        The score comparison is the load-bearing assertion (tie-break
        independent); the top_k=2 membership check then confirms the connected
        node is actually surfaced and the isolated one dropped. The proof is
        scoped to backends that score on query relevance (the family contract):
        the two zero-overlap nodes are only separable via the edge."""
        nodes = self.sample_nodes()
        full = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        score = {p["doc_id"]: p["score"] for p in full}
        assert score[self.expected_connected_doc_id()] > score[self.expected_isolated_doc_id()]

        top2 = [p["doc_id"] for p in self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=2)]
        assert self.expected_top_doc_id() in top2
        assert self.expected_connected_doc_id() in top2
        assert self.expected_isolated_doc_id() not in top2

    def test_passage_text_matches_nodes(self) -> None:
        nodes = self.sample_nodes()
        text_by_doc_id = {str(n["doc_id"]): str(n["text"]) for n in nodes}
        passages = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        for passage in passages:
            assert passage["text"] == text_by_doc_id[passage["doc_id"]]

    def test_doc_ids_unique_and_cover_nodes(self) -> None:
        nodes = self.sample_nodes()
        passages = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        returned = [p["doc_id"] for p in passages]
        assert len(returned) == len(set(returned))
        assert set(returned) == {str(n["doc_id"]) for n in nodes}

    def test_top_k_respected(self) -> None:
        passages = self._passages(self.sample_query(), self.sample_nodes(), self.sample_edges(), top_k=2)
        assert len(passages) == 2
        assert passages[0]["doc_id"] == self.expected_top_doc_id()

    def test_duplicate_doc_ids_rejected(self) -> None:
        """Duplicate doc_ids make edges ambiguous; the base must refuse them loudly
        instead of silently last-wins overwriting the doc_id -> index map."""
        nodes = self.sample_nodes()
        duplicated = nodes + [{"doc_id": nodes[0]["doc_id"], "text": "an unrelated duplicate"}]
        with pytest.raises(ValueError, match="duplicate doc_id"):
            self._passages(self.sample_query(), duplicated, self.sample_edges(), top_k=len(duplicated))

    def test_top_k_clamped_to_nodes(self) -> None:
        nodes = self.sample_nodes()
        passages = self._passages(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes) + 50)
        assert len(passages) == len(nodes)

    def test_empty_nodes_returns_empty(self) -> None:
        assert self._passages(self.sample_query(), [], [], top_k=5) == []

    def test_idempotent(self) -> None:
        nodes = self.sample_nodes()
        edges = self.sample_edges()
        first = self._passages(self.sample_query(), nodes, edges, top_k=len(nodes))
        second = self._passages(self.sample_query(), nodes, edges, top_k=len(nodes))
        assert first == second

    # -- End to end -----------------------------------------------------------

    def test_end_to_end_run_all(self) -> None:
        nodes = self.sample_nodes()
        passages = self._run_all(self.sample_query(), nodes, self.sample_edges(), top_k=len(nodes))
        assert passages
        assert passages[0]["doc_id"] == self.expected_top_doc_id()
