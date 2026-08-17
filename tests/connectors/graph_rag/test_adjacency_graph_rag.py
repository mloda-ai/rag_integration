"""Contract test for :class:`AdjacencyGraphRag` (zero-download CI anchor).

Inherits the whole graph_rag contract suite, including the graph not-a-stub
proof: a zero-overlap node connected to the relevant node must outscore an
equally non-overlapping isolated node, purely because of the edge. The fixture
is the same shape as the networkx backend's, so the proof holds for this
engine-free adjacency implementation too.
"""

from __future__ import annotations

from typing import Any

from rag_integration.feature_groups.connectors.graph_rag.adjacency_graph_rag import AdjacencyGraphRag
from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from tests.connectors.graph_rag.graph_rag_contract import GraphRagConnectorContractBase


class TestAdjacencyGraphRag(GraphRagConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseGraphRagConnector]:
        return AdjacencyGraphRag

    @classmethod
    def backend_value(cls) -> str:
        return "adjacency"

    @classmethod
    def sample_nodes(cls) -> list[dict[str, Any]]:
        return [
            # Relevant: shares "photosynthesis" and "plants" with the query.
            {"doc_id": "rel", "text": "Photosynthesis lets plants make energy from sunlight."},
            # Connected context: zero query overlap, but edged to "rel".
            {"doc_id": "ctx", "text": "It happens inside the chloroplast organelle."},
            # Isolated: zero query overlap and no edges.
            {"doc_id": "iso", "text": "The stock market fell sharply on Tuesday."},
        ]

    @classmethod
    def sample_edges(cls) -> list[list[str]]:
        return [["rel", "ctx"]]

    @classmethod
    def sample_query(cls) -> str:
        return "photosynthesis plants"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "rel"

    @classmethod
    def expected_connected_doc_id(cls) -> str:
        return "ctx"

    @classmethod
    def expected_isolated_doc_id(cls) -> str:
        return "iso"
