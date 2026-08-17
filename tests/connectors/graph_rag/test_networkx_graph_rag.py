"""Contract test for :class:`NetworkxGraphRag` (canonical networkx backend).

Skipped cleanly when the ``graph`` extra (networkx) is not installed; the
stdlib-only :class:`AdjacencyGraphRag` test is the family's zero-download CI
anchor.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.connectors.graph_rag.networkx_graph_rag import NetworkxGraphRag
from tests.connectors.graph_rag.graph_rag_contract import GraphRagConnectorContractBase

# Clean skip (not an error) when the `graph` extra is not installed.
pytest.importorskip("networkx")


class TestNetworkxGraphRag(GraphRagConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseGraphRagConnector]:
        return NetworkxGraphRag

    @classmethod
    def backend_value(cls) -> str:
        return "networkx"

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
