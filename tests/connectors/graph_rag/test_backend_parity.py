"""Backend parity for the ``graph_rag`` family.

:class:`NetworkxGraphRag` and :class:`AdjacencyGraphRag` document identical
scoring (query overlap + neighbour bonus, index tie-break) but implement it
independently; nothing in the code enforces the parity beyond copy-discipline.
This suite pins it: both backends must return identical passage lists for the
same inputs, so a drift in either implementation fails loudly here.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_integration.feature_groups.connectors.graph_rag.adjacency_graph_rag import AdjacencyGraphRag
from rag_integration.feature_groups.connectors.graph_rag.networkx_graph_rag import NetworkxGraphRag

# Clean skip (not an error) when the `graph` extra is not installed.
pytest.importorskip("networkx")

_NODES: list[dict[str, Any]] = [
    {"doc_id": "rel", "text": "Photosynthesis lets plants make energy from sunlight."},
    {"doc_id": "ctx", "text": "It happens inside the chloroplast organelle."},
    {"doc_id": "alt", "text": "Plants also respire and grow at night."},
    {"doc_id": "iso", "text": "The stock market fell sharply on Tuesday."},
]

_QUERY = "photosynthesis plants"


def _assert_parity(query: str, nodes: list[dict[str, Any]], edges: list[tuple[str, str]], top_k: int) -> None:
    networkx_passages = NetworkxGraphRag._retrieve(query, nodes, edges, top_k)
    adjacency_passages = AdjacencyGraphRag._retrieve(query, nodes, edges, top_k)
    assert networkx_passages == adjacency_passages
    assert networkx_passages, "parity assertion would be vacuous on an empty result"


def test_parity_connected_graph() -> None:
    _assert_parity(_QUERY, _NODES, [("rel", "ctx"), ("ctx", "alt"), ("alt", "iso")], top_k=len(_NODES))


def test_parity_duplicate_and_reversed_edges() -> None:
    _assert_parity(_QUERY, _NODES, [("rel", "ctx"), ("rel", "ctx"), ("ctx", "rel")], top_k=len(_NODES))


def test_parity_zero_overlap_query() -> None:
    _assert_parity("quantum entanglement", _NODES, [("rel", "ctx"), ("ctx", "alt")], top_k=len(_NODES))


def test_parity_isolated_nodes() -> None:
    _assert_parity(_QUERY, _NODES, [("rel", "ctx")], top_k=len(_NODES))


def test_parity_empty_edges() -> None:
    _assert_parity(_QUERY, _NODES, [], top_k=len(_NODES))
