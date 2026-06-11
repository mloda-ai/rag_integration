"""Base-level edge handling and ranking edge cases for the ``graph_rag`` family.

Exercises the ``BaseGraphRagConnector`` invariants through the stdlib-only
:class:`AdjacencyGraphRag` (no extras required): ``_resolve_edges`` sanitising
(container guard, malformed elements, self-loops), duplicate doc_id rejection,
unknown doc_id edges, neighbour-bonus counting under duplicate/reversed edges,
and deterministic ordering on a zero-overlap query.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from rag_integration.feature_groups.connectors.graph_rag.adjacency_graph_rag import AdjacencyGraphRag

_NODES: List[Dict[str, Any]] = [
    {"doc_id": "rel", "text": "Photosynthesis lets plants make energy from sunlight."},
    {"doc_id": "ctx", "text": "It happens inside the chloroplast organelle."},
    {"doc_id": "iso", "text": "The stock market fell sharply on Tuesday."},
]

_QUERY = "photosynthesis plants"


def _resolve(raw_edges: Any) -> List[Tuple[str, str]]:
    return AdjacencyGraphRag._resolve_edges(raw_edges)


def _passages(edges: List[Tuple[str, str]], query: str = _QUERY) -> List[Dict[str, Any]]:
    return AdjacencyGraphRag._retrieve(query, _NODES, edges, len(_NODES))


# -- _resolve_edges: option extraction and sanitising --------------------------


def test_edges_omitted_resolve_to_empty() -> None:
    assert AdjacencyGraphRag._resolve_edges(None) == []


def test_non_sequence_edges_container_rejected() -> None:
    # A string EDGES previously dropped all edges silently; a non-iterable
    # crashed with a bare TypeError. Both must be a loud ValueError.
    with pytest.raises(ValueError, match="must be a list"):
        _resolve("rel,ctx")
    with pytest.raises(ValueError, match="must be a list"):
        _resolve(42)


def test_malformed_edge_elements_skipped() -> None:
    raw = [
        ["rel", "ctx", "extra"],  # length-3 list
        None,  # not a sequence
        {"a": "rel", "b": "ctx"},  # dict
        "ab",  # plain string element
        ["rel", "ctx"],  # the only real pair
    ]
    assert _resolve(raw) == [("rel", "ctx")]


def test_self_loop_edges_skipped() -> None:
    assert _resolve([["rel", "rel"]]) == []


# -- _retrieve: doc_id mapping ---------------------------------------------------


def test_unknown_doc_id_edges_skipped() -> None:
    with_ghost_edges = _passages([("rel", "ghost"), ("ghost", "ctx")])
    assert with_ghost_edges == _passages([])


def test_duplicate_doc_ids_rejected_even_after_str_coercion() -> None:
    nodes: List[Dict[str, Any]] = [{"doc_id": 1, "text": "first"}, {"doc_id": "1", "text": "second"}]
    with pytest.raises(ValueError, match="duplicate doc_id"):
        AdjacencyGraphRag._retrieve(_QUERY, nodes, [], len(nodes))


# -- Ranking edge cases ----------------------------------------------------------


def test_duplicate_and_reversed_edges_count_bonus_once() -> None:
    single = {p["doc_id"]: p["score"] for p in _passages([("rel", "ctx")])}
    repeated = {p["doc_id"]: p["score"] for p in _passages([("rel", "ctx"), ("ctx", "rel"), ("rel", "ctx")])}
    assert repeated == single
    assert repeated["ctx"] == pytest.approx(0.5)


def test_zero_overlap_query_returns_deterministic_index_order() -> None:
    passages = _passages([("rel", "ctx")], query="quantum entanglement")
    assert [p["doc_id"] for p in passages] == ["rel", "ctx", "iso"]
    assert all(p["score"] == 0.0 for p in passages)
