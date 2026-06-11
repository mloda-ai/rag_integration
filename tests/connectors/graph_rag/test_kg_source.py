"""Tests for the knowledge-graph source feature groups (issue #45).

Covers the ``TriplesKnowledgeGraph`` payload construction (node and edge
derivation from subject-predicate-object triples), the selector-gated matching
surface, and the standalone end-to-end run.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.graph_rag.kg_source import (
    BaseKnowledgeGraphSource,
    TriplesKnowledgeGraph,
)

_TRIPLES: List[List[str]] = [
    ["chloroplast", "hosts", "photosynthesis"],
    ["photosynthesis", "produces", "glucose"],
    ["mitochondria", "consumes", "glucose"],
]


def _build(triples: Any) -> Dict[str, Any]:
    options = Options(context={TriplesKnowledgeGraph.TRIPLES: triples})
    return TriplesKnowledgeGraph._build_graph(options)


# -- Matching / honest surface -------------------------------------------------


def test_matches_root_feature_for_declared_backend() -> None:
    opts = Options(context={TriplesKnowledgeGraph.KG_BACKEND: "triples"})
    assert TriplesKnowledgeGraph.match_feature_group_criteria("knowledge_graph", opts) is True


def test_does_not_match_other_feature_name() -> None:
    opts = Options(context={TriplesKnowledgeGraph.KG_BACKEND: "triples"})
    assert TriplesKnowledgeGraph.match_feature_group_criteria("graph_passages", opts) is False


def test_unknown_backend_does_not_match() -> None:
    opts = Options(context={TriplesKnowledgeGraph.KG_BACKEND: "definitely_not_a_backend_xyz"})
    assert TriplesKnowledgeGraph.match_feature_group_criteria("knowledge_graph", opts) is False


def test_base_never_matches() -> None:
    opts = Options(context={BaseKnowledgeGraphSource.KG_BACKEND: "triples"})
    assert BaseKnowledgeGraphSource.match_feature_group_criteria("knowledge_graph", opts) is False


# -- Payload construction --------------------------------------------------------


def test_nodes_cover_entities_in_first_appearance_order() -> None:
    graph = _build(_TRIPLES)
    doc_ids = [node["doc_id"] for node in graph["nodes"]]
    assert doc_ids == ["chloroplast", "photosynthesis", "glucose", "mitochondria"]


def test_node_text_carries_entity_and_its_triples() -> None:
    graph = _build(_TRIPLES)
    text_by_id = {node["doc_id"]: node["text"] for node in graph["nodes"]}
    assert text_by_id["photosynthesis"] == (
        "photosynthesis chloroplast hosts photosynthesis photosynthesis produces glucose"
    )
    assert text_by_id["mitochondria"] == "mitochondria mitochondria consumes glucose"


def test_edges_follow_triples() -> None:
    graph = _build(_TRIPLES)
    assert graph["edges"] == [
        ["chloroplast", "photosynthesis"],
        ["photosynthesis", "glucose"],
        ["mitochondria", "glucose"],
    ]


def test_repeated_pairs_emitted_once() -> None:
    graph = _build([["a", "likes", "b"], ["a", "knows", "b"]])
    assert graph["edges"] == [["a", "b"]]


def test_self_loop_triples_yield_no_edge() -> None:
    graph = _build([["a", "is", "a"]])
    assert graph["edges"] == []
    assert [node["doc_id"] for node in graph["nodes"]] == ["a"]
    # The self-loop sentence appears once, not once per entity slot.
    assert graph["nodes"][0]["text"] == "a a is a"


def test_repeated_triple_contributes_its_sentence_once() -> None:
    graph = _build([["a", "likes", "b"], ["a", "likes", "b"]])
    text_by_id = {node["doc_id"]: node["text"] for node in graph["nodes"]}
    assert text_by_id["a"] == "a a likes b"
    assert text_by_id["b"] == "b a likes b"


def test_empty_triples_yield_empty_graph() -> None:
    graph = _build([])
    assert graph == {"nodes": [], "edges": []}


# -- Validation -------------------------------------------------------------------


def test_missing_triples_raises() -> None:
    with pytest.raises(ValueError, match=TriplesKnowledgeGraph.TRIPLES):
        TriplesKnowledgeGraph._build_graph(Options(context={}))


def test_non_list_triples_rejected() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        _build("a,b,c")


def test_malformed_triple_rejected() -> None:
    with pytest.raises(ValueError, match="index 1"):
        _build([["a", "b", "c"], ["a", "b"]])


# -- End to end --------------------------------------------------------------------


def test_end_to_end_run_all() -> None:
    feature = Feature(
        TriplesKnowledgeGraph.ROOT_FEATURE_NAME,
        options=Options(
            context={
                TriplesKnowledgeGraph.KG_BACKEND: "triples",
                TriplesKnowledgeGraph.TRIPLES: _TRIPLES,
            }
        ),
    )
    result = mlodaAPI.run_all(
        [feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups({TriplesKnowledgeGraph}),
    )
    for partition in result:
        for row in partition:
            if TriplesKnowledgeGraph.ROOT_FEATURE_NAME in row:
                graph = row[TriplesKnowledgeGraph.ROOT_FEATURE_NAME]
                assert [node["doc_id"] for node in graph["nodes"]] == [
                    "chloroplast",
                    "photosynthesis",
                    "glucose",
                    "mitochondria",
                ]
                assert len(graph["edges"]) == 3
                return
    raise AssertionError(f"run_all returned no knowledge_graph row: {result!r}")
