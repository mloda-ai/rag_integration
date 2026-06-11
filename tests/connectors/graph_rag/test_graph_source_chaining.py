"""graph_rag consuming a knowledge-graph source as its corpus (issue #45).

End to end: a ``graph_passages`` feature with ``graph_source="knowledge_graph"``
declares the KG source feature as its input, the engine chains the two, and the
ranking runs over the KG-derived nodes and edges, no inline ``nodes``/``edges``
involved. The inline path stays the default and is untouched (covered by the
existing contract suite).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from mloda.provider import DefaultOptionKeys
from mloda.user import mlodaAPI, Feature, FeatureName, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.graph_rag.adjacency_graph_rag import AdjacencyGraphRag
from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.connectors.graph_rag.kg_source import TriplesKnowledgeGraph
from rag_integration.feature_groups.connectors.graph_rag.networkx_graph_rag import NetworkxGraphRag

# Connected entities carry their triple sentences, so they overlap the query
# via the relations; the disconnected stock_market branch shares no token with
# it (the tokenizer splits "stock_market" into "stock" and "market").
_TRIPLES: List[List[str]] = [
    ["chloroplast", "hosts", "photosynthesis"],
    ["photosynthesis", "produces", "glucose"],
    ["stock_market", "fell_on", "tuesday"],
]

_QUERY = "photosynthesis"


def _chained_context(backend: str = "adjacency", extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        AdjacencyGraphRag.GRAPH_BACKEND: backend,
        AdjacencyGraphRag.GRAPH_SOURCE: TriplesKnowledgeGraph.ROOT_FEATURE_NAME,
        AdjacencyGraphRag.QUERY_TEXT: _QUERY,
        TriplesKnowledgeGraph.KG_BACKEND: "triples",
        TriplesKnowledgeGraph.TRIPLES: _TRIPLES,
    }
    if extra:
        context.update(extra)
    return context


def _chained_options(extra: Dict[str, Any] | None = None) -> Options:
    return Options(context=_chained_context(extra=extra))


def _run_chained(options: Options, connector: type[BaseGraphRagConnector] = AdjacencyGraphRag) -> List[Dict[str, Any]]:
    feature = Feature(connector.ROOT_FEATURE_NAME, options=options)
    result = mlodaAPI.run_all(
        [feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups({connector, TriplesKnowledgeGraph}),
    )
    for partition in result:
        for row in partition:
            if connector.ROOT_FEATURE_NAME in row:
                passages: List[Dict[str, Any]] = row[connector.ROOT_FEATURE_NAME]
                return passages
    raise AssertionError(f"run_all returned no graph_passages row: {result!r}")


def _feature_set(options: Options) -> Any:
    feature = MagicMock()
    feature.options = options
    features = MagicMock()
    features.features = [feature]
    return features


# -- input_features declaration ---------------------------------------------------


def test_no_graph_source_stays_root() -> None:
    options = Options(context={AdjacencyGraphRag.GRAPH_BACKEND: "adjacency"})
    assert AdjacencyGraphRag().input_features(options, FeatureName(AdjacencyGraphRag.ROOT_FEATURE_NAME)) is None


def test_graph_source_declares_input_feature_with_forwarded_context() -> None:
    inputs = AdjacencyGraphRag().input_features(_chained_options(), FeatureName(AdjacencyGraphRag.ROOT_FEATURE_NAME))
    assert inputs is not None
    (feature,) = inputs
    assert str(feature.name) == TriplesKnowledgeGraph.ROOT_FEATURE_NAME
    # The source's own options are forwarded; the family's keys are not.
    assert feature.options.get(TriplesKnowledgeGraph.KG_BACKEND) == "triples"
    assert feature.options.get(TriplesKnowledgeGraph.TRIPLES) == _TRIPLES
    assert feature.options.get(AdjacencyGraphRag.GRAPH_BACKEND) is None
    assert feature.options.get(AdjacencyGraphRag.GRAPH_SOURCE) is None
    assert feature.options.get(AdjacencyGraphRag.QUERY_TEXT) is None
    # The family keys are merge-protected, so the engine's group-option merge
    # cannot re-add query-specific keys to the source feature.
    assert feature.options.get(DefaultOptionKeys.feature_chainer_parser_key) == AdjacencyGraphRag.FAMILY_OPTION_KEYS


def test_graph_source_forwards_group_options_without_family_keys() -> None:
    # Group-style caller: Options(dict) puts everything into group options.
    inputs = AdjacencyGraphRag().input_features(
        Options(_chained_context()), FeatureName(AdjacencyGraphRag.ROOT_FEATURE_NAME)
    )
    assert inputs is not None
    (feature,) = inputs
    assert feature.options.group.get(TriplesKnowledgeGraph.KG_BACKEND) == "triples"
    for family_key in AdjacencyGraphRag.FAMILY_OPTION_KEYS:
        assert family_key not in feature.options.group


# -- End to end ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("connector", "backend"),
    [(AdjacencyGraphRag, "adjacency"), (NetworkxGraphRag, "networkx")],
    ids=["adjacency", "networkx"],
)
def test_chained_ranking_runs_over_kg_nodes(connector: type[BaseGraphRagConnector], backend: str) -> None:
    passages = _run_chained(Options(context=_chained_context(backend=backend)), connector=connector)
    by_id = {p["doc_id"]: p["score"] for p in passages}

    # The query entity wins; its text carries two matching triple sentences.
    assert passages[0]["doc_id"] == "photosynthesis"
    # KG neighbours get the one-hop bonus; the disconnected branch scores zero.
    assert by_id["chloroplast"] > by_id["stock_market"]
    assert by_id["glucose"] > by_id["stock_market"]
    assert by_id["stock_market"] == 0.0


def test_chained_top_k_applies() -> None:
    passages = _run_chained(_chained_options(extra={AdjacencyGraphRag.TOP_K: 2}))
    assert len(passages) == 2
    assert passages[0]["doc_id"] == "photosynthesis"


# -- Validation -----------------------------------------------------------------------


def test_graph_source_with_inline_nodes_raises() -> None:
    options = _chained_options(extra={AdjacencyGraphRag.NODES: [{"doc_id": "n", "text": "t"}]})
    with pytest.raises(ValueError, match="one graph only"):
        AdjacencyGraphRag.calculate_feature([], _feature_set(options))


def test_graph_source_with_inline_edges_raises() -> None:
    options = _chained_options(extra={AdjacencyGraphRag.EDGES: [["a", "b"]]})
    with pytest.raises(ValueError, match="one graph only"):
        AdjacencyGraphRag.calculate_feature([], _feature_set(options))


def test_graph_source_without_upstream_row_raises() -> None:
    with pytest.raises(ValueError, match="produced no row"):
        AdjacencyGraphRag.calculate_feature([], _feature_set(_chained_options()))


def test_graph_source_with_malformed_payload_raises() -> None:
    data = [{TriplesKnowledgeGraph.ROOT_FEATURE_NAME: ["not", "a", "dict"]}]
    with pytest.raises(ValueError, match="nodes"):
        AdjacencyGraphRag.calculate_feature(data, _feature_set(_chained_options()))
