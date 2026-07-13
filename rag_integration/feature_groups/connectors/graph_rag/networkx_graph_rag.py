"""NetworkX graph-RAG backend.

Canonical concrete for the ``graph_rag`` family: zero-download, deterministic,
backed by networkx (BSD, pure-Python, no model). Scores each node by its own
query-term overlap plus a bonus for neighbouring relevant nodes, so passages
that are connected to the answer are surfaced even when they share no query
term. This is the distinguishing value of graph RAG over plain retrieval.
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Weight added to a node for each relevant (query-overlapping) neighbour.
_NEIGHBOUR_BONUS = 0.5


class NetworkxGraphRag(BaseGraphRagConnector):
    """Graph-expansion retrieval over networkx (``graph_backend="networkx"``).

    ``score(node) = lexical_overlap(node) + 0.5 * (relevant neighbours)``, where
    a relevant neighbour is one with non-zero query overlap. Ties are broken by
    node index, so the ranking is stable and deterministic.
    """

    GRAPH_BACKENDS = {
        "networkx": "Graph-expansion retrieval over networkx",
    }

    PROPERTY_MAPPING = {
        BaseGraphRagConnector.GRAPH_BACKEND: property_spec(
            "Use 'networkx' for graph-expansion retrieval", context=False
        ),
        BaseGraphRagConnector.QUERY_TEXT: property_spec("Raw text query to search the graph", context=False),
        BaseGraphRagConnector.TOP_K: property_spec(
            f"Number of passages to return (default {BaseGraphRagConnector.DEFAULT_TOP_K})", context=False
        ),
        BaseGraphRagConnector.NODES: property_spec("Graph nodes: a list of {doc_id, text} dicts", context=False),
        BaseGraphRagConnector.EDGES: property_spec(
            "Graph edges: a list of [doc_id_a, doc_id_b] pairs."
            " Optional: omitting it degrades scoring to lexical-only (no neighbour bonus)",
            context=False,
        ),
    }

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_RE.findall(text.lower()))

    @classmethod
    def _rank(cls, query: str, texts: List[str], edges: List[Tuple[int, int]], top_k: int) -> List[Tuple[int, float]]:
        import networkx as nx

        graph: Any = nx.Graph()
        graph.add_nodes_from(range(len(texts)))
        graph.add_edges_from(edges)

        query_tokens = cls._tokenize(query)
        overlap = [len(query_tokens & cls._tokenize(text)) for text in texts]
        seeds = {i for i, count in enumerate(overlap) if count > 0}

        scored: List[Tuple[int, float]] = []
        for node in range(len(texts)):
            relevant_neighbours = sum(1 for nb in graph.neighbors(node) if nb in seeds)
            score = float(overlap[node]) + _NEIGHBOUR_BONUS * relevant_neighbours
            scored.append((node, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
