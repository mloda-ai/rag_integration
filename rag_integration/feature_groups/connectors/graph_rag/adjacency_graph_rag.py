"""Adjacency-map graph-RAG backend (no networkx).

Second concrete for the ``graph_rag`` family: same query-overlap +
neighbour-bonus scoring as :class:`NetworkxGraphRag`, but over a hand-built
adjacency map walked with the standard library instead of networkx. Zero
dependency (pure Python stdlib), zero-download, deterministic. It proves the
family contract is not tied to one graph library: swap the engine, keep the
behaviour.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Weight added to a node for each relevant (query-overlapping) neighbour.
# Matches NetworkxGraphRag so the two backends rank identically.
_NEIGHBOUR_BONUS = 0.5


class AdjacencyGraphRag(BaseGraphRagConnector):
    """Graph-expansion retrieval over a plain adjacency map (``graph_backend="adjacency"``).

    ``score(node) = lexical_overlap(node) + 0.5 * (relevant neighbours)``, where
    a relevant neighbour is a directly-connected node with non-zero query
    overlap. The adjacency map is built by walking the resolved edge list (each
    edge wired both ways, since the graph is undirected). Ties are broken by
    node index, so the ranking is stable and deterministic. This is the same
    scoring as :class:`NetworkxGraphRag`; only the graph engine differs.
    """

    GRAPH_BACKENDS = {
        "adjacency": "Graph-expansion retrieval over a hand-built adjacency map (no networkx)",
    }

    PROPERTY_MAPPING = {
        BaseGraphRagConnector.GRAPH_BACKEND: property_spec(
            "Use 'adjacency' for graph-expansion retrieval (no networkx)", context=False
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
        # Build an undirected adjacency map from the resolved (index, index)
        # edges; the base has already dropped self-loops and unknown ids.
        adjacency: dict[int, set[int]] = {node: set() for node in range(len(texts))}
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)

        query_tokens = cls._tokenize(query)
        overlap = [len(query_tokens & cls._tokenize(text)) for text in texts]
        seeds = {i for i, count in enumerate(overlap) if count > 0}

        scored: List[Tuple[int, float]] = []
        for node in range(len(texts)):
            relevant_neighbours = sum(1 for neighbour in adjacency[node] if neighbour in seeds)
            score = float(overlap[node]) + _NEIGHBOUR_BONUS * relevant_neighbours
            scored.append((node, score))

        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
