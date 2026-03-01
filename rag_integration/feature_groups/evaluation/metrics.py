"""Retrieval evaluation metrics.

Pure Python implementation — no external dependencies required.
"""

from __future__ import annotations

from typing import Dict, List, Set


def recall_at_k(relevant_ids: Set[str], ranked_ids: List[str], k: int) -> float:
    """Compute Recall@K for a single query.

    Args:
        relevant_ids: Set of ground-truth relevant document IDs.
        ranked_ids:   List of retrieved document IDs ordered by descending score.
        k:            Cut-off rank.

    Returns:
        Fraction of relevant items found in the top-K results.
        Returns 0.0 when ``relevant_ids`` is empty.

    Example::

        recall_at_k({"a", "b"}, ["a", "c", "d", "b"], k=3)
        # → 0.5  (only "a" found in top-3, "b" is at rank 4)
    """
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(relevant_ids & top_k) / len(relevant_ids)


def mean_recall_at_k(
    query_relevant: Dict[str, Set[str]],
    query_ranked: Dict[str, List[str]],
    k: int,
) -> float:
    """Compute mean Recall@K across all queries.

    Args:
        query_relevant: Mapping from query_id to set of relevant doc IDs.
        query_ranked:   Mapping from query_id to ranked list of doc IDs.
        k:              Cut-off rank.

    Returns:
        Mean Recall@K across all queries.
    """
    if not query_relevant:
        return 0.0
    scores = [recall_at_k(query_relevant[qid], query_ranked.get(qid, []), k) for qid in query_relevant]
    return sum(scores) / len(scores)
