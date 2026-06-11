"""Cross-cutting rank fusion shared by the connector families (issue #46).

Fusion mechanics are family-agnostic, so they live here next to ``mixins.py``
and ``errors.py`` rather than inside any single family: :func:`rrf_fuse`
operates on best-first rankings of hashable keys, where a key is whatever
identifies a result in the caller's space (corpus indices inside the
``retrieve`` family, ``doc_id`` strings when blending ranked passages across
families, e.g. ``retrieve`` + ``graph_rag``). A family exposes fusion through a
thin backend (see ``retrieve/hybrid_rrf_retriever.py``); the mechanics stay
shared here.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Sequence, Tuple, TypeVar

from rag_integration.feature_groups.connectors.errors import InvalidOptionError, RankingContractError

K = TypeVar("K", bound=Hashable)

# The standard RRF dampening constant from Cormack et al. (2009).
DEFAULT_RRF_K = 60


def rrf_fuse(rankings: Sequence[Sequence[K]], top_k: int, k: int = DEFAULT_RRF_K) -> List[Tuple[K, float]]:
    """Fuse best-first key rankings by reciprocal-rank fusion.

    ``score(key) = sum(1 / (k + position + 1))`` over every ranking that
    contains ``key`` (``position`` is 0-based), so consensus across rankings
    beats a single high placement. Returns at most ``top_k`` ``(key, score)``
    pairs, best-first. Scores are positive by construction and a key absent
    from every ranking never appears, which preserves the families'
    only-positive-scores rule. Ties break by first appearance (ranking order,
    then position), so fusion is deterministic. A duplicate key inside one
    ranking raises: it would silently double-count that result.
    """
    if k <= 0:
        raise InvalidOptionError(f"rrf_fuse requires a positive dampening constant k, got {k}.")
    if top_k <= 0:
        return []

    scores: Dict[K, float] = {}
    first_seen: Dict[K, int] = {}
    order = 0
    for ranking in rankings:
        seen_here: set[K] = set()
        for position, key in enumerate(ranking):
            if key in seen_here:
                raise RankingContractError(f"rrf_fuse got duplicate key {key!r} inside one ranking.")
            seen_here.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            if key not in first_seen:
                first_seen[key] = order
                order += 1

    fused = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return fused[:top_k]
