"""Unit tests for the cross-cutting RRF fusion facility (issue #46)."""

from __future__ import annotations

import pytest

from rag_integration.feature_groups.connectors.fusion import DEFAULT_RRF_K, rrf_fuse


def test_single_ranking_preserves_order_and_scores() -> None:
    fused = rrf_fuse([[7, 3, 5]], top_k=3)
    assert [key for key, _ in fused] == [7, 3, 5]
    assert [score for _, score in fused] == pytest.approx(
        [1 / (DEFAULT_RRF_K + 1), 1 / (DEFAULT_RRF_K + 2), 1 / (DEFAULT_RRF_K + 3)]
    )


def test_consensus_beats_single_champion() -> None:
    """A key both rankings place second outranks each ranking's own top key."""
    fused = rrf_fuse([["a", "c"], ["b", "c"]], top_k=3)
    assert [key for key, _ in fused] == ["c", "a", "b"]
    assert fused[0][1] == pytest.approx(2 / (DEFAULT_RRF_K + 2))


def test_scores_positive_and_non_increasing() -> None:
    fused = rrf_fuse([["a", "b"], ["b", "d"]], top_k=4)
    scores = [score for _, score in fused]
    assert all(score > 0.0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_ties_break_by_first_appearance() -> None:
    # "a" and "b" each hold one first place; "a" appears in an earlier ranking.
    fused = rrf_fuse([["a"], ["b"]], top_k=2)
    assert [key for key, _ in fused] == ["a", "b"]
    reversed_inputs = rrf_fuse([["b"], ["a"]], top_k=2)
    assert [key for key, _ in reversed_inputs] == ["b", "a"]


def test_top_k_cuts_fused_list() -> None:
    fused = rrf_fuse([["a", "b", "c"]], top_k=2)
    assert len(fused) == 2


def test_top_k_zero_or_negative_returns_empty() -> None:
    assert rrf_fuse([["a"]], top_k=0) == []
    assert rrf_fuse([["a"]], top_k=-1) == []


def test_empty_rankings_return_empty() -> None:
    assert rrf_fuse([], top_k=5) == []
    assert rrf_fuse([[], []], top_k=5) == []


def test_duplicate_key_in_one_ranking_raises() -> None:
    with pytest.raises(ValueError, match="duplicate key"):
        rrf_fuse([["a", "a"]], top_k=2)


def test_duplicate_across_rankings_is_consensus_not_error() -> None:
    fused = rrf_fuse([["a"], ["a"]], top_k=1)
    assert [key for key, _ in fused] == ["a"]
    assert fused[0][1] == pytest.approx(2 / (DEFAULT_RRF_K + 1))


def test_non_positive_k_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        rrf_fuse([["a"]], top_k=1, k=0)


def test_cross_family_doc_id_fusion() -> None:
    """The facility is family-agnostic: fuse retrieve and graph_rag passages by doc_id."""
    retrieved = [{"doc_id": "d2", "rank": 0}, {"doc_id": "d1", "rank": 1}]
    graph = [{"doc_id": "d3", "rank": 0}, {"doc_id": "d2", "rank": 1}]
    fused = rrf_fuse(
        [[str(p["doc_id"]) for p in retrieved], [str(p["doc_id"]) for p in graph]],
        top_k=3,
    )
    assert [key for key, _ in fused] == ["d2", "d3", "d1"]
