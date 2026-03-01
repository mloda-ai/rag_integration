"""Unit tests for retrieval evaluation metrics."""

from __future__ import annotations

import pytest

from rag_integration.feature_groups.evaluation.metrics import recall_at_k, mean_recall_at_k


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k({"a", "b"}, ["a", "b", "c"], k=2) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k({"a", "b"}, ["a", "c", "d", "b"], k=3) == 0.5

    def test_zero_recall(self) -> None:
        assert recall_at_k({"a", "b"}, ["c", "d", "e"], k=3) == 0.0

    def test_empty_relevant(self) -> None:
        assert recall_at_k(set(), ["a", "b"], k=2) == 0.0

    def test_k_larger_than_ranked(self) -> None:
        # k=10 but only 3 ranked — should still work
        assert recall_at_k({"a"}, ["a", "b", "c"], k=10) == 1.0

    def test_recall_at_1(self) -> None:
        assert recall_at_k({"a"}, ["a", "b"], k=1) == 1.0
        assert recall_at_k({"a"}, ["b", "a"], k=1) == 0.0


class TestMeanRecallAtK:
    def test_mean_over_two_queries(self) -> None:
        query_relevant = {
            "q1": {"a"},
            "q2": {"b"},
        }
        query_ranked = {
            "q1": ["a", "c"],  # recall@1 = 1.0
            "q2": ["c", "b"],  # recall@1 = 0.0
        }
        result = mean_recall_at_k(query_relevant, query_ranked, k=1)
        assert result == pytest.approx(0.5)

    def test_empty_queries(self) -> None:
        assert mean_recall_at_k({}, {}, k=5) == 0.0

    def test_missing_ranked_for_query(self) -> None:
        query_relevant = {"q1": {"a"}}
        result = mean_recall_at_k(query_relevant, {}, k=5)
        assert result == 0.0
