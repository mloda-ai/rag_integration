"""Tests for RetrievalEvaluator feature group."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from rag_integration.feature_groups.evaluation.retrieval_evaluator import RetrievalEvaluator

pytest.importorskip("numpy")


def _make_features(source_name: str = "eval_docs__embedded") -> Any:
    """Build a minimal FeatureSet mock."""
    feature = MagicMock()
    feature.get_name.return_value = f"{source_name}__evaluated"
    feature.options.get.return_value = None

    # FeatureChainParserMixin._extract_source_features reads feature_name
    feature.feature_name = MagicMock()
    feature.feature_name.name = f"{source_name}__evaluated"

    features = MagicMock()
    features.features = [feature]
    return features


def _embed(values: List[float]) -> List[float]:
    """Return a unit-normalised vector."""
    import numpy as np

    v = np.array(values, dtype=np.float32)
    return list(v / np.linalg.norm(v))


def _make_data(source: str = "eval_docs__embedded") -> List[Dict[str, Any]]:
    """Two corpus docs, two queries; query 0 matches corpus 0, query 1 matches corpus 1."""
    return [
        {"doc_id": "d0", "row_type": "corpus", source: _embed([1.0, 0.0, 0.0])},
        {"doc_id": "d1", "row_type": "corpus", source: _embed([0.0, 1.0, 0.0])},
        {"doc_id": "q0", "row_type": "query", source: _embed([1.0, 0.0, 0.0]), "relevant_doc_ids": ["d0"]},
        {"doc_id": "q1", "row_type": "query", source: _embed([0.0, 1.0, 0.0]), "relevant_doc_ids": ["d1"]},
    ]


class TestRetrievalEvaluator:
    def test_perfect_recall(self) -> None:
        source = "eval_docs__embedded"
        data = _make_data(source)
        features = _make_features(source)

        with patch.object(RetrievalEvaluator, "_extract_source_features", return_value=[source]):
            result = RetrievalEvaluator.calculate_feature(data, features)

        assert len(result) == 1
        assert result[0]["recall@1"] == pytest.approx(1.0)
        assert result[0]["recall@5"] == pytest.approx(1.0)
        assert result[0]["recall@10"] == pytest.approx(1.0)

    def test_zero_recall(self) -> None:
        source = "eval_docs__embedded"
        data = [
            {"doc_id": "d0", "row_type": "corpus", source: _embed([1.0, 0.0, 0.0])},
            {"doc_id": "q0", "row_type": "query", source: _embed([0.0, 1.0, 0.0]), "relevant_doc_ids": ["d_missing"]},
        ]
        features = _make_features(source)

        with patch.object(RetrievalEvaluator, "_extract_source_features", return_value=[source]):
            result = RetrievalEvaluator.calculate_feature(data, features)

        assert result[0]["recall@1"] == pytest.approx(0.0)

    def test_counts_returned(self) -> None:
        source = "eval_docs__embedded"
        data = _make_data(source)
        features = _make_features(source)

        with patch.object(RetrievalEvaluator, "_extract_source_features", return_value=[source]):
            result = RetrievalEvaluator.calculate_feature(data, features)

        assert result[0]["num_corpus"] == 2
        assert result[0]["num_queries"] == 2

    def test_empty_corpus(self) -> None:
        source = "eval_docs__embedded"
        data = [
            {"doc_id": "q0", "row_type": "query", source: _embed([1.0, 0.0]), "relevant_doc_ids": ["d0"]},
        ]
        features = _make_features(source)

        with patch.object(RetrievalEvaluator, "_extract_source_features", return_value=[source]):
            result = RetrievalEvaluator.calculate_feature(data, features)

        assert result[0]["recall@1"] == 0.0
