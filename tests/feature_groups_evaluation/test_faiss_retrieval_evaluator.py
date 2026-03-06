"""Tests for FaissRetrievalEvaluator feature group."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from rag_integration.feature_groups.evaluation.faiss_retrieval_evaluator import FaissRetrievalEvaluator

pytest.importorskip("numpy")
pytest.importorskip("faiss")

# The embedding feature is one level above __indexed
_EMBEDDING_FEATURE = "eval_docs__chunked__deduped__embedded"
_INDEXED_FEATURE = f"{_EMBEDDING_FEATURE}__indexed"
_FEATURE_NAME = f"{_INDEXED_FEATURE}__evaluated"


def _make_features(indexed_feature: str = _INDEXED_FEATURE) -> Any:
    """Build a minimal FeatureSet mock for FaissRetrievalEvaluator."""
    feature = MagicMock()
    feature.get_name.return_value = f"{indexed_feature}__evaluated"
    feature.options.get.return_value = None

    feature.feature_name = MagicMock()
    feature.feature_name.name = f"{indexed_feature}__evaluated"

    features = MagicMock()
    features.features = [feature]
    return features


def _embed(values: List[float]) -> List[float]:
    """Return a unit-normalised vector."""
    import numpy as np

    v = np.array(values, dtype=np.float32)
    return list(v / np.linalg.norm(v))


def _make_data(emb_feature: str = _EMBEDDING_FEATURE) -> List[Dict[str, Any]]:
    """Two corpus docs, two queries; query i matches corpus i exactly."""
    return [
        {"doc_id": "d0", "row_type": "corpus", emb_feature: _embed([1.0, 0.0, 0.0]), _INDEXED_FEATURE: 0},
        {"doc_id": "d1", "row_type": "corpus", emb_feature: _embed([0.0, 1.0, 0.0]), _INDEXED_FEATURE: 1},
        {
            "doc_id": "q0",
            "row_type": "query",
            emb_feature: _embed([1.0, 0.0, 0.0]),
            _INDEXED_FEATURE: 2,
            "relevant_doc_ids": ["d0"],
        },
        {
            "doc_id": "q1",
            "row_type": "query",
            emb_feature: _embed([0.0, 1.0, 0.0]),
            _INDEXED_FEATURE: 3,
            "relevant_doc_ids": ["d1"],
        },
    ]


class TestFaissRetrievalEvaluator:
    def test_perfect_recall(self) -> None:
        """Identical query/corpus embeddings → FAISS ranks exact match first → Recall@1 = 1.0."""
        data = _make_data()
        features = _make_features()

        with patch.object(FaissRetrievalEvaluator, "_extract_source_features", return_value=[_INDEXED_FEATURE]):
            result = FaissRetrievalEvaluator.calculate_feature(data, features)

        assert len(result) == 1
        assert result[0]["recall@1"] == pytest.approx(1.0)
        assert result[0]["recall@5"] == pytest.approx(1.0)
        assert result[0]["recall@10"] == pytest.approx(1.0)
        assert result[0]["num_queries"] == 2
        assert result[0]["num_corpus"] == 2

    def test_zero_recall(self) -> None:
        """Orthogonal query embeddings never match → Recall@1 = 0.0."""
        emb = _EMBEDDING_FEATURE
        data = [
            {"doc_id": "d0", "row_type": "corpus", emb: _embed([1.0, 0.0, 0.0]), _INDEXED_FEATURE: 0},
            {"doc_id": "d1", "row_type": "corpus", emb: _embed([0.0, 1.0, 0.0]), _INDEXED_FEATURE: 1},
            # Query points to d0 but its embedding is orthogonal to d0
            {
                "doc_id": "q0",
                "row_type": "query",
                emb: _embed([0.0, 1.0, 0.0]),  # matches d1, not d0
                _INDEXED_FEATURE: 2,
                "relevant_doc_ids": ["d0"],
            },
        ]
        features = _make_features()

        with patch.object(FaissRetrievalEvaluator, "_extract_source_features", return_value=[_INDEXED_FEATURE]):
            result = FaissRetrievalEvaluator.calculate_feature(data, features)

        assert result[0]["recall@1"] == pytest.approx(0.0)

    def test_chunked_doc_recall(self) -> None:
        """Corpus doc split into 2 chunks; query matches chunk 1 → doc-level Recall@1 = 1.0."""
        emb = _EMBEDDING_FEATURE
        # d0 has two chunks, both share doc_id="d0"
        data = [
            {
                "doc_id": "d0",
                "chunk_id": "d0_chunk_0",
                "row_type": "corpus",
                emb: _embed([1.0, 0.0, 0.0]),
                _INDEXED_FEATURE: 0,
            },
            {
                "doc_id": "d0",
                "chunk_id": "d0_chunk_1",
                "row_type": "corpus",
                emb: _embed([0.9, 0.1, 0.0]),
                _INDEXED_FEATURE: 1,
            },
            {
                "doc_id": "d1",
                "chunk_id": "d1_chunk_0",
                "row_type": "corpus",
                emb: _embed([0.0, 1.0, 0.0]),
                _INDEXED_FEATURE: 2,
            },
            {
                "doc_id": "q0",
                "row_type": "query",
                emb: _embed([0.9, 0.1, 0.0]),  # closest to d0_chunk_1
                _INDEXED_FEATURE: 3,
                "relevant_doc_ids": ["d0"],
            },
        ]
        features = _make_features()

        with patch.object(FaissRetrievalEvaluator, "_extract_source_features", return_value=[_INDEXED_FEATURE]):
            result = FaissRetrievalEvaluator.calculate_feature(data, features)

        # d0_chunk_1 is the top result; its doc_id="d0" matches relevant_doc_ids → hit
        assert result[0]["recall@1"] == pytest.approx(1.0)

    def test_empty_corpus_returns_zero_metrics(self) -> None:
        """No corpus rows → returns zeros gracefully."""
        emb = _EMBEDDING_FEATURE
        data = [
            {
                "doc_id": "q0",
                "row_type": "query",
                emb: _embed([1.0, 0.0, 0.0]),
                _INDEXED_FEATURE: 0,
                "relevant_doc_ids": ["d0"],
            }
        ]
        features = _make_features()

        with patch.object(FaissRetrievalEvaluator, "_extract_source_features", return_value=[_INDEXED_FEATURE]):
            result = FaissRetrievalEvaluator.calculate_feature(data, features)

        assert result[0]["recall@1"] == pytest.approx(0.0)
        assert result[0]["num_corpus"] == 0
