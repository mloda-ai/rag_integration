"""Integration tests for the evaluation pipeline.

Tests the full end-to-end evaluation pipeline via the mloda feature chain:

    dataset source → embedder → RetrievalEvaluator (→ Recall@K)

Uses in-memory fixture data — no file system access, no network, no large datasets.
The same mlodaAPI.run_all() pattern used in the RAG/image pipeline integration tests.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mloda.user import mlodaAPI, PluginCollector, Feature, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.datasets.text.base import BaseTextDatasetSource
from rag_integration.feature_groups.datasets.image.base import BaseImageDatasetSource
from rag_integration.feature_groups.evaluation.retrieval_evaluator import RetrievalEvaluator
from rag_integration.feature_groups.rag_pipeline.embedding.mock import MockEmbedder
from rag_integration.feature_groups.image_pipeline.embedding.mock import MockImageEmbedder

pytest.importorskip("numpy")


# =============================================================================
# In-memory fixture dataset sources (no file system / network access needed)
# =============================================================================


class FixtureTextDatasetSource(BaseTextDatasetSource):
    """Tiny in-memory text corpus + queries for integration testing.

    Query texts are identical to their target corpus texts so that
    the deterministic MockEmbedder assigns the same vector → Recall@1 = 1.0.
    """

    @classmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "antigen regulates protein expression", "row_type": "corpus"},
            {"doc_id": "d1", "text": "cells divide through mitosis", "row_type": "corpus"},
            {
                "doc_id": "q0",
                "text": "antigen regulates protein expression",
                "row_type": "query",
                "relevant_doc_ids": ["d0"],
            },
            {
                "doc_id": "q1",
                "text": "cells divide through mitosis",
                "row_type": "query",
                "relevant_doc_ids": ["d1"],
            },
        ]


class FixtureImageDatasetSource(BaseImageDatasetSource):
    """Tiny in-memory image corpus + caption queries for integration testing."""

    @classmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        img0 = b"fixture_image_data_corpus_0" * 8  # deterministic non-empty bytes
        img1 = b"fixture_image_data_corpus_1" * 8
        return [
            {"image_id": "img0", "image_data": img0, "format": "jpeg", "row_type": "corpus"},
            {"image_id": "img1", "image_data": img1, "format": "jpeg", "row_type": "corpus"},
            {
                "image_id": "img0_cap0",
                "image_data": None,
                "caption": "A photograph of the first subject",
                "row_type": "query",
                "relevant_image_ids": ["img0"],
            },
            {
                "image_id": "img1_cap0",
                "image_data": None,
                "caption": "A photograph of the second subject",
                "row_type": "query",
                "relevant_image_ids": ["img1"],
            },
        ]


# =============================================================================
# Helpers
# =============================================================================


def flatten_result(raw: Any) -> List[Dict[str, Any]]:
    """Unwrap nested mlodaAPI result to a flat list of dicts."""
    if raw and isinstance(raw[0], list):
        return raw[0]
    return list(raw)


# =============================================================================
# Text evaluation pipeline
# =============================================================================


_TEXT_FEATURE = "eval_docs__embedded__evaluated"
_IMAGE_FEATURE = "eval_images__embedded__evaluated"


def _get_metrics(raw_result: Any, feature_name: str) -> Dict[str, Any]:
    """Unwrap the mloda result and extract the metrics dict stored under feature_name."""
    rows = flatten_result(raw_result[0])
    assert len(rows) == 1, "RetrievalEvaluator should return exactly one aggregate row"
    row = rows[0]
    assert feature_name in row, f"Expected '{feature_name}' key in result row, got keys: {list(row.keys())}"
    metrics: Dict[str, Any] = row[feature_name]
    return metrics


class TestTextEvaluationPipeline:
    """
    Full mloda chain: eval_docs → eval_docs__embedded → eval_docs__embedded__evaluated

    FixtureTextDatasetSource  →  MockEmbedder  →  RetrievalEvaluator
    """

    def test_pipeline_produces_recall_metrics(self) -> None:
        """Pipeline runs end-to-end and returns Recall@1/5/10 plus corpus/query counts."""
        feature = Feature(_TEXT_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureTextDatasetSource, MockEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = _get_metrics(raw_result, _TEXT_FEATURE)
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "num_corpus" in metrics
        assert "num_queries" in metrics
        assert metrics["num_corpus"] == 2
        assert metrics["num_queries"] == 2

    def test_recall_values_are_floats_in_unit_interval(self) -> None:
        """Recall@K values are valid floats in [0.0, 1.0]."""
        feature = Feature(_TEXT_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureTextDatasetSource, MockEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = _get_metrics(raw_result, _TEXT_FEATURE)
        for key in ("recall@1", "recall@5", "recall@10"):
            assert isinstance(metrics[key], float), f"{key} should be a float"
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} outside [0, 1]"

    def test_perfect_recall_when_query_matches_corpus(self) -> None:
        """
        MockEmbedder is deterministic: identical text → identical unit vector.
        Both queries use the exact same text as their target corpus document,
        so cosine similarity is 1.0 for the relevant doc → Recall@1 = 1.0.
        """
        feature = Feature(_TEXT_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureTextDatasetSource, MockEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = _get_metrics(raw_result, _TEXT_FEATURE)
        assert metrics["recall@1"] == pytest.approx(1.0), (
            "Recall@1 should be 1.0 when query text == corpus text (deterministic mock embedder)"
        )
        assert metrics["recall@10"] == pytest.approx(1.0)


# =============================================================================
# Image evaluation pipeline
# =============================================================================


class TestImageEvaluationPipeline:
    """
    Full mloda chain: eval_images → eval_images__embedded → eval_images__embedded__evaluated

    FixtureImageDatasetSource  →  MockImageEmbedder  →  RetrievalEvaluator
    """

    def test_pipeline_produces_recall_metrics(self) -> None:
        """Pipeline runs end-to-end and returns Recall@1/5/10 plus corpus/query counts."""
        feature = Feature(_IMAGE_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureImageDatasetSource, MockImageEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = _get_metrics(raw_result, _IMAGE_FEATURE)
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert metrics["num_corpus"] == 2
        assert metrics["num_queries"] == 2

    def test_recall_values_are_floats_in_unit_interval(self) -> None:
        """Recall@K values are valid floats in [0.0, 1.0]."""
        feature = Feature(_IMAGE_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureImageDatasetSource, MockImageEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = _get_metrics(raw_result, _IMAGE_FEATURE)
        for key in ("recall@1", "recall@5", "recall@10"):
            assert isinstance(metrics[key], float), f"{key} should be a float"
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} outside [0, 1]"

    def test_recall_at_k_covers_full_corpus(self) -> None:
        """
        With 2 corpus images and k >= 2, every query's relevant image is
        reachable → Recall@2 = 1.0 (and Recall@5 / Recall@10 the same).
        """
        feature = Feature(_IMAGE_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureImageDatasetSource, MockImageEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = _get_metrics(raw_result, _IMAGE_FEATURE)
        # k=5 and k=10 both exceed corpus size (2) → all relevant docs reachable
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)
