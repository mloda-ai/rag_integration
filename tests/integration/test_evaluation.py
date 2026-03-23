"""
Integration tests for evaluation pipelines.

Tests three evaluation scenarios:
1. Text: eval_docs -> embedded -> evaluated (cosine similarity)
2. Image: eval_images -> embedded -> evaluated (cosine similarity)
3. FAISS: eval_docs -> chunked -> deduped -> embedded -> indexed -> evaluated (FAISS search)

Uses in-memory fixture data. No file system access, no network, no large datasets.
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
from tests.integration.helpers import get_metrics

pytest.importorskip("numpy")


# =============================================================================
# Fixture dataset sources
# =============================================================================


class FixtureTextDatasetSource(BaseTextDatasetSource):
    """Tiny in-memory text corpus + queries.

    Query texts are identical to their target corpus texts so that
    the deterministic MockEmbedder assigns the same vector, giving Recall@1 = 1.0.
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
    """Tiny in-memory image corpus + caption queries."""

    @classmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        img0 = b"fixture_image_data_corpus_0" * 8
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
# Text evaluation pipeline
# =============================================================================

_TEXT_FEATURE = "eval_docs__embedded__evaluated"
_IMAGE_FEATURE = "eval_images__embedded__evaluated"


class TestTextEvaluationPipeline:
    """eval_docs -> eval_docs__embedded -> eval_docs__embedded__evaluated"""

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

        metrics = get_metrics(raw_result, _TEXT_FEATURE)
        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
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

        metrics = get_metrics(raw_result, _TEXT_FEATURE)
        for key in ("recall@1", "recall@5", "recall@10"):
            assert isinstance(metrics[key], float), f"{key} should be a float"
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} outside [0, 1]"

    def test_perfect_recall_when_query_matches_corpus(self) -> None:
        """Identical text gives identical MockEmbedder vector, so Recall@1 = 1.0."""
        feature = Feature(_TEXT_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureTextDatasetSource, MockEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = get_metrics(raw_result, _TEXT_FEATURE)
        assert metrics["recall@1"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)


# =============================================================================
# Image evaluation pipeline
# =============================================================================


class TestImageEvaluationPipeline:
    """eval_images -> eval_images__embedded -> eval_images__embedded__evaluated"""

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

        metrics = get_metrics(raw_result, _IMAGE_FEATURE)
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

        metrics = get_metrics(raw_result, _IMAGE_FEATURE)
        for key in ("recall@1", "recall@5", "recall@10"):
            assert isinstance(metrics[key], float), f"{key} should be a float"
            assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} outside [0, 1]"

    def test_recall_at_k_covers_full_corpus(self) -> None:
        """With 2 corpus images and k >= 2, every query's relevant image is reachable."""
        feature = Feature(_IMAGE_FEATURE)

        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {FixtureImageDatasetSource, MockImageEmbedder, RetrievalEvaluator}
            ),
        )

        metrics = get_metrics(raw_result, _IMAGE_FEATURE)
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)


# =============================================================================
# FAISS evaluation pipeline (full ingestion chain)
# =============================================================================

_FAISS_FEATURE = "eval_docs__chunked__deduped__embedded__indexed__evaluated"


class FixtureFaissTextDatasetSource(BaseTextDatasetSource):
    """Tiny in-memory corpus + queries for FAISS pipeline testing.

    Query texts are identical to their target corpus texts so MockEmbedder
    assigns the same deterministic vector. FAISS ranks the exact match first,
    giving Recall@1 = 1.0.

    ``keep_strategy="all_unique"`` must be passed in Options so the
    ExactHashDeduplicator does not remove query rows that share text with
    corpus rows.
    """

    @classmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "antigen regulates protein expression levels", "row_type": "corpus"},
            {"doc_id": "d1", "text": "neural networks learn from gradient descent", "row_type": "corpus"},
            {
                "doc_id": "q0",
                "text": "antigen regulates protein expression levels",
                "row_type": "query",
                "relevant_doc_ids": ["d0"],
            },
            {
                "doc_id": "q1",
                "text": "neural networks learn from gradient descent",
                "row_type": "query",
                "relevant_doc_ids": ["d1"],
            },
        ]


class TestFaissEvaluationPipeline:
    """End-to-end: full ingestion pipeline through FAISS evaluation."""

    def _run(self, options: Dict[str, Any]) -> Dict[str, Any]:
        from rag_integration.feature_groups.evaluation.faiss_retrieval_evaluator import FaissRetrievalEvaluator
        from rag_integration.feature_groups.rag_pipeline.chunking.fixed_size import FixedSizeChunker
        from rag_integration.feature_groups.rag_pipeline.deduplication.exact_hash import ExactHashDeduplicator
        from rag_integration.feature_groups.rag_pipeline.vector_store.faiss_flat import FaissFlatIndexer

        feature = Feature(_FAISS_FEATURE, options=Options(options))
        raw_result = mlodaAPI.run_all(
            features=[feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups(
                {
                    FixtureFaissTextDatasetSource,
                    FixedSizeChunker,
                    ExactHashDeduplicator,
                    MockEmbedder,
                    FaissFlatIndexer,
                    FaissRetrievalEvaluator,
                }
            ),
        )
        return get_metrics(raw_result, _FAISS_FEATURE)

    @pytest.mark.skipif(not pytest.importorskip("faiss", reason="faiss required"), reason="faiss required")
    def test_perfect_recall_through_full_pipeline(self) -> None:
        """Identical query/corpus texts with same MockEmbedding gives FAISS Recall@1 = 1.0."""
        metrics = self._run(
            {
                "chunking_method": "fixed_size",
                "deduplication_method": "exact_hash",
                "keep_strategy": "all_unique",
                "embedding_method": "mock",
                "index_method": "flat",
            }
        )

        assert metrics["recall@1"] == pytest.approx(1.0)
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)

    @pytest.mark.skipif(not pytest.importorskip("faiss", reason="faiss required"), reason="faiss required")
    def test_metrics_shape(self) -> None:
        """Result row contains all expected metric keys."""
        metrics = self._run(
            {
                "chunking_method": "fixed_size",
                "deduplication_method": "exact_hash",
                "keep_strategy": "all_unique",
                "embedding_method": "mock",
                "index_method": "flat",
            }
        )

        assert "recall@1" in metrics
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "num_corpus" in metrics
        assert "num_queries" in metrics
        assert metrics["num_queries"] == 2
        assert metrics["num_corpus"] >= 2
