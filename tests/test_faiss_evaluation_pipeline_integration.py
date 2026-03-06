"""Integration test: full pipeline evaluation with FAISS.

Tests the complete end-to-end chain through every stage of the main pipeline:

    dataset source → chunker → deduplicator → embedder → FAISS indexer → evaluator

    eval_docs__chunked__deduped__embedded__indexed__evaluated

This verifies that:
1. All pipeline stages are wired correctly via mloda feature chaining.
2. FaissFlatIndexer (Tom's code, unmodified) builds a valid FAISS index.
3. FaissRetrievalEvaluator computes correct Recall@K through FAISS search.
4. doc-level recall is measured properly even for chunked corpora
   (original doc_id is preserved on chunk rows by the chunker).

Uses in-memory fixture data — no file system access, no network, no large datasets.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from mloda.user import mlodaAPI, PluginCollector, Feature, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.datasets.text.base import BaseTextDatasetSource
from rag_integration.feature_groups.evaluation.faiss_retrieval_evaluator import FaissRetrievalEvaluator
from rag_integration.feature_groups.rag_pipeline.chunking.fixed_size import FixedSizeChunker
from rag_integration.feature_groups.rag_pipeline.deduplication.exact_hash import ExactHashDeduplicator
from rag_integration.feature_groups.rag_pipeline.embedding.mock import MockEmbedder
from rag_integration.feature_groups.rag_pipeline.vector_store.faiss_flat import FaissFlatIndexer

pytest.importorskip("numpy")
pytest.importorskip("faiss")

# =============================================================================
# Feature chain terminal name
# =============================================================================

_FEATURE = "eval_docs__chunked__deduped__embedded__indexed__evaluated"


# =============================================================================
# In-memory fixture dataset source
# =============================================================================


class FixtureFaissTextDatasetSource(BaseTextDatasetSource):
    """Tiny in-memory corpus + queries for FAISS pipeline testing.

    Query texts are identical to their target corpus texts so MockEmbedder
    assigns the same deterministic vector → FAISS ranks the exact match first
    → Recall@1 = 1.0.

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


# =============================================================================
# Helper
# =============================================================================


def flatten_result(raw: Any) -> List[Dict[str, Any]]:
    """Unwrap mlodaAPI result to a flat list of row dicts."""
    if raw and isinstance(raw[0], list):
        return raw[0]
    return list(raw)


def _get_metrics(raw_result: Any, feature_name: str) -> Dict[str, Any]:
    rows = flatten_result(raw_result[0])
    assert len(rows) == 1, f"Expected 1 aggregate row, got {len(rows)}"
    row = rows[0]
    assert feature_name in row, f"Feature '{feature_name}' not in result row keys: {list(row.keys())}"
    metrics: Dict[str, Any] = row[feature_name]
    return metrics


# =============================================================================
# Tests
# =============================================================================


class TestFaissEvaluationPipeline:
    """End-to-end: full ingestion pipeline → FAISS evaluation."""

    def _run(self, options: Dict[str, Any]) -> Dict[str, Any]:
        feature = Feature(_FEATURE, options=Options(options))
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
        return _get_metrics(raw_result, _FEATURE)

    def test_perfect_recall_through_full_pipeline(self) -> None:
        """Identical query/corpus texts → same MockEmbedding → FAISS Recall@1 = 1.0."""
        metrics = self._run(
            {
                "chunking_method": "fixed_size",
                "deduplication_method": "exact_hash",
                "keep_strategy": "all_unique",  # preserve query rows that share text with corpus
                "embedding_method": "mock",
                "index_method": "flat",
            }
        )

        assert metrics["recall@1"] == pytest.approx(1.0), f"Expected Recall@1=1.0, got {metrics}"
        assert metrics["recall@5"] == pytest.approx(1.0)
        assert metrics["recall@10"] == pytest.approx(1.0)

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
        # Corpus has 2 docs; each sentence fits in a single fixed-size chunk
        assert metrics["num_corpus"] >= 2
