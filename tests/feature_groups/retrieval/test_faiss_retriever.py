"""Tests for FaissRetriever."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.retrieval import FaissRetriever


class TestFaissRetriever:
    """Tests for FaissRetriever."""

    def _create_test_index(self, tmp_path: Path, dimension: int = 4, n_vectors: int = 10) -> tuple[str, str]:
        """Create a test FAISS index with metadata on disk."""
        index = faiss.IndexFlatL2(dimension)
        vectors = np.random.default_rng(42).random((n_vectors, dimension)).astype(np.float32)
        index.add(vectors)

        index_path = str(tmp_path / "test_index.faiss")
        faiss.write_index(index, index_path)

        texts = [f"Document {i}" for i in range(n_vectors)]
        doc_ids = [f"doc_{i}" for i in range(n_vectors)]
        metadata = {"texts": texts, "doc_ids": doc_ids, "num_vectors": n_vectors}

        metadata_path = str(tmp_path / "test_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Reset cache for clean test
        FaissRetriever._cached_index = None
        FaissRetriever._cached_index_path = None
        FaissRetriever._cached_metadata = None
        FaissRetriever._cached_metadata_path = None

        return index_path, metadata_path

    def test_pre_embedded_query(self, tmp_path: Path) -> None:
        """Should return top-k results with pre-embedded query vector."""
        index_path, metadata_path = self._create_test_index(tmp_path, dimension=4, n_vectors=10)

        query_vector = [0.5, 0.5, 0.5, 0.5]
        options = Options(
            {
                "index_path": index_path,
                "metadata_path": metadata_path,
                "query_embedding": query_vector,
                "top_k": 3,
            }
        )

        result = FaissRetriever._search(query_vector, 3, options)

        assert len(result["indices"]) == 3
        assert len(result["distances"]) == 3
        assert len(result["texts"]) == 3
        assert len(result["doc_ids"]) == 3

        # All distances should be non-negative
        for d in result["distances"]:
            assert d >= 0

        # All texts should be non-empty
        for t in result["texts"]:
            assert len(t) > 0

    def test_no_metadata_mode(self, tmp_path: Path) -> None:
        """Should return results without texts/doc_ids when no metadata_path."""
        index = faiss.IndexFlatL2(4)
        vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        index.add(vectors)

        index_path = str(tmp_path / "no_meta_index.faiss")
        faiss.write_index(index, index_path)

        FaissRetriever._cached_index = None
        FaissRetriever._cached_index_path = None

        options = Options({"index_path": index_path, "top_k": 2})
        result = FaissRetriever._search([1.0, 0.0, 0.0, 0.0], 2, options)

        assert len(result["indices"]) == 2
        assert len(result["distances"]) == 2
        assert result["texts"] == []
        assert result["doc_ids"] == []

    def test_top_k_clamping(self, tmp_path: Path) -> None:
        """top_k should be clamped to index size."""
        index_path, metadata_path = self._create_test_index(tmp_path, dimension=4, n_vectors=3)

        options = Options(
            {
                "index_path": index_path,
                "metadata_path": metadata_path,
                "top_k": 100,
            }
        )

        result = FaissRetriever._search([0.5, 0.5, 0.5, 0.5], 100, options)

        # Should only return 3 results (index size)
        assert len(result["indices"]) == 3

    def test_feature_matching(self) -> None:
        """Should match 'retrieved' feature name."""
        assert FaissRetriever.match_feature_group_criteria("retrieved", Options())
        assert not FaissRetriever.match_feature_group_criteria("docs", Options())
        assert not FaissRetriever.match_feature_group_criteria("docs__indexed", Options())
