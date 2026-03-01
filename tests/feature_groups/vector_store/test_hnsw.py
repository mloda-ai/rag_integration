"""Tests for FaissHNSWIndexer."""

from __future__ import annotations

import numpy as np

from rag_integration.feature_groups.rag_pipeline.vector_store import FaissHNSWIndexer


class TestFaissHNSWIndexer:
    """Tests for FaissHNSWIndexer."""

    def test_build_index(self) -> None:
        """Should build an IndexHNSWFlat with correct number of vectors."""
        embeddings = [[float(i == j) for j in range(8)] for i in range(8)]
        index = FaissHNSWIndexer._build_index(embeddings, 8)
        assert index.ntotal == 8

    def test_search_shape(self) -> None:
        """Search results should have correct shape."""
        embeddings = [[float(i == j) for j in range(8)] for i in range(8)]
        index = FaissHNSWIndexer._build_index(embeddings, 8)

        query = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = index.search(query, 3)

        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)

    def test_index_type_name(self) -> None:
        """Should return correct index type name."""
        assert FaissHNSWIndexer._index_type_name() == "hnsw_flat"
