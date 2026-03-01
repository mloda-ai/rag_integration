"""Tests for FaissIVFIndexer."""

from __future__ import annotations

import numpy as np

from rag_integration.feature_groups.rag_pipeline.vector_store import FaissIVFIndexer


class TestFaissIVFIndexer:
    """Tests for FaissIVFIndexer."""

    def test_build_index(self) -> None:
        """Should build an IndexIVFFlat with correct number of vectors."""
        embeddings = [[float(i == j) for j in range(8)] for i in range(20)]
        index = FaissIVFIndexer._build_index(embeddings, 8)
        assert index.ntotal == 20

    def test_nlist_clamping(self) -> None:
        """nlist should be clamped to not exceed vector count."""
        # Only 3 vectors, nlist default=10 should clamp to 3
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        index = FaissIVFIndexer._build_index(embeddings, 2)
        assert index.ntotal == 3

    def test_search(self) -> None:
        """Should find approximate nearest neighbors."""
        embeddings = [[float(i == j) for j in range(8)] for i in range(8)]
        index = FaissIVFIndexer._build_index(embeddings, 8)

        query = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = index.search(query, 1)

        assert indices[0][0] == 0

    def test_index_type_name(self) -> None:
        """Should return correct index type name."""
        assert FaissIVFIndexer._index_type_name() == "ivf_flat"
