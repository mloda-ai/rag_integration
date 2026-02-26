"""Tests for FaissFlatIndexer."""

from __future__ import annotations

import numpy as np

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.vector_store import FaissFlatIndexer


class TestFaissFlatIndexer:
    """Tests for FaissFlatIndexer."""

    def test_build_index(self) -> None:
        """Should build an IndexFlatL2 with correct number of vectors."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        index = FaissFlatIndexer._build_index(embeddings, 3)
        assert index.ntotal == 3

    def test_search(self) -> None:
        """Should find the nearest neighbor correctly."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        index = FaissFlatIndexer._build_index(embeddings, 3)

        query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = index.search(query, 1)

        assert indices[0][0] == 0
        assert distances[0][0] < 1e-6

    def test_index_type_name(self) -> None:
        """Should return correct index type name."""
        assert FaissFlatIndexer._index_type_name() == "flat_l2"

    def test_feature_matching_pattern(self) -> None:
        """Should match __indexed features."""
        assert FaissFlatIndexer.match_feature_group_criteria("docs__embedded__indexed", Options())
        assert not FaissFlatIndexer.match_feature_group_criteria("docs__embedded", Options())
