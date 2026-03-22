"""Tests for FaissFlatIndexer."""

from __future__ import annotations

from typing import List, Type

import numpy as np

from rag_integration.feature_groups.rag_pipeline.vector_store import FaissFlatIndexer
from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore
from tests.feature_groups.vector_store.vector_store_test_base import VectorStoreTestBase


class TestFaissFlatIndexer(VectorStoreTestBase):
    """Tests for FaissFlatIndexer."""

    @property
    def indexer_class(self) -> Type[BaseVectorStore]:
        return FaissFlatIndexer

    @property
    def test_embeddings(self) -> List[List[float]]:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    @property
    def embedding_dim(self) -> int:
        return 3

    @property
    def expected_type_name(self) -> str:
        return "flat_l2"

    def test_search(self) -> None:
        """Should find the nearest neighbor correctly."""
        index = FaissFlatIndexer._build_index(self.test_embeddings, 3)
        query = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = index.search(query, 1)
        assert indices[0][0] == 0
        assert distances[0][0] < 1e-6
