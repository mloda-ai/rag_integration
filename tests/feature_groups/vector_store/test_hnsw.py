"""Tests for FaissHNSWIndexer."""

from __future__ import annotations

from typing import List, Type

import numpy as np

from rag_integration.feature_groups.rag_pipeline.vector_store import FaissHNSWIndexer
from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore
from tests.feature_groups.vector_store.vector_store_test_base import VectorStoreTestBase


class TestFaissHNSWIndexer(VectorStoreTestBase):
    """Tests for FaissHNSWIndexer."""

    @property
    def indexer_class(self) -> Type[BaseVectorStore]:
        return FaissHNSWIndexer

    @property
    def test_embeddings(self) -> List[List[float]]:
        return [[float(i == j) for j in range(8)] for i in range(8)]

    @property
    def embedding_dim(self) -> int:
        return 8

    @property
    def expected_type_name(self) -> str:
        return "hnsw_flat"

    def test_search_shape(self) -> None:
        """Search results should have correct shape."""
        index = FaissHNSWIndexer._build_index(self.test_embeddings, 8)
        query = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        distances, indices = index.search(query, 3)
        assert distances.shape == (1, 3)
        assert indices.shape == (1, 3)
