"""Tests for FaissIVFIndexer."""

from __future__ import annotations

from typing import List, Type

import numpy as np

from rag_integration.feature_groups.rag_pipeline.vector_store import FaissIVFIndexer
from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore
from tests.feature_groups.vector_store.vector_store_test_base import VectorStoreTestBase


class TestFaissIVFIndexer(VectorStoreTestBase):
    """Tests for FaissIVFIndexer."""

    @property
    def indexer_class(self) -> Type[BaseVectorStore]:
        return FaissIVFIndexer

    @property
    def test_embeddings(self) -> List[List[float]]:
        return [[float(i == j) for j in range(8)] for i in range(20)]

    @property
    def embedding_dim(self) -> int:
        return 8

    @property
    def expected_type_name(self) -> str:
        return "ivf_flat"

    def test_nlist_clamping(self) -> None:
        """nlist should be clamped to not exceed vector count."""
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
