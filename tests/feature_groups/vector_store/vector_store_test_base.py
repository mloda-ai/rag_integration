"""Base test class for vector store feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Type

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.vector_store.base import BaseVectorStore


class VectorStoreTestBase(ABC):
    """Abstract base providing shared tests for all vector store implementations."""

    @property
    @abstractmethod
    def indexer_class(self) -> Type[BaseVectorStore]: ...

    @property
    @abstractmethod
    def test_embeddings(self) -> List[List[float]]: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def expected_type_name(self) -> str: ...

    def test_build_index(self) -> None:
        """Should build index with correct number of vectors."""
        index = self.indexer_class._build_index(self.test_embeddings, self.embedding_dim)
        assert index.ntotal == len(self.test_embeddings)

    def test_index_type_name(self) -> None:
        """Should return correct index type name."""
        assert self.indexer_class._index_type_name() == self.expected_type_name

    def test_feature_matching_pattern(self) -> None:
        """Should match indexed features and reject non-indexed."""
        assert self.indexer_class.match_feature_group_criteria("docs__embedded__indexed", Options())
        assert not self.indexer_class.match_feature_group_criteria("docs__embedded", Options())
