"""Base test class for text embedding feature groups."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Type

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder


class TextEmbeddingTestBase(ABC):
    """Abstract base providing shared tests for all text embedding implementations."""

    @property
    @abstractmethod
    def embedder_class(self) -> Type[BaseEmbedder]: ...

    @property
    @abstractmethod
    def sample_texts(self) -> List[str]: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        embeddings = self.embedder_class._embed_texts([self.sample_texts[0]], self.embedding_dim, self.model_name)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == self.embedding_dim

    def test_deterministic_embeddings(self) -> None:
        """Same text should produce same embedding."""
        texts = [self.sample_texts[0]]
        emb1 = self.embedder_class._embed_texts(texts, self.embedding_dim, self.model_name)
        emb2 = self.embedder_class._embed_texts(texts, self.embedding_dim, self.model_name)
        assert emb1[0] == emb2[0]

    def test_different_texts_different_embeddings(self) -> None:
        """Different texts should produce different embeddings."""
        embeddings = self.embedder_class._embed_texts(self.sample_texts[:2], self.embedding_dim, self.model_name)
        assert embeddings[0] != embeddings[1]

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        embeddings = self.embedder_class._embed_texts([self.sample_texts[0]], self.embedding_dim, self.model_name)
        magnitude = math.sqrt(sum(x * x for x in embeddings[0]))
        assert abs(magnitude - 1.0) < 0.001

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features and reject non-embedded."""
        assert self.embedder_class.match_feature_group_criteria("docs__deduped__embedded", Options())
        assert not self.embedder_class.match_feature_group_criteria("docs__deduped", Options())
