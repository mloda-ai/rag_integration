"""Tests for HashEmbedder."""

from typing import List, Type

from rag_integration.feature_groups.rag_pipeline.embedding import HashEmbedder
from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder
from tests.feature_groups.embedding.text_embedding_test_base import TextEmbeddingTestBase


class TestHashEmbedder(TextEmbeddingTestBase):
    """Tests for HashEmbedder."""

    @property
    def embedder_class(self) -> Type[BaseEmbedder]:
        return HashEmbedder

    @property
    def sample_texts(self) -> List[str]:
        return ["Hello world", "Different text"]

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "default"

    def test_empty_text_zero_vector(self) -> None:
        """Empty text should produce zero vector (not normalized)."""
        embeddings = HashEmbedder._embed_texts([""], 128, "default")
        assert all(x == 0.0 for x in embeddings[0])

    def test_multiple_texts(self) -> None:
        """Should handle multiple texts."""
        embeddings = HashEmbedder._embed_texts(["First text", "Second text", "Third text"], 64, "default")
        assert len(embeddings) == 3
        assert all(len(e) == 64 for e in embeddings)
