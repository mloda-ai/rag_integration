"""Tests for HashEmbedder."""

import math

from rag_integration.feature_groups.rag_pipeline.embedding import HashEmbedder


class TestHashEmbedder:
    """Tests for HashEmbedder."""

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        texts = ["Hello world"]
        embeddings = HashEmbedder._embed_texts(texts, 256, "default")
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 256

    def test_deterministic(self) -> None:
        """Same text should produce same embedding."""
        texts = ["Hello world"]
        emb1 = HashEmbedder._embed_texts(texts, 128, "default")
        emb2 = HashEmbedder._embed_texts(texts, 128, "default")
        assert emb1[0] == emb2[0]

    def test_different_texts_different_embeddings(self) -> None:
        """Different texts should produce different embeddings."""
        texts = ["Hello", "World"]
        embeddings = HashEmbedder._embed_texts(texts, 128, "default")
        assert embeddings[0] != embeddings[1]

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        texts = ["Test text with multiple words"]
        embeddings = HashEmbedder._embed_texts(texts, 128, "default")
        magnitude = math.sqrt(sum(x * x for x in embeddings[0]))
        assert abs(magnitude - 1.0) < 0.001

    def test_empty_text_zero_vector(self) -> None:
        """Empty text should produce zero vector (not normalized)."""
        texts = [""]
        embeddings = HashEmbedder._embed_texts(texts, 128, "default")
        # Empty text has no words, so all zeros
        assert all(x == 0.0 for x in embeddings[0])

    def test_multiple_texts(self) -> None:
        """Should handle multiple texts."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = HashEmbedder._embed_texts(texts, 64, "default")
        assert len(embeddings) == 3
        assert all(len(e) == 64 for e in embeddings)

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features."""
        from mloda.user import Options

        assert HashEmbedder.match_feature_group_criteria("docs__deduped__embedded", Options())
        assert not HashEmbedder.match_feature_group_criteria("docs__deduped", Options())
