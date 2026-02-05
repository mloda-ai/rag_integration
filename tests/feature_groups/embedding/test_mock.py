"""Tests for MockEmbedder."""

import math

from rag_integration.feature_groups.rag_pipeline.embedding import MockEmbedder


class TestMockEmbedder:
    """Tests for MockEmbedder."""

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        texts = ["Hello world"]
        embeddings = MockEmbedder._embed_texts(texts, 384, "default")
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_deterministic_embeddings(self) -> None:
        """Same text should produce same embedding."""
        texts = ["Hello world"]
        emb1 = MockEmbedder._embed_texts(texts, 128, "default")
        emb2 = MockEmbedder._embed_texts(texts, 128, "default")
        assert emb1[0] == emb2[0]

    def test_different_texts_different_embeddings(self) -> None:
        """Different texts should produce different embeddings."""
        texts = ["Hello", "World"]
        embeddings = MockEmbedder._embed_texts(texts, 128, "default")
        assert embeddings[0] != embeddings[1]

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        texts = ["Test text"]
        embeddings = MockEmbedder._embed_texts(texts, 128, "default")
        magnitude = math.sqrt(sum(x * x for x in embeddings[0]))
        assert abs(magnitude - 1.0) < 0.001

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features."""
        from mloda.user import Options

        assert MockEmbedder.match_feature_group_criteria("docs__deduped__embedded", Options())
        assert not MockEmbedder.match_feature_group_criteria("docs__deduped", Options())
