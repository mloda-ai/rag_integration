"""Tests for MockImageEmbedder."""

import math

from rag_integration.feature_groups.image_pipeline.embedding import MockImageEmbedder


class TestMockImageEmbedder:
    """Tests for MockImageEmbedder."""

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        embedding = MockImageEmbedder._embed_image(b"test_image", 512, "default")
        assert len(embedding) == 512

    def test_deterministic_embeddings(self) -> None:
        """Same image should produce same embedding."""
        emb1 = MockImageEmbedder._embed_image(b"test_image", 128, "default")
        emb2 = MockImageEmbedder._embed_image(b"test_image", 128, "default")
        assert emb1 == emb2

    def test_different_images_different_embeddings(self) -> None:
        """Different images should produce different embeddings."""
        emb1 = MockImageEmbedder._embed_image(b"image_a", 128, "default")
        emb2 = MockImageEmbedder._embed_image(b"image_b", 128, "default")
        assert emb1 != emb2

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        embedding = MockImageEmbedder._embed_image(b"test_image", 128, "default")
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 0.001

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features."""
        from mloda.user import Options

        assert MockImageEmbedder.match_feature_group_criteria("image_docs__deduped__embedded", Options())
        assert not MockImageEmbedder.match_feature_group_criteria("image_docs__deduped", Options())
