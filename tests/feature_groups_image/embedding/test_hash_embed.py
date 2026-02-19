"""Tests for HashImageEmbedder."""

import math

import pytest

from rag_integration.feature_groups.image_pipeline.embedding import HashImageEmbedder


def _can_import_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def _create_test_image(color: tuple[int, ...] = (128, 64, 32)) -> bytes:
    """Create a simple test image."""
    from PIL import Image
    import io

    img = Image.new("RGB", (64, 64), color=color)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.skipif(not _can_import_pillow(), reason="Pillow required")
class TestHashImageEmbedder:
    """Tests for HashImageEmbedder."""

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        image_data = _create_test_image()
        embedding = HashImageEmbedder._embed_image(image_data, 256, "default")
        assert len(embedding) == 256

    def test_deterministic_embeddings(self) -> None:
        """Same image should produce same embedding."""
        image_data = _create_test_image()
        emb1 = HashImageEmbedder._embed_image(image_data, 128, "default")
        emb2 = HashImageEmbedder._embed_image(image_data, 128, "default")
        assert emb1 == emb2

    def test_different_images_different_embeddings(self) -> None:
        """Different images should produce different embeddings."""
        img1 = _create_test_image((255, 0, 0))
        img2 = _create_test_image((0, 0, 255))
        emb1 = HashImageEmbedder._embed_image(img1, 128, "default")
        emb2 = HashImageEmbedder._embed_image(img2, 128, "default")
        assert emb1 != emb2

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        image_data = _create_test_image()
        embedding = HashImageEmbedder._embed_image(image_data, 128, "default")
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 0.001

    def test_empty_image_returns_zero_vector(self) -> None:
        """Empty image data should return zero vector."""
        embedding = HashImageEmbedder._embed_image(b"", 128, "default")
        assert len(embedding) == 128
        assert all(x == 0.0 for x in embedding)

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features."""
        from mloda.user import Options

        assert HashImageEmbedder.match_feature_group_criteria("image_docs__deduped__embedded", Options())
        assert not HashImageEmbedder.match_feature_group_criteria("image_docs__deduped", Options())
