"""Tests for PerceptualHashImageDeduplicator."""

import pytest

from rag_integration.feature_groups.image_pipeline.deduplication import PerceptualHashImageDeduplicator


def _can_import_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _create_test_image(color: tuple = (255, 0, 0), size: tuple = (64, 64)) -> bytes:
    """Create a simple solid-color test image."""
    from PIL import Image
    import io

    img = Image.new("RGB", size, color=color)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.skipif(not _can_import_pillow(), reason="Pillow required")
class TestPerceptualHashImageDeduplicator:
    """Tests for PerceptualHashImageDeduplicator."""

    def test_identical_images_detected(self) -> None:
        """Identical images should be detected as duplicates."""
        img = _create_test_image((255, 0, 0))
        images = [img, img]
        result = PerceptualHashImageDeduplicator._find_duplicates(images, 0.9)
        assert result[0] is None
        assert result[1] == 0

    def test_different_images_not_duplicates(self) -> None:
        """Structurally different images should not be duplicates."""
        from PIL import Image, ImageDraw
        import io

        # Create images with different internal structure (not just solid colors)
        img1 = Image.new("RGB", (64, 64), color=(255, 255, 255))
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([0, 0, 32, 64], fill=(0, 0, 0))  # Left half black

        img2 = Image.new("RGB", (64, 64), color=(255, 255, 255))
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([0, 0, 64, 32], fill=(0, 0, 0))  # Top half black

        buf1, buf2 = io.BytesIO(), io.BytesIO()
        img1.save(buf1, format="PNG")
        img2.save(buf2, format="PNG")

        images = [buf1.getvalue(), buf2.getvalue()]
        result = PerceptualHashImageDeduplicator._find_duplicates(images, 1.0)
        # At threshold 1.0 (exact hash match), structurally different images should not match
        assert result[0] is None
        assert result[1] is None

    def test_hamming_distance(self) -> None:
        """Hamming distance should be computed correctly."""
        assert PerceptualHashImageDeduplicator._hamming_distance(0b1100, 0b1010) == 2
        assert PerceptualHashImageDeduplicator._hamming_distance(0, 0) == 0

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features."""
        from mloda.user import Options

        assert PerceptualHashImageDeduplicator.match_feature_group_criteria(
            "image_docs__preprocessed__deduped", Options()
        )
