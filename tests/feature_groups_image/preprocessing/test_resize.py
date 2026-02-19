"""Tests for ResizePreprocessor."""

import pytest

from rag_integration.feature_groups.image_pipeline.preprocessing import ResizePreprocessor


def _can_import_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        return False


def _create_test_image(width: int = 200, height: int = 300) -> bytes:
    """Create a simple test PNG image."""
    from PIL import Image
    import io

    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.skipif(not _can_import_pillow(), reason="Pillow required")
class TestResizePreprocessor:
    """Tests for ResizePreprocessor."""

    def test_resize_to_target(self) -> None:
        """Should resize image to exact target dimensions."""
        from PIL import Image
        import io

        image_data = _create_test_image(200, 300)
        result = ResizePreprocessor._preprocess_image(image_data, "png", [100, 100])

        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 100)

    def test_resize_upscale(self) -> None:
        """Should upscale smaller images."""
        from PIL import Image
        import io

        image_data = _create_test_image(50, 50)
        result = ResizePreprocessor._preprocess_image(image_data, "png", [224, 224])

        img = Image.open(io.BytesIO(result))
        assert img.size == (224, 224)

    def test_output_is_bytes(self) -> None:
        """Result should be bytes."""
        image_data = _create_test_image()
        result = ResizePreprocessor._preprocess_image(image_data, "png", [64, 64])
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_feature_matching_pattern(self) -> None:
        """Should match preprocessed features."""
        from mloda.user import Options

        assert ResizePreprocessor.match_feature_group_criteria("image_docs__pii_redacted__preprocessed", Options())
        assert not ResizePreprocessor.match_feature_group_criteria("image_docs__pii_redacted", Options())
