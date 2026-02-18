"""Tests for NormalizePreprocessor."""

import pytest

from rag_integration.feature_groups.image_pipeline.preprocessing import NormalizePreprocessor


def _can_import_pillow() -> bool:
    try:
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _create_test_image_rgba(width: int = 100, height: int = 100) -> bytes:
    """Create a test RGBA PNG image."""
    from PIL import Image
    import io

    img = Image.new("RGBA", (width, height), color=(128, 64, 32, 200))
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.skipif(not _can_import_pillow(), reason="Pillow required")
class TestNormalizePreprocessor:
    """Tests for NormalizePreprocessor."""

    def test_converts_rgba_to_rgb(self) -> None:
        """Should convert RGBA images to RGB."""
        from PIL import Image
        import io

        image_data = _create_test_image_rgba()
        result = NormalizePreprocessor._preprocess_image(image_data, "png", [100, 100])

        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_resize_to_target(self) -> None:
        """Should resize to target dimensions."""
        from PIL import Image
        import io

        image_data = _create_test_image_rgba(200, 300)
        result = NormalizePreprocessor._preprocess_image(image_data, "png", [64, 64])

        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_output_is_png(self) -> None:
        """Normalized output should always be PNG."""
        from PIL import Image
        import io

        image_data = _create_test_image_rgba()
        result = NormalizePreprocessor._preprocess_image(image_data, "jpeg", [100, 100])

        img = Image.open(io.BytesIO(result))
        assert img.format == "PNG"

    def test_feature_matching_pattern(self) -> None:
        """Should match preprocessed features."""
        from mloda.user import Options

        assert NormalizePreprocessor.match_feature_group_criteria("image_docs__preprocessed", Options())
        assert not NormalizePreprocessor.match_feature_group_criteria("image_docs__embedded", Options())
