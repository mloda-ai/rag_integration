"""Tests for FileImageSource."""

import tempfile
from pathlib import Path

import pytest

from rag_integration.feature_groups.image_pipeline.image_source import FileImageSource


class TestFileImageSource:
    """Tests for FileImageSource."""

    def test_load_from_file_paths(self) -> None:
        """Should load images from explicit file paths."""
        from mloda.user import Options

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test image files
            img_path = Path(tmp_dir) / "test.png"
            img_path.write_bytes(b"\x89PNG_test_content")

            result = FileImageSource._load_images(Options(context={"file_paths": [str(img_path)]}))
            assert len(result) == 1
            assert result[0]["image_data"] == b"\x89PNG_test_content"
            assert result[0]["format"] == "png"
            assert result[0]["image_id"] == "test"

    def test_load_from_directory(self) -> None:
        """Should load all images from directory."""
        from mloda.user import Options

        with tempfile.TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "a.png").write_bytes(b"img_a")
            (Path(tmp_dir) / "b.jpg").write_bytes(b"img_b")
            (Path(tmp_dir) / "readme.txt").write_bytes(b"not an image")

            result = FileImageSource._load_images(Options(context={"image_dir": tmp_dir}))
            assert len(result) == 2
            formats = {r["format"] for r in result}
            assert "png" in formats
            assert "jpeg" in formats

    def test_missing_source_error(self) -> None:
        """Should raise ValueError when no source provided."""
        from mloda.user import Options

        with pytest.raises(ValueError, match="Either image_dir or file_paths is required"):
            FileImageSource._load_images(Options())

    def test_directory_not_found_error(self) -> None:
        """Should raise ValueError for nonexistent directory."""
        from mloda.user import Options

        with pytest.raises(ValueError, match="Directory not found"):
            FileImageSource._load_images(Options(context={"image_dir": "/nonexistent/path"}))
