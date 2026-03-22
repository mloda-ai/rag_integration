"""Tests for FileImageSource."""

import tempfile
from pathlib import Path
from typing import Type

import pytest
from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.image_source import FileImageSource
from rag_integration.feature_groups.image_pipeline.image_source.base import BaseImageSource
from tests.feature_groups_image.image_source.image_source_test_base import ImageSourceTestBase


class TestFileImageSource(ImageSourceTestBase):
    """Tests for FileImageSource."""

    @property
    def source_class(self) -> Type[BaseImageSource]:
        return FileImageSource

    def test_load_from_file_paths(self) -> None:
        """Should load images from explicit file paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test.png"
            img_path.write_bytes(b"\x89PNG_test_content")

            result = FileImageSource._load_images(Options(context={"file_paths": [str(img_path)]}))
            assert len(result) == 1
            assert result[0]["image_data"] == b"\x89PNG_test_content"
            assert result[0]["format"] == "png"
            assert result[0]["image_id"] == "test"

    def test_load_from_directory(self) -> None:
        """Should load all images from directory."""
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
        with pytest.raises(ValueError, match="Either image_dir or file_paths is required"):
            FileImageSource._load_images(Options())

    def test_directory_not_found_error(self) -> None:
        """Should raise ValueError for nonexistent directory."""
        with pytest.raises(ValueError, match="Directory not found"):
            FileImageSource._load_images(Options(context={"image_dir": "/nonexistent/path"}))
