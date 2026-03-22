"""Tests for DictImageSource."""

from typing import Type

import pytest
from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.image_source import DictImageSource
from rag_integration.feature_groups.image_pipeline.image_source.base import BaseImageSource
from tests.feature_groups_image.image_source.image_source_test_base import ImageSourceTestBase


class TestDictImageSource(ImageSourceTestBase):
    """Tests for DictImageSource."""

    @property
    def source_class(self) -> Type[BaseImageSource]:
        return DictImageSource

    def test_load_from_dict_list(self) -> None:
        """Should load images from list of dicts."""
        images = [
            {"image_id": "img_001", "image_data": b"\x89PNG_fake_data_1", "format": "png"},
            {"image_id": "img_002", "image_data": b"\xff\xd8_fake_data_2", "format": "jpeg"},
        ]
        result = DictImageSource._load_images(Options(context={"images": images}))
        assert len(result) == 2
        assert result[0]["image_id"] == "img_001"
        assert result[0]["image_data"] == b"\x89PNG_fake_data_1"
        assert result[1]["format"] == "jpeg"

    def test_load_raw_bytes(self) -> None:
        """Should handle list of raw bytes."""
        images = [b"raw_image_1", b"raw_image_2"]
        result = DictImageSource._load_images(Options(context={"images": images}))
        assert len(result) == 2
        assert result[0]["image_data"] == b"raw_image_1"
        assert result[0]["image_id"] == "img_0"
        assert result[1]["image_id"] == "img_1"

    def test_custom_field_mapping(self) -> None:
        """Should use custom field names."""
        images = [{"id": "abc", "data": b"custom_data"}]
        result = DictImageSource._load_images(
            Options(context={"images": images, "image_data_field": "data", "id_field": "id"})
        )
        assert result[0]["image_id"] == "abc"
        assert result[0]["image_data"] == b"custom_data"

    def test_missing_images_error(self) -> None:
        """Should raise ValueError when images not provided."""
        with pytest.raises(ValueError, match="images list is required"):
            DictImageSource._load_images(Options())

    def test_preserve_metadata(self) -> None:
        """Should preserve extra fields as metadata."""
        images = [{"image_id": "1", "image_data": b"data", "format": "png", "camera": "Canon", "tags": ["nature"]}]
        result = DictImageSource._load_images(Options(context={"images": images}))
        assert result[0]["camera"] == "Canon"
        assert result[0]["tags"] == ["nature"]
