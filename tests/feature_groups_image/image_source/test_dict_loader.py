"""Tests for DictImageSource."""

import pytest

from rag_integration.feature_groups.image_pipeline.image_source import DictImageSource


class TestDictImageSource:
    """Tests for DictImageSource."""

    def test_load_from_dict_list(self) -> None:
        """Should load images from list of dicts."""
        from mloda.user import Options

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
        from mloda.user import Options

        images = [b"raw_image_1", b"raw_image_2"]
        result = DictImageSource._load_images(Options(context={"images": images}))
        assert len(result) == 2
        assert result[0]["image_data"] == b"raw_image_1"
        assert result[0]["image_id"] == "img_0"
        assert result[1]["image_id"] == "img_1"

    def test_custom_field_mapping(self) -> None:
        """Should use custom field names."""
        from mloda.user import Options

        images = [{"id": "abc", "data": b"custom_data"}]
        result = DictImageSource._load_images(
            Options(context={"images": images, "image_data_field": "data", "id_field": "id"})
        )
        assert result[0]["image_id"] == "abc"
        assert result[0]["image_data"] == b"custom_data"

    def test_missing_images_error(self) -> None:
        """Should raise ValueError when images not provided."""
        from mloda.user import Options

        with pytest.raises(ValueError, match="images list is required"):
            DictImageSource._load_images(Options())

    def test_preserve_metadata(self) -> None:
        """Should preserve extra fields as metadata."""
        from mloda.user import Options

        images = [{"image_id": "1", "image_data": b"data", "format": "png", "camera": "Canon", "tags": ["nature"]}]
        result = DictImageSource._load_images(Options(context={"images": images}))
        assert result[0]["camera"] == "Canon"
        assert result[0]["tags"] == ["nature"]

    def test_feature_matching_pattern(self) -> None:
        """Should match image_docs features."""
        from mloda.user import Options

        assert DictImageSource.match_feature_group_criteria("image_docs", Options())
        assert not DictImageSource.match_feature_group_criteria("docs", Options())
        assert not DictImageSource.match_feature_group_criteria("image_docs__embedded", Options())
