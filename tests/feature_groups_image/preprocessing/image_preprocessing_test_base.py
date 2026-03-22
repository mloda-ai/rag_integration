"""Base test class for image preprocessing feature groups."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import List, Type

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.preprocessing.base import BaseImagePreprocessor


def can_import_pillow() -> bool:
    """Check if Pillow is available."""
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def create_test_image(
    width: int = 200,
    height: int = 300,
    mode: str = "RGB",
    color: tuple[int, ...] = (128, 64, 32),
) -> bytes:
    """Create a test PNG image."""
    from PIL import Image

    if mode == "RGBA":
        color = color + (200,) if len(color) == 3 else color
    img = Image.new(mode, (width, height), color=color)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


class ImagePreprocessingTestBase(ABC):
    """Abstract base providing shared tests for all image preprocessing implementations."""

    @property
    @abstractmethod
    def preprocessor_class(self) -> Type[BaseImagePreprocessor]: ...

    @property
    @abstractmethod
    def target_size(self) -> List[int]: ...

    @property
    @abstractmethod
    def feature_match_name(self) -> str: ...

    @property
    @abstractmethod
    def feature_reject_name(self) -> str: ...

    def test_resize_to_target(self) -> None:
        """Should resize image to target dimensions."""
        from PIL import Image

        image_data = create_test_image(200, 300)
        result = self.preprocessor_class._preprocess_image(image_data, "png", self.target_size)
        img = Image.open(io.BytesIO(result))
        assert img.size[0] <= self.target_size[0]
        assert img.size[1] <= self.target_size[1]

    def test_output_is_bytes(self) -> None:
        """Result should be bytes."""
        image_data = create_test_image()
        result = self.preprocessor_class._preprocess_image(image_data, "png", self.target_size)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_feature_matching_pattern(self) -> None:
        """Should match preprocessed features and reject others."""
        assert self.preprocessor_class.match_feature_group_criteria(self.feature_match_name, Options())
        assert not self.preprocessor_class.match_feature_group_criteria(self.feature_reject_name, Options())
