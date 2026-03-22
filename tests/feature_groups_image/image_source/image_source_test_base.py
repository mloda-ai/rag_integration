"""Base test class for image source feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.image_source.base import BaseImageSource


class ImageSourceTestBase(ABC):
    """Abstract base providing shared tests for all image source implementations."""

    @property
    @abstractmethod
    def source_class(self) -> Type[BaseImageSource]: ...

    def test_feature_matching_pattern(self) -> None:
        """Should match image_docs features and reject others."""
        assert self.source_class.match_feature_group_criteria("image_docs", Options())
        assert not self.source_class.match_feature_group_criteria("docs", Options())
        assert not self.source_class.match_feature_group_criteria("image_docs__embedded", Options())
