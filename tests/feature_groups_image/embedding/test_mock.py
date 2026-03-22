"""Tests for MockImageEmbedder."""

from typing import Type

from rag_integration.feature_groups.image_pipeline.embedding import MockImageEmbedder
from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder
from tests.feature_groups_image.embedding.image_embedding_test_base import ImageEmbeddingTestBase


class TestMockImageEmbedder(ImageEmbeddingTestBase):
    """Tests for MockImageEmbedder."""

    @property
    def embedder_class(self) -> Type[BaseImageEmbedder]:
        return MockImageEmbedder

    @property
    def sample_image(self) -> bytes:
        return b"test_image"

    @property
    def sample_image_alt(self) -> bytes:
        return b"different_image"

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "default"
