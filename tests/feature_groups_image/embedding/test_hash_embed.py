"""Tests for HashImageEmbedder."""

from typing import Type

import pytest

from rag_integration.feature_groups.image_pipeline.embedding import HashImageEmbedder
from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder
from tests.feature_groups_image.embedding.image_embedding_test_base import (
    ImageEmbeddingTestBase,
    can_import_pillow,
    create_test_image,
)


@pytest.mark.skipif(not can_import_pillow(), reason="Pillow required")
class TestHashImageEmbedder(ImageEmbeddingTestBase):
    """Tests for HashImageEmbedder."""

    @property
    def embedder_class(self) -> Type[BaseImageEmbedder]:
        return HashImageEmbedder

    @property
    def sample_image(self) -> bytes:
        return create_test_image((128, 64, 32))

    @property
    def sample_image_alt(self) -> bytes:
        return create_test_image((0, 0, 255))

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "default"

    def test_empty_image_returns_zero_vector(self) -> None:
        """Empty image data should return zero vector."""
        embedding = HashImageEmbedder._embed_image(b"", 128, "default")
        assert len(embedding) == 128
        assert all(x == 0.0 for x in embedding)
