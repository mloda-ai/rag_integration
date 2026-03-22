"""Base test class for image embedding feature groups."""

from __future__ import annotations

import io
import math
from abc import ABC, abstractmethod
from typing import Type

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder


def can_import_pillow() -> bool:
    """Check if Pillow is available."""
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def create_test_image(color: tuple[int, ...] = (128, 64, 32), size: tuple[int, int] = (64, 64)) -> bytes:
    """Create a simple solid-color test image."""
    from PIL import Image

    img = Image.new("RGB", size, color=color)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()


class ImageEmbeddingTestBase(ABC):
    """Abstract base providing shared tests for all image embedding implementations."""

    @property
    @abstractmethod
    def embedder_class(self) -> Type[BaseImageEmbedder]: ...

    @property
    @abstractmethod
    def sample_image(self) -> bytes: ...

    @property
    @abstractmethod
    def sample_image_alt(self) -> bytes: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        embedding = self.embedder_class._embed_image(self.sample_image, self.embedding_dim, self.model_name)
        assert len(embedding) == self.embedding_dim

    def test_deterministic_embeddings(self) -> None:
        """Same image should produce same embedding."""
        emb1 = self.embedder_class._embed_image(self.sample_image, self.embedding_dim, self.model_name)
        emb2 = self.embedder_class._embed_image(self.sample_image, self.embedding_dim, self.model_name)
        assert emb1 == emb2

    def test_different_images_different_embeddings(self) -> None:
        """Different images should produce different embeddings."""
        emb1 = self.embedder_class._embed_image(self.sample_image, self.embedding_dim, self.model_name)
        emb2 = self.embedder_class._embed_image(self.sample_image_alt, self.embedding_dim, self.model_name)
        assert emb1 != emb2

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        embedding = self.embedder_class._embed_image(self.sample_image, self.embedding_dim, self.model_name)
        magnitude = math.sqrt(sum(x * x for x in embedding))
        assert abs(magnitude - 1.0) < 0.001

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features and reject non-embedded."""
        assert self.embedder_class.match_feature_group_criteria("image_docs__deduped__embedded", Options())
        assert not self.embedder_class.match_feature_group_criteria("image_docs__deduped", Options())
