"""Image embedding feature groups."""

from rag_integration.feature_groups.image_pipeline.embedding.base import BaseImageEmbedder
from rag_integration.feature_groups.image_pipeline.embedding.mock import MockImageEmbedder
from rag_integration.feature_groups.image_pipeline.embedding.hash_embed import HashImageEmbedder
from rag_integration.feature_groups.image_pipeline.embedding.clip import CLIPImageEmbedder

__all__ = [
    "BaseImageEmbedder",
    "MockImageEmbedder",
    "HashImageEmbedder",
    "CLIPImageEmbedder",
]
