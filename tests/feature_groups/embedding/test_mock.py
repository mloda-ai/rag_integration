"""Tests for MockEmbedder."""

from typing import List, Type

from rag_integration.feature_groups.rag_pipeline.embedding import MockEmbedder
from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder
from tests.feature_groups.embedding.text_embedding_test_base import TextEmbeddingTestBase


class TestMockEmbedder(TextEmbeddingTestBase):
    """Tests for MockEmbedder."""

    @property
    def embedder_class(self) -> Type[BaseEmbedder]:
        return MockEmbedder

    @property
    def sample_texts(self) -> List[str]:
        return ["Hello world", "Different text"]

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "default"
