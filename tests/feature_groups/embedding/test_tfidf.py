"""Tests for TfidfEmbedder."""

import math
from typing import List, Type

from rag_integration.feature_groups.rag_pipeline.embedding import TfidfEmbedder
from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder
from tests.feature_groups.embedding.text_embedding_test_base import TextEmbeddingTestBase


class TestTfidfEmbedder(TextEmbeddingTestBase):
    """Tests for TfidfEmbedder."""

    @property
    def embedder_class(self) -> Type[BaseEmbedder]:
        return TfidfEmbedder

    @property
    def sample_texts(self) -> List[str]:
        return ["Hello world test", "Different text here"]

    @property
    def embedding_dim(self) -> int:
        return 128

    @property
    def model_name(self) -> str:
        return "default"

    def test_tokenization(self) -> None:
        """Should filter out words with 2 or fewer characters."""
        tokens = TfidfEmbedder._tokenize("I am a big cat")
        assert "big" in tokens
        assert "cat" in tokens
        assert "I" not in tokens
        assert "am" not in tokens
        assert "a" not in tokens

    def test_tokenization_lowercase(self) -> None:
        """Should lowercase all tokens."""
        tokens = TfidfEmbedder._tokenize("Hello WORLD Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tf_computation(self) -> None:
        """Should compute term frequency correctly."""
        tokens = ["word", "word", "other", "word"]
        tf = TfidfEmbedder._compute_tf(tokens)
        assert tf["word"] == 0.75
        assert tf["other"] == 0.25

    def test_tf_empty_list(self) -> None:
        """Should handle empty token list."""
        tf = TfidfEmbedder._compute_tf([])
        assert tf == {}

    def test_idf_computation(self) -> None:
        """Should compute IDF correctly."""
        documents = [["apple", "banana"], ["apple", "cherry"], ["banana", "cherry"]]
        vocab = {"apple", "banana", "cherry"}
        idf = TfidfEmbedder._compute_idf(documents, vocab)
        expected_idf = math.log(4 / 3) + 1
        assert abs(idf["apple"] - expected_idf) < 0.01
        assert abs(idf["banana"] - expected_idf) < 0.01

    def test_empty_text_handling(self) -> None:
        """Should handle empty text gracefully."""
        embeddings = TfidfEmbedder._embed_texts([""], 128, "default")
        assert all(x == 0.0 for x in embeddings[0])

    def test_multiple_texts(self) -> None:
        """Should handle multiple texts with shared vocabulary."""
        embeddings = TfidfEmbedder._embed_texts(["apple banana", "banana cherry", "cherry apple"], 64, "default")
        assert len(embeddings) == 3
        assert all(len(e) == 64 for e in embeddings)
