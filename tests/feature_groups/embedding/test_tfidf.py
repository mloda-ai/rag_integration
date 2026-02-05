"""Tests for TfidfEmbedder."""

import math

from rag_integration.feature_groups.rag_pipeline.embedding import TfidfEmbedder


class TestTfidfEmbedder:
    """Tests for TfidfEmbedder."""

    def test_embedding_dimension(self) -> None:
        """Embeddings should have correct dimension."""
        texts = ["Hello world test"]
        embeddings = TfidfEmbedder._embed_texts(texts, 256, "default")
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 256

    def test_tokenization(self) -> None:
        """Should filter out words with 2 or fewer characters."""
        tokens = TfidfEmbedder._tokenize("I am a big cat")
        # "I", "am", "a" should be filtered out (len <= 2)
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
        assert tf["word"] == 0.75  # 3/4
        assert tf["other"] == 0.25  # 1/4

    def test_tf_empty_list(self) -> None:
        """Should handle empty token list."""
        tf = TfidfEmbedder._compute_tf([])
        assert tf == {}

    def test_idf_computation(self) -> None:
        """Should compute IDF correctly."""
        documents = [["apple", "banana"], ["apple", "cherry"], ["banana", "cherry"]]
        vocab = {"apple", "banana", "cherry"}
        idf = TfidfEmbedder._compute_idf(documents, vocab)
        # apple: in 2 docs, banana: in 2 docs, cherry: in 2 docs
        # IDF = log((n_docs + 1) / (doc_count + 1)) + 1
        # All words appear in 2 of 3 docs
        expected_idf = math.log(4 / 3) + 1
        assert abs(idf["apple"] - expected_idf) < 0.01
        assert abs(idf["banana"] - expected_idf) < 0.01

    def test_unit_length_normalization(self) -> None:
        """Embeddings should be normalized to unit length."""
        texts = ["This is a test document with several words"]
        embeddings = TfidfEmbedder._embed_texts(texts, 128, "default")
        magnitude = math.sqrt(sum(x * x for x in embeddings[0]))
        assert abs(magnitude - 1.0) < 0.001

    def test_empty_text_handling(self) -> None:
        """Should handle empty text gracefully."""
        texts = [""]
        embeddings = TfidfEmbedder._embed_texts(texts, 128, "default")
        # Empty text should produce zero vector
        assert all(x == 0.0 for x in embeddings[0])

    def test_multiple_texts(self) -> None:
        """Should handle multiple texts with shared vocabulary."""
        texts = ["apple banana", "banana cherry", "cherry apple"]
        embeddings = TfidfEmbedder._embed_texts(texts, 64, "default")
        assert len(embeddings) == 3
        assert all(len(e) == 64 for e in embeddings)

    def test_feature_matching_pattern(self) -> None:
        """Should match embedded features."""
        from mloda.user import Options

        assert TfidfEmbedder.match_feature_group_criteria("docs__deduped__embedded", Options())
        assert not TfidfEmbedder.match_feature_group_criteria("docs__deduped", Options())
