"""Tests for NGramDeduplicator."""

from rag_integration.feature_groups.rag_pipeline.deduplication import NGramDeduplicator


class TestNGramDeduplicator:
    """Tests for NGramDeduplicator."""

    def test_find_similar_texts(self) -> None:
        """Should find near-duplicates with low threshold."""
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",  # Similar
            "Completely different text here",
        ]
        result = NGramDeduplicator._find_duplicates(texts, 0.7)
        assert result[0] is None  # First is canonical
        assert result[1] == 0  # Second is similar to first
        assert result[2] is None  # Third is unique

    def test_exact_match_threshold(self) -> None:
        """Should only find exact matches with threshold=1.0."""
        texts = [
            "The quick brown fox",
            "The quick brown fox",  # Exact match
            "The quick brown foxes",  # Slightly different
        ]
        result = NGramDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None
        assert result[1] == 0  # Exact match
        assert result[2] is None  # Not exact, so unique

    def test_jaccard_similarity(self) -> None:
        """Should compute Jaccard similarity correctly."""
        set1 = {"abc", "bcd", "cde"}
        set2 = {"abc", "bcd", "xyz"}
        similarity = NGramDeduplicator._jaccard_similarity(set1, set2)
        # Intersection: {"abc", "bcd"} = 2, Union: {"abc", "bcd", "cde", "xyz"} = 4
        assert abs(similarity - 0.5) < 0.01

    def test_jaccard_similarity_empty(self) -> None:
        """Should return 0 for empty sets."""
        assert NGramDeduplicator._jaccard_similarity(set(), {"abc"}) == 0.0
        assert NGramDeduplicator._jaccard_similarity({"abc"}, set()) == 0.0
        assert NGramDeduplicator._jaccard_similarity(set(), set()) == 0.0

    def test_ngram_extraction(self) -> None:
        """Should extract character n-grams correctly."""
        ngrams = NGramDeduplicator._get_ngrams("hello", 3)
        assert ngrams == {"hel", "ell", "llo"}

    def test_ngram_short_text(self) -> None:
        """Should handle text shorter than n-gram size."""
        ngrams = NGramDeduplicator._get_ngrams("ab", 3)
        assert ngrams == {"ab"}

    def test_no_duplicates(self) -> None:
        """Should return None for all unique texts."""
        texts = ["apple", "banana", "cherry"]
        result = NGramDeduplicator._find_duplicates(texts, 0.9)
        assert all(r is None for r in result)

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features."""
        from mloda.user import Options

        assert NGramDeduplicator.match_feature_group_criteria("docs__chunked__deduped", Options())
        assert not NGramDeduplicator.match_feature_group_criteria("docs__chunked", Options())
