"""Tests for NGramDeduplicator."""

from typing import List, Optional, Type

from rag_integration.feature_groups.rag_pipeline.deduplication import NGramDeduplicator
from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator
from tests.feature_groups.deduplication.text_dedup_test_base import TextDeduplicationTestBase


class TestNGramDeduplicator(TextDeduplicationTestBase):
    """Tests for NGramDeduplicator."""

    @property
    def deduplicator_class(self) -> Type[BaseDeduplicator]:
        return NGramDeduplicator

    @property
    def duplicate_texts(self) -> List[str]:
        return [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Completely different text here",
        ]

    @property
    def duplicate_expected_indices(self) -> List[Optional[int]]:
        return [None, 0, None]

    @property
    def unique_texts(self) -> List[str]:
        return ["apple", "banana", "cherry"]

    @property
    def default_threshold(self) -> float:
        return 0.7

    def test_exact_match_threshold(self) -> None:
        """Should only find exact matches with threshold=1.0."""
        texts = [
            "The quick brown fox",
            "The quick brown fox",
            "The quick brown foxes",
        ]
        result = NGramDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] is None

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
