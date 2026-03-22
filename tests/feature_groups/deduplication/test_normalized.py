"""Tests for NormalizedDeduplicator."""

from typing import List, Optional, Type

from rag_integration.feature_groups.rag_pipeline.deduplication import NormalizedDeduplicator
from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator
from tests.feature_groups.deduplication.text_dedup_test_base import TextDeduplicationTestBase


class TestNormalizedDeduplicator(TextDeduplicationTestBase):
    """Tests for NormalizedDeduplicator."""

    @property
    def deduplicator_class(self) -> Type[BaseDeduplicator]:
        return NormalizedDeduplicator

    @property
    def duplicate_texts(self) -> List[str]:
        return ["hello world", "hello  world", "hello   world"]

    @property
    def duplicate_expected_indices(self) -> List[Optional[int]]:
        return [None, 0, 0]

    @property
    def unique_texts(self) -> List[str]:
        return ["Hello", "World", "Test"]

    @property
    def default_threshold(self) -> float:
        return 1.0

    def test_find_case_duplicates(self) -> None:
        """Should find duplicates that differ only in case."""
        texts = ["Hello World", "hello world", "HELLO WORLD"]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] == 0

    def test_combined_normalization(self) -> None:
        """Should find duplicates with both case and whitespace differences."""
        texts = ["Hello World", "HELLO   world", "  hello world  "]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] == 0

    def test_empty_texts(self) -> None:
        """Should handle empty strings."""
        texts = ["", "", "not empty"]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] is None
