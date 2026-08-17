"""Tests for ExactHashDeduplicator."""

from rag_integration.feature_groups.rag_pipeline.deduplication import ExactHashDeduplicator
from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator
from tests.feature_groups.deduplication.text_dedup_test_base import TextDeduplicationTestBase


class TestExactHashDeduplicator(TextDeduplicationTestBase):
    """Tests for ExactHashDeduplicator."""

    @property
    def deduplicator_class(self) -> type[BaseDeduplicator]:
        return ExactHashDeduplicator

    @property
    def duplicate_texts(self) -> list[str]:
        return ["Hello", "World", "Hello", "Test"]

    @property
    def duplicate_expected_indices(self) -> list[int | None]:
        return [None, None, 0, None]

    @property
    def unique_texts(self) -> list[str]:
        return ["A", "B", "C"]

    @property
    def default_threshold(self) -> float:
        return 1.0
