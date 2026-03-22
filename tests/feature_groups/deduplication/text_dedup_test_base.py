"""Base test class for text deduplication feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Type

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.deduplication.base import BaseDeduplicator


class TextDeduplicationTestBase(ABC):
    """Abstract base providing shared tests for all text deduplication implementations."""

    @property
    @abstractmethod
    def deduplicator_class(self) -> Type[BaseDeduplicator]: ...

    @property
    @abstractmethod
    def duplicate_texts(self) -> List[str]: ...

    @property
    @abstractmethod
    def duplicate_expected_indices(self) -> List[Optional[int]]: ...

    @property
    @abstractmethod
    def unique_texts(self) -> List[str]: ...

    @property
    @abstractmethod
    def default_threshold(self) -> float: ...

    def test_find_duplicates(self) -> None:
        """Should detect duplicates in the provided test data."""
        result = self.deduplicator_class._find_duplicates(self.duplicate_texts, self.default_threshold)
        assert result == self.duplicate_expected_indices

    def test_no_duplicates(self) -> None:
        """Should return None for all unique texts."""
        result = self.deduplicator_class._find_duplicates(self.unique_texts, self.default_threshold)
        assert all(r is None for r in result)

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features and reject non-deduped."""
        assert self.deduplicator_class.match_feature_group_criteria("docs__chunked__deduped", Options())
        assert not self.deduplicator_class.match_feature_group_criteria("docs__chunked", Options())
