"""Base test class for text chunking feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker


class TextChunkingTestBase(ABC):
    """Abstract base providing shared tests for all text chunking implementations."""

    @property
    @abstractmethod
    def chunker_class(self) -> Type[BaseChunker]: ...

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        chunks = self.chunker_class._chunk_text("", 100, 10)
        assert len(chunks) == 1

    def test_feature_matching_pattern(self) -> None:
        """Should match chunked features and reject non-chunked."""
        assert self.chunker_class.match_feature_group_criteria("docs__pii_redacted__chunked", Options())
        assert not self.chunker_class.match_feature_group_criteria("docs__pii_redacted", Options())
