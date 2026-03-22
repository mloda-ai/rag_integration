"""Tests for FixedSizeChunker."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.chunking import FixedSizeChunker
from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker
from tests.feature_groups.chunking.text_chunking_test_base import TextChunkingTestBase


class TestFixedSizeChunker(TextChunkingTestBase):
    """Tests for FixedSizeChunker."""

    @property
    def chunker_class(self) -> Type[BaseChunker]:
        return FixedSizeChunker

    def test_chunk_short_text(self) -> None:
        """Short text should return single chunk."""
        chunks = FixedSizeChunker._chunk_text("Hello world", 100, 10)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_chunk_long_text(self) -> None:
        """Long text should be split into multiple chunks."""
        text = "This is a test. " * 50  # ~800 chars
        chunks = FixedSizeChunker._chunk_text(text, 100, 10)
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)

    def test_chunk_overlap(self) -> None:
        """Chunks should have overlap."""
        text = "word " * 100  # 500 chars
        chunks = FixedSizeChunker._chunk_text(text, 100, 50)
        assert len(chunks) > 2
