"""Tests for ParagraphChunker."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.chunking import ParagraphChunker
from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker
from tests.feature_groups.chunking.text_chunking_test_base import TextChunkingTestBase


class TestParagraphChunker(TextChunkingTestBase):
    """Tests for ParagraphChunker."""

    @property
    def chunker_class(self) -> Type[BaseChunker]:
        return ParagraphChunker

    def test_split_at_double_newline(self) -> None:
        """Should split on paragraph boundaries (double newline)."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = ParagraphChunker._chunk_text(text, 500, 0)
        assert len(chunks) >= 1

    def test_combine_short_paragraphs(self) -> None:
        """Should combine short paragraphs until chunk_size is reached."""
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
        chunks = ParagraphChunker._chunk_text(text, 100, 0)
        assert len(chunks) < 4

    def test_long_paragraph_split(self) -> None:
        """Should split long paragraphs that exceed chunk_size."""
        long_para = "word " * 100
        text = f"{long_para}\n\nShort para."
        chunks = ParagraphChunker._chunk_text(text, 100, 0)
        assert len(chunks) > 2

    def test_single_paragraph(self) -> None:
        """Should handle single paragraph."""
        text = "Just one paragraph without any double newlines."
        chunks = ParagraphChunker._chunk_text(text, 200, 0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_whitespace_only(self) -> None:
        """Should handle whitespace-only text."""
        chunks = ParagraphChunker._chunk_text("   ", 100, 0)
        assert len(chunks) == 1

    def test_paragraph_preserved_within_chunk_size(self) -> None:
        """Should preserve paragraph structure when within chunk_size."""
        text = "Para one.\n\nPara two."
        chunks = ParagraphChunker._chunk_text(text, 500, 0)
        assert len(chunks) == 1
        assert "\n\n" in chunks[0]
