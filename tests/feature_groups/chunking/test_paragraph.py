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

    def test_overlap_repeats_trailing_paragraph(self) -> None:
        """chunk_overlap (characters) should repeat whole trailing paragraphs across chunks."""
        text = "\n\n".join(f"Paragraph number {i} here ok." for i in range(8))
        # chunk_size forces multiple chunks; overlap large enough to fit a trailing paragraph.
        chunks = ParagraphChunker._chunk_text(text, 90, 40)
        assert len(chunks) >= 2
        # Each chunk must start by repeating the last paragraph of the previous chunk.
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.split("\n\n")[0] == prev.split("\n\n")[-1]

    def test_overlap_clamped_to_chunk_size(self) -> None:
        """overlap >= chunk_size is clamped so it cannot stall progress or duplicate unboundedly."""
        text = "\n\n".join(f"Paragraph number {i} here ok." for i in range(8))
        # An overlap at/above chunk_size collapses to the same output as chunk_size - 1.
        clamped = ParagraphChunker._chunk_text(text, 90, 89)
        assert ParagraphChunker._chunk_text(text, 90, 100000) == clamped

    def test_zero_overlap_no_repeat(self) -> None:
        """With chunk_overlap=0 no paragraph should be repeated between consecutive chunks."""
        text = "\n\n".join(f"Paragraph number {i} here ok." for i in range(8))
        chunks = ParagraphChunker._chunk_text(text, 90, 0)
        joined = "\n\n".join(chunks)
        for i in range(8):
            assert joined.count(f"Paragraph number {i} here ok.") == 1

    def test_overlap_helper_respects_character_budget(self) -> None:
        """_overlap_paragraphs should only keep trailing paragraphs within the character budget."""
        paragraphs = ["aaaa", "bb", "cccccc"]
        # Budget 12: 'cccccc' (6) fits, adding 'bb' needs 6+2+2=10 which fits, 'aaaa' would exceed.
        assert ParagraphChunker._overlap_paragraphs(paragraphs, 12) == ["bb", "cccccc"]
        # Budget smaller than the last paragraph -> no overlap.
        assert ParagraphChunker._overlap_paragraphs(paragraphs, 3) == []
        assert ParagraphChunker._overlap_paragraphs(paragraphs, 0) == []
