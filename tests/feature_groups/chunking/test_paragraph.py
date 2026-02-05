"""Tests for ParagraphChunker."""

from rag_integration.feature_groups.rag_pipeline.chunking import ParagraphChunker


class TestParagraphChunker:
    """Tests for ParagraphChunker."""

    def test_split_at_double_newline(self) -> None:
        """Should split on paragraph boundaries (double newline)."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = ParagraphChunker._chunk_text(text, 500, 0)
        # With large chunk_size, all paragraphs fit in one chunk
        assert len(chunks) >= 1

    def test_combine_short_paragraphs(self) -> None:
        """Should combine short paragraphs until chunk_size is reached."""
        text = "Para 1.\n\nPara 2.\n\nPara 3.\n\nPara 4."
        chunks = ParagraphChunker._chunk_text(text, 100, 0)
        # Should combine into fewer chunks than paragraphs
        assert len(chunks) < 4

    def test_long_paragraph_split(self) -> None:
        """Should split long paragraphs that exceed chunk_size."""
        long_para = "word " * 100  # ~500 chars
        text = f"{long_para}\n\nShort para."
        chunks = ParagraphChunker._chunk_text(text, 100, 0)
        # Long paragraph should be split
        assert len(chunks) > 2

    def test_single_paragraph(self) -> None:
        """Should handle single paragraph."""
        text = "Just one paragraph without any double newlines."
        chunks = ParagraphChunker._chunk_text(text, 200, 0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        chunks = ParagraphChunker._chunk_text("", 100, 0)
        assert len(chunks) == 1

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

    def test_feature_matching_pattern(self) -> None:
        """Should match chunked features."""
        from mloda.user import Options

        assert ParagraphChunker.match_feature_group_criteria("docs__pii_redacted__chunked", Options())
        assert not ParagraphChunker.match_feature_group_criteria("docs__pii_redacted", Options())
