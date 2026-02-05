"""Tests for SentenceChunker."""

from rag_integration.feature_groups.rag_pipeline.chunking import SentenceChunker


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_split_at_sentence_boundaries(self) -> None:
        """Should split on sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = SentenceChunker._chunk_text(text, 50, 0)
        # Each sentence is ~15 chars, so chunks should contain multiple sentences
        assert len(chunks) >= 1
        # All chunks should contain complete sentences (end with period)
        for chunk in chunks:
            assert "." in chunk

    def test_combine_short_sentences(self) -> None:
        """Should combine sentences until chunk_size is reached."""
        text = "A. B. C. D. E. F."  # Short sentences
        chunks = SentenceChunker._chunk_text(text, 100, 0)
        # Should combine into fewer chunks
        assert len(chunks) < 6

    def test_long_sentence_preserved(self) -> None:
        """Should preserve long sentences without splitting mid-sentence."""
        long_sentence = "This is a very long sentence that exceeds the chunk size limit by quite a bit"
        text = f"{long_sentence}. Short one."
        chunks = SentenceChunker._chunk_text(text, 50, 0)
        # Long sentence should be preserved intact
        assert any(long_sentence in chunk for chunk in chunks)

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        chunks = SentenceChunker._chunk_text("", 100, 0)
        assert len(chunks) == 1

    def test_whitespace_only(self) -> None:
        """Should handle whitespace-only text."""
        chunks = SentenceChunker._chunk_text("   ", 100, 0)
        assert len(chunks) == 1

    def test_single_sentence(self) -> None:
        """Should handle single sentence."""
        text = "Just one sentence here."
        chunks = SentenceChunker._chunk_text(text, 100, 0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_feature_matching_pattern(self) -> None:
        """Should match chunked features."""
        from mloda.user import Options

        assert SentenceChunker.match_feature_group_criteria("docs__pii_redacted__chunked", Options())
        assert not SentenceChunker.match_feature_group_criteria("docs__pii_redacted", Options())
