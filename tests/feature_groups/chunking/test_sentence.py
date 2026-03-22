"""Tests for SentenceChunker."""

from typing import Type

from rag_integration.feature_groups.rag_pipeline.chunking import SentenceChunker
from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker
from tests.feature_groups.chunking.text_chunking_test_base import TextChunkingTestBase


class TestSentenceChunker(TextChunkingTestBase):
    """Tests for SentenceChunker."""

    @property
    def chunker_class(self) -> Type[BaseChunker]:
        return SentenceChunker

    def test_split_at_sentence_boundaries(self) -> None:
        """Should split on sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = SentenceChunker._chunk_text(text, 50, 0)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "." in chunk

    def test_combine_short_sentences(self) -> None:
        """Should combine sentences until chunk_size is reached."""
        text = "A. B. C. D. E. F."
        chunks = SentenceChunker._chunk_text(text, 100, 0)
        assert len(chunks) < 6

    def test_long_sentence_preserved(self) -> None:
        """Should preserve long sentences without splitting mid-sentence."""
        long_sentence = "This is a very long sentence that exceeds the chunk size limit by quite a bit"
        text = f"{long_sentence}. Short one."
        chunks = SentenceChunker._chunk_text(text, 50, 0)
        assert any(long_sentence in chunk for chunk in chunks)

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
