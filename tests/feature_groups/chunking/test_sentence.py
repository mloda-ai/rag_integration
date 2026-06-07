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

    def test_overlap_repeats_trailing_sentence(self) -> None:
        """chunk_overlap (characters) should repeat whole trailing sentences across chunks."""
        text = "Alpha one. Beta two. Gamma three. Delta four."
        # chunk_size forces multiple chunks; overlap large enough to fit a trailing sentence.
        chunks = SentenceChunker._chunk_text(text, 22, 12)
        assert len(chunks) >= 2
        # Each chunk must start by repeating the last sentence of the previous chunk.
        for prev, nxt in zip(chunks, chunks[1:]):
            assert nxt.split(". ")[0].rstrip(".") == prev.split(". ")[-1].rstrip(".")

    def test_overlap_clamped_to_chunk_size(self) -> None:
        """overlap >= chunk_size is clamped so it cannot stall progress or duplicate unboundedly."""
        text = "Alpha one. Beta two. Gamma three. Delta four."
        # An overlap at/above chunk_size collapses to the same output as chunk_size - 1.
        clamped = SentenceChunker._chunk_text(text, 22, 21)
        assert SentenceChunker._chunk_text(text, 22, 1000) == clamped

    def test_zero_overlap_no_repeat(self) -> None:
        """With chunk_overlap=0 no sentence should be repeated between consecutive chunks."""
        text = "Alpha one. Beta two. Gamma three. Delta four."
        chunks = SentenceChunker._chunk_text(text, 22, 0)
        joined = " ".join(chunks)
        # Each sentence appears exactly once when there is no overlap.
        assert joined.count("Alpha one") == 1
        assert joined.count("Beta two") == 1

    def test_overlap_helper_respects_character_budget(self) -> None:
        """_overlap_sentences should only keep trailing sentences within the character budget."""
        sentences = ["aaaa", "bb", "cccccc"]
        # Budget 9: 'cccccc' (6) fits, adding 'bb' needs 6+1+2=9 which fits, 'aaaa' would exceed.
        assert SentenceChunker._overlap_sentences(sentences, 9) == ["bb", "cccccc"]
        # Budget smaller than the last sentence -> no overlap.
        assert SentenceChunker._overlap_sentences(sentences, 3) == []
        assert SentenceChunker._overlap_sentences(sentences, 0) == []
