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

    @staticmethod
    def _leading_overlap(first: str, second: str) -> int:
        """Length of the longest suffix of ``first`` that is a prefix of ``second``."""
        for size in range(min(len(first), len(second)), 0, -1):
            if first[-size:] == second[:size]:
                return size
        return 0

    def test_overlap_stays_consistent_after_word_boundary_trim(self) -> None:
        """Word-boundary trims must not erode the requested overlap (issue #6).

        Varied word lengths force trims at different offsets within each chunk.
        Before the fix, ``start`` advanced by the full fixed step regardless of
        how far the trim pulled ``end`` back, so the real overlap drifted well
        below ``chunk_overlap``. The overlap should now stay at ``chunk_overlap``
        (give or take one character lost to whitespace stripping).
        """
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 10
        chunk_size, chunk_overlap = 80, 20
        chunks = FixedSizeChunker._chunk_text(text, chunk_size, chunk_overlap)
        assert len(chunks) > 2

        overlaps = [self._leading_overlap(a, b) for a, b in zip(chunks, chunks[1:])]
        assert all(overlap >= chunk_overlap - 1 for overlap in overlaps), overlaps

    def test_no_source_text_dropped_after_word_boundary_trim(self) -> None:
        """Word-boundary trims must not drop source text (issue #6).

        The same root cause behind the overlap drift also lost text: when a trim
        pulled ``end`` back by more than ``chunk_overlap``, the old fixed-step
        ``start`` jumped past ``end``, so the characters in between landed in no
        chunk at all. The effect is worst at low overlap. Advancing relative to
        the trimmed ``end`` keeps coverage continuous: every source character is
        covered by at least one chunk.

        Note this is about *coverage*, not whole-word presence: a chunk may start
        mid-word, so a word can legitimately be split across two chunks while all
        of its characters remain present. Fixed-width unique tokens make every
        chunk a unique substring, so each chunk's source span is unambiguous and
        we can check coverage directly.
        """
        text = " ".join(f"t{i:04d}" for i in range(120))

        # Low-overlap combinations are where the old step skipped trimmed text.
        for chunk_size, chunk_overlap in [(100, 0), (40, 3), (30, 2)]:
            chunks = FixedSizeChunker._chunk_text(text, chunk_size, chunk_overlap)
            covered = [False] * len(text)
            for chunk in chunks:
                assert text.count(chunk) == 1, (chunk_size, chunk_overlap, chunk)
                start = text.index(chunk)
                for index in range(start, start + len(chunk)):
                    covered[index] = True
            dropped = [i for i, char in enumerate(text) if not covered[i] and not char.isspace()]
            assert not dropped, (chunk_size, chunk_overlap, dropped)
