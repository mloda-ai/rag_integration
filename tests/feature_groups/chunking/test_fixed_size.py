"""Tests for FixedSizeChunker."""

from rag_integration.feature_groups.rag_pipeline.chunking import FixedSizeChunker


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

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
        # With 50 char overlap, chunks should share some content
        assert len(chunks) > 2

    def test_empty_text(self) -> None:
        """Empty text should return empty chunk."""
        chunks = FixedSizeChunker._chunk_text("", 100, 10)
        assert len(chunks) == 1

    def test_feature_matching_pattern(self) -> None:
        """Should match chunked features."""
        from mloda.user import Options

        assert FixedSizeChunker.match_feature_group_criteria("docs__pii_redacted__chunked", Options())
        assert not FixedSizeChunker.match_feature_group_criteria("docs__pii_redacted", Options())
