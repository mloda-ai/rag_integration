"""Tests for NormalizedDeduplicator."""

from rag_integration.feature_groups.rag_pipeline.deduplication import NormalizedDeduplicator


class TestNormalizedDeduplicator:
    """Tests for NormalizedDeduplicator."""

    def test_find_whitespace_duplicates(self) -> None:
        """Should find duplicates that differ only in whitespace."""
        texts = ["hello world", "hello  world", "hello   world"]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None  # First is canonical
        assert result[1] == 0  # Second is dup of first
        assert result[2] == 0  # Third is dup of first

    def test_find_case_duplicates(self) -> None:
        """Should find duplicates that differ only in case."""
        texts = ["Hello World", "hello world", "HELLO WORLD"]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None  # First is canonical
        assert result[1] == 0  # Second is dup of first
        assert result[2] == 0  # Third is dup of first

    def test_combined_normalization(self) -> None:
        """Should find duplicates with both case and whitespace differences."""
        texts = ["Hello World", "HELLO   world", "  hello world  "]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] == 0

    def test_no_duplicates(self) -> None:
        """Should return None for all unique texts."""
        texts = ["Hello", "World", "Test"]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert all(r is None for r in result)

    def test_empty_texts(self) -> None:
        """Should handle empty strings."""
        texts = ["", "", "not empty"]
        result = NormalizedDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None  # First empty string
        assert result[1] == 0  # Second empty is dup of first
        assert result[2] is None  # Non-empty is unique

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features."""
        from mloda.user import Options

        assert NormalizedDeduplicator.match_feature_group_criteria("docs__chunked__deduped", Options())
        assert not NormalizedDeduplicator.match_feature_group_criteria("docs__chunked", Options())
