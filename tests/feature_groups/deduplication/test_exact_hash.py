"""Tests for ExactHashDeduplicator."""

from rag_integration.feature_groups.rag_pipeline.deduplication import ExactHashDeduplicator


class TestExactHashDeduplicator:
    """Tests for ExactHashDeduplicator."""

    def test_find_exact_duplicates(self) -> None:
        """Should find exact duplicates."""
        texts = ["Hello", "World", "Hello", "Test"]
        result = ExactHashDeduplicator._find_duplicates(texts, 1.0)
        assert result[0] is None  # First "Hello"
        assert result[1] is None  # "World"
        assert result[2] == 0  # Second "Hello" is dup of index 0
        assert result[3] is None  # "Test"

    def test_no_duplicates(self) -> None:
        """Should return None for all unique texts."""
        texts = ["A", "B", "C"]
        result = ExactHashDeduplicator._find_duplicates(texts, 1.0)
        assert all(r is None for r in result)

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features."""
        from mloda.user import Options

        assert ExactHashDeduplicator.match_feature_group_criteria("docs__chunked__deduped", Options())
        assert not ExactHashDeduplicator.match_feature_group_criteria("docs__chunked", Options())
