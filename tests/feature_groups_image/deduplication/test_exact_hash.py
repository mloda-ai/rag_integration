"""Tests for ExactHashImageDeduplicator."""

from rag_integration.feature_groups.image_pipeline.deduplication import ExactHashImageDeduplicator


class TestExactHashImageDeduplicator:
    """Tests for ExactHashImageDeduplicator."""

    def test_find_exact_duplicates(self) -> None:
        """Should find exact byte-identical duplicates."""
        images = [b"image_a", b"image_b", b"image_a", b"image_c"]
        result = ExactHashImageDeduplicator._find_duplicates(images, 1.0)
        assert result[0] is None  # First "image_a"
        assert result[1] is None  # "image_b"
        assert result[2] == 0  # Second "image_a" is dup of index 0
        assert result[3] is None  # "image_c"

    def test_no_duplicates(self) -> None:
        """Should return None for all unique images."""
        images = [b"img_1", b"img_2", b"img_3"]
        result = ExactHashImageDeduplicator._find_duplicates(images, 1.0)
        assert all(r is None for r in result)

    def test_all_same(self) -> None:
        """Should detect all-same images."""
        images = [b"same", b"same", b"same"]
        result = ExactHashImageDeduplicator._find_duplicates(images, 1.0)
        assert result[0] is None
        assert result[1] == 0
        assert result[2] == 0

    def test_feature_matching_pattern(self) -> None:
        """Should match deduped features."""
        from mloda.user import Options

        assert ExactHashImageDeduplicator.match_feature_group_criteria("image_docs__preprocessed__deduped", Options())
        assert not ExactHashImageDeduplicator.match_feature_group_criteria("image_docs__preprocessed", Options())
