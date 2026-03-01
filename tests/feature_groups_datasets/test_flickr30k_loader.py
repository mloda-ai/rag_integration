"""Tests for Flickr30kDatasetSource."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag_integration.feature_groups.datasets.image.flickr30k import Flickr30kDatasetSource
from mloda.user import Options


def _make_options(data_dir: str, max_samples: int = 2) -> Options:
    return Options(
        context={
            Flickr30kDatasetSource.DATA_DIR: data_dir,
            Flickr30kDatasetSource.MAX_SAMPLES: max_samples,
        }
    )


def _create_fixture_dir() -> tempfile.TemporaryDirectory:  # type: ignore[type-arg]
    """Create a minimal on-disk fixture matching the Flickr30K structure."""
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)

    # Create images dir with a tiny 1x1 JPEG
    images_dir = base / "flickr30k-images"
    images_dir.mkdir()
    try:
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (1, 1), color=(255, 0, 0))
        img.save(images_dir / "test_image.jpg", format="JPEG")
    except ImportError:
        (images_dir / "test_image.jpg").write_bytes(b"fake")

    # Create CSV — inner quotes must be doubled per CSV spec so pandas + ast.literal_eval work
    csv_content = (
        'raw,sentids,split,filename,img_id\n"[""A red square."", ""A tiny image.""]","[0, 1]",test,test_image.jpg,42\n'
    )
    (base / "flickr_annotations_30k.csv").write_text(csv_content)

    return tmp


class TestFlickr30kDatasetSource:
    def test_match_feature_name(self) -> None:
        assert Flickr30kDatasetSource.match_feature_group_criteria("eval_images", Options()) is True
        assert Flickr30kDatasetSource.match_feature_group_criteria("image_docs", Options()) is False

    def test_missing_data_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="data_dir"):
            Flickr30kDatasetSource._load_dataset(Options())

    def test_missing_csv_raises(self) -> None:
        pytest.importorskip("pandas")
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(FileNotFoundError, match="flickr_annotations_30k.csv"):
                Flickr30kDatasetSource._load_dataset(_make_options(tmp))

    def test_loads_corpus_and_query_rows(self) -> None:
        pytest.importorskip("PIL")
        pytest.importorskip("pandas")

        tmp = _create_fixture_dir()
        try:
            rows = Flickr30kDatasetSource._load_dataset(_make_options(tmp.name))
            corpus_rows = [r for r in rows if r["row_type"] == "corpus"]
            query_rows = [r for r in rows if r["row_type"] == "query"]

            assert len(corpus_rows) == 1
            assert len(query_rows) == 2  # 2 captions per image
        finally:
            tmp.cleanup()

    def test_query_row_has_relevant_image_id(self) -> None:
        pytest.importorskip("PIL")
        pytest.importorskip("pandas")

        tmp = _create_fixture_dir()
        try:
            rows = Flickr30kDatasetSource._load_dataset(_make_options(tmp.name))
            query_rows = [r for r in rows if r["row_type"] == "query"]

            for q in query_rows:
                assert "42" in q["relevant_image_ids"]
                assert q["caption"] in ["A red square.", "A tiny image."]
        finally:
            tmp.cleanup()
