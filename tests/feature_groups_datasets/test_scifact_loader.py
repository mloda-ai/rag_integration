"""Tests for ScifactDatasetSource."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_integration.feature_groups.datasets.text.scifact import ScifactDatasetSource
from mloda.user import Options


def _make_options(data_dir: str = "/fake/scifact") -> Options:
    return Options(context={ScifactDatasetSource.DATA_DIR: data_dir})


def _mock_beir_loader() -> Any:
    """Return a mock GenericDataLoader that yields tiny fixture data."""
    corpus = {
        "4983": {"title": "Antigen study", "text": "Antigens regulate protein expression."},
        "5129": {"title": "Cell biology", "text": "Cells divide through mitosis."},
    }
    queries = {
        "0": "Antigens adjust protein expression in lymphocytes.",
        "1": "Cells do not divide.",
    }
    qrels = {
        "0": {"4983": 1},
        "1": {"5129": 0},
    }
    loader_instance = MagicMock()
    loader_instance.load.return_value = (corpus, queries, qrels)
    return loader_instance


class TestScifactDatasetSource:
    def test_match_feature_name(self) -> None:
        assert ScifactDatasetSource.match_feature_group_criteria("eval_docs", Options()) is True
        assert ScifactDatasetSource.match_feature_group_criteria("docs", Options()) is False
        assert ScifactDatasetSource.match_feature_group_criteria("eval_images", Options()) is False

    def test_missing_data_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="data_dir"):
            ScifactDatasetSource._load_dataset(Options())

    def test_loads_corpus_and_query_rows(self) -> None:
        pytest.importorskip("beir")
        with patch("beir.datasets.data_loader.GenericDataLoader") as MockLoader:
            MockLoader.return_value = _mock_beir_loader()
            rows = ScifactDatasetSource._load_dataset(_make_options())

        corpus_rows = [r for r in rows if r["row_type"] == "corpus"]
        query_rows = [r for r in rows if r["row_type"] == "query"]

        assert len(corpus_rows) == 2
        assert len(query_rows) == 2

    def test_corpus_row_format(self) -> None:
        pytest.importorskip("beir")
        with patch("beir.datasets.data_loader.GenericDataLoader") as MockLoader:
            MockLoader.return_value = _mock_beir_loader()
            rows = ScifactDatasetSource._load_dataset(_make_options())

        corpus_rows = [r for r in rows if r["row_type"] == "corpus"]
        for row in corpus_rows:
            assert "doc_id" in row
            assert "text" in row
            assert row["row_type"] == "corpus"

    def test_query_row_has_relevant_ids(self) -> None:
        pytest.importorskip("beir")
        with patch("beir.datasets.data_loader.GenericDataLoader") as MockLoader:
            MockLoader.return_value = _mock_beir_loader()
            rows = ScifactDatasetSource._load_dataset(_make_options())

        query_rows = [r for r in rows if r["row_type"] == "query"]
        q0 = next(r for r in query_rows if r["doc_id"] == "0")
        assert "4983" in q0["relevant_doc_ids"]
        assert q0["relevance_scores"]["4983"] == 1
