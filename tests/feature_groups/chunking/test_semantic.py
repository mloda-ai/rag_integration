"""Tests for SemanticChunker.

The sentence-transformer model is mocked so these tests stay fast and offline;
they exercise the chunking/grouping logic and the per-feature option wiring.
"""

from __future__ import annotations

from typing import Any, List, Type
from unittest.mock import patch

import numpy as np
from mloda.user import Feature, Options

from rag_integration.feature_groups.rag_pipeline.chunking import SemanticChunker
from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker
from tests.feature_groups.chunking.text_chunking_test_base import TextChunkingTestBase


class _FakeModel:
    """Stand-in for a SentenceTransformer returning fixed embeddings."""

    def __init__(self, vectors: List[List[float]]) -> None:
        self._vectors = vectors

    def encode(self, sentences: List[str]) -> Any:
        return np.array(self._vectors, dtype=np.float32)


class TestSemanticChunker(TextChunkingTestBase):
    """Shared chunker contract tests."""

    @property
    def chunker_class(self) -> Type[BaseChunker]:
        return SemanticChunker


class TestSplitSentences:
    """Tests for _split_sentences."""

    def test_splits_on_boundaries(self) -> None:
        sentences = SemanticChunker._split_sentences("Cats purr softly. Dogs bark loudly.")
        assert sentences == ["Cats purr softly.", "Dogs bark loudly."]

    def test_strips_and_drops_empty(self) -> None:
        assert SemanticChunker._split_sentences("   ") == []


class TestCosineSimilarity:
    """Tests for _cosine_similarity."""

    def test_identical_vectors(self) -> None:
        assert SemanticChunker._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self) -> None:
        assert SemanticChunker._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert SemanticChunker._cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestOptionGetters:
    """Tests for the per-feature option getters (issue: promote hardcoded values)."""

    def test_similarity_threshold_default(self) -> None:
        feature = Feature("docs__chunked", options=Options())
        assert SemanticChunker._get_similarity_threshold(feature) == SemanticChunker.DEFAULT_SIMILARITY_THRESHOLD

    def test_similarity_threshold_from_options(self) -> None:
        feature = Feature("docs__chunked", options=Options(context={"similarity_threshold": 0.9}))
        assert SemanticChunker._get_similarity_threshold(feature) == 0.9

    def test_model_name_default(self) -> None:
        feature = Feature("docs__chunked", options=Options())
        assert SemanticChunker._get_model_name(feature) == SemanticChunker.DEFAULT_MODEL

    def test_model_name_from_options(self) -> None:
        feature = Feature("docs__chunked", options=Options(context={"model_name": "custom-model"}))
        assert SemanticChunker._get_model_name(feature) == "custom-model"


class TestChunkTextForFeature:
    """Tests that _chunk_text_for_feature threads the per-feature options through."""

    def test_passes_options_to_semantic(self) -> None:
        feature = Feature(
            "docs__chunked",
            options=Options(
                context={
                    "chunk_size": 256,
                    "similarity_threshold": 0.7,
                    "model_name": "custom-model",
                }
            ),
        )
        with patch.object(SemanticChunker, "_chunk_text_semantic", return_value=["chunk"]) as mock_semantic:
            result = SemanticChunker._chunk_text_for_feature("some text", feature)

        assert result == ["chunk"]
        mock_semantic.assert_called_once_with("some text", 256, 0.7, "custom-model")


class TestSemanticGrouping:
    """Tests for the semantic grouping logic with a mocked model."""

    def test_topic_shift_splits(self) -> None:
        """Dissimilar consecutive sentences land in separate chunks."""
        fake = _FakeModel([[1.0, 0.0], [0.0, 1.0]])
        with patch.object(SemanticChunker, "_get_model", return_value=fake):
            chunks = SemanticChunker._chunk_text_semantic(
                "Cats purr softly. Dogs bark loudly.",
                chunk_size=1000,
                similarity_threshold=0.5,
                model_name="x",
            )
        assert chunks == ["Cats purr softly.", "Dogs bark loudly."]

    def test_similar_sentences_grouped(self) -> None:
        """Similar consecutive sentences are grouped into one chunk."""
        fake = _FakeModel([[1.0, 0.0], [1.0, 0.0]])
        with patch.object(SemanticChunker, "_get_model", return_value=fake):
            chunks = SemanticChunker._chunk_text_semantic(
                "Cats purr softly. Dogs bark loudly.",
                chunk_size=1000,
                similarity_threshold=0.5,
                model_name="x",
            )
        assert chunks == ["Cats purr softly. Dogs bark loudly."]

    def test_size_limit_forces_split(self) -> None:
        """A small chunk_size splits even similar sentences."""
        fake = _FakeModel([[1.0, 0.0], [1.0, 0.0]])
        with patch.object(SemanticChunker, "_get_model", return_value=fake):
            chunks = SemanticChunker._chunk_text_semantic(
                "Cats purr softly. Dogs bark loudly.",
                chunk_size=5,
                similarity_threshold=0.5,
                model_name="x",
            )
        assert chunks == ["Cats purr softly.", "Dogs bark loudly."]

    def test_single_sentence_no_model_call(self) -> None:
        """A single sentence is returned without invoking the model."""
        with patch.object(SemanticChunker, "_get_model") as mock_model:
            chunks = SemanticChunker._chunk_text_semantic(
                "Only one sentence here.",
                chunk_size=1000,
                similarity_threshold=0.5,
                model_name="x",
            )
        assert chunks == ["Only one sentence here."]
        mock_model.assert_not_called()

    def test_empty_text(self) -> None:
        """Empty text returns a single empty chunk without a model call."""
        with patch.object(SemanticChunker, "_get_model") as mock_model:
            chunks = SemanticChunker._chunk_text_semantic(
                "",
                chunk_size=1000,
                similarity_threshold=0.5,
                model_name="x",
            )
        assert chunks == [""]
        mock_model.assert_not_called()
