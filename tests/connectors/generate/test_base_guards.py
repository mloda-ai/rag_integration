"""Negative-path tests for the ``BaseGenerateConnector`` grounding guards.

The contract suite proves well-behaved backends pass; these tests prove a
misbehaving backend cannot. Each deliberately broken ``_generate`` stub trips
exactly one guard (hallucinated citation, uncited answer, citations without an
answer, duplicate citations), and the option guards (missing ``query_text``,
missing ``passages``, duplicate passage doc_ids) are exercised directly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from mloda.user import Options

from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector

_PASSAGES: list[dict[str, Any]] = [
    {"doc_id": "d0", "text": "Alpha fact."},
    {"doc_id": "d1", "text": "Beta fact."},
]


class _UnknownCitationStub(BaseGenerateConnector):
    """Misbehaving backend: cites a doc_id that was never supplied."""

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return "Alpha fact.", ["not_a_supplied_doc"]


class _UncitedAnswerStub(BaseGenerateConnector):
    """Misbehaving backend: non-empty answer with no citations."""

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return "Alpha fact.", []


class _CitationsWithoutAnswerStub(BaseGenerateConnector):
    """Misbehaving backend: citations attached to a whitespace-only answer."""

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return "   ", ["d0"]


class _DuplicateCitationsStub(BaseGenerateConnector):
    """Misbehaving backend: cites the same passage twice."""

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return "Alpha fact.", ["d0", "d0"]


class _WhitespaceAnswerStub(BaseGenerateConnector):
    """Degenerate but legal backend: whitespace-only answer, no citations."""

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return " \n\t ", []


class _WellBehavedStub(BaseGenerateConnector):
    """Minimal correct backend, used to exercise the option guards."""

    @classmethod
    def _generate(cls, query: str, passages: list[dict[str, Any]]) -> tuple[str, list[str]]:
        return "Alpha fact.", ["d0"]


def _feature_set(context: dict[str, Any]) -> Any:
    """Build a minimal FeatureSet stand-in with a single feature and the given options."""
    feature = MagicMock()
    feature.options = Options(context=context)
    features = MagicMock()
    features.features = [feature]
    return features


class TestGenerateGuards:
    """Each guard rejects its misbehaving backend with a ValueError."""

    def test_unknown_citation_raises(self) -> None:
        with pytest.raises(ValueError, match="not among the supplied passages"):
            _UnknownCitationStub._answer("query", _PASSAGES)

    def test_nonempty_answer_without_citations_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty answer with no citations"):
            _UncitedAnswerStub._answer("query", _PASSAGES)

    def test_citations_without_answer_raises(self) -> None:
        with pytest.raises(ValueError, match="citations with an empty answer"):
            _CitationsWithoutAnswerStub._answer("query", _PASSAGES)

    def test_duplicate_citations_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate citations"):
            _DuplicateCitationsStub._answer("query", _PASSAGES)

    def test_whitespace_answer_normalized_to_canonical_empty_shape(self) -> None:
        result = _WhitespaceAnswerStub._answer("query", _PASSAGES)
        assert result == {"answer": "", "citations": []}


class TestGenerateOptionGuards:
    """Missing or malformed options are rejected with a ValueError."""

    def test_missing_query_text_raises(self) -> None:
        features = _feature_set(
            {
                BaseGenerateConnector.GENERATE_BACKEND: "stub",
                BaseGenerateConnector.PASSAGES: _PASSAGES,
            }
        )
        with pytest.raises(ValueError, match=BaseGenerateConnector.QUERY_TEXT):
            _WellBehavedStub.calculate_feature(None, features)

    def test_missing_passages_raises(self) -> None:
        features = _feature_set(
            {
                BaseGenerateConnector.GENERATE_BACKEND: "stub",
                BaseGenerateConnector.QUERY_TEXT: "query",
            }
        )
        with pytest.raises(ValueError, match=BaseGenerateConnector.PASSAGES):
            _WellBehavedStub.calculate_feature(None, features)

    def test_duplicate_explicit_doc_ids_raise(self) -> None:
        options = Options(
            context={
                BaseGenerateConnector.PASSAGES: [
                    {"doc_id": "d0", "text": "Alpha."},
                    {"doc_id": "d0", "text": "Beta."},
                ]
            }
        )
        with pytest.raises(ValueError, match="duplicate passage doc_id 'd0'"):
            _WellBehavedStub._get_passages(options)

    def test_explicit_doc_id_colliding_with_positional_fallback_raises(self) -> None:
        # The first passage falls back to its positional index "0"; the second
        # explicitly claims "0" (after str() coercion), so they collide.
        options = Options(
            context={
                BaseGenerateConnector.PASSAGES: [
                    {"text": "Alpha."},
                    {"doc_id": 0, "text": "Beta."},
                ]
            }
        )
        with pytest.raises(ValueError, match="duplicate passage doc_id '0'"):
            _WellBehavedStub._get_passages(options)
