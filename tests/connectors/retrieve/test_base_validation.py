"""Negative tests for the base retrieve-connector validation.

A deliberately misbehaving stub backend returns whatever ``(index, score)``
pairs a test injects, proving that every ``_rank`` requirement the base
documents (indices in range, indices unique, at most ``top_k`` pairs, scores
non-increasing) fails loudly in ``_validate_ranking`` instead of silently
corrupting the passage list. The corpus validation in ``_retrieve``, the
one-feature-per-run limit of ``calculate_feature``, the ``top_k`` option
parsing, and the ``corpus_source`` declaration and validation are covered
here too: they are base behavior, not per-backend behavior, so they live
outside the inheritable contract suite.
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from mloda.user import FeatureName, Options

from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector


def _stub_returning(pairs: list[tuple[int, float]]) -> type[BaseRetrieveConnector]:
    """Build a stub backend whose ``_rank`` returns ``pairs`` verbatim."""

    class _MisbehavingRetriever(BaseRetrieveConnector):
        RETRIEVE_BACKENDS: ClassVar = {"misbehaving_stub": "Deliberately misbehaving _rank for validation tests"}

        @classmethod
        def _rank(cls, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
            return pairs

    return _MisbehavingRetriever


def _corpus() -> list[dict[str, Any]]:
    return [
        {"doc_id": "d0", "text": "alpha"},
        {"doc_id": "d1", "text": "beta"},
        {"doc_id": "d2", "text": "gamma"},
    ]


class TestRankValidation:
    """Each documented ``_rank`` requirement is enforced, not just trusted."""

    def test_duplicate_indices_raise(self) -> None:
        stub = _stub_returning([(0, 1.0), (0, 0.5)])
        with pytest.raises(ValueError, match="duplicate index"):
            stub._retrieve("alpha", _corpus(), top_k=2)

    def test_out_of_range_index_raises(self) -> None:
        stub = _stub_returning([(99, 1.0)])
        with pytest.raises(ValueError, match="out-of-range"):
            stub._retrieve("alpha", _corpus(), top_k=2)

    def test_increasing_scores_raise(self) -> None:
        stub = _stub_returning([(0, 0.1), (1, 0.9)])
        with pytest.raises(ValueError, match="non-increasing"):
            stub._retrieve("alpha", _corpus(), top_k=2)

    def test_more_pairs_than_top_k_raise(self) -> None:
        stub = _stub_returning([(0, 0.9), (1, 0.5)])
        with pytest.raises(ValueError, match="pairs for top_k"):
            stub._retrieve("alpha", _corpus(), top_k=1)


class TestCorpusValidation:
    """Malformed corpora raise a clear ValueError instead of an AttributeError."""

    def test_non_dict_corpus_entry_raises(self) -> None:
        stub = _stub_returning([(0, 1.0)])
        corpus: list[Any] = [{"doc_id": "d0", "text": "alpha"}, "not-a-dict"]
        with pytest.raises(ValueError, match="not a dict"):
            stub._retrieve("alpha", corpus, top_k=1)

    def test_duplicate_effective_doc_ids_raise(self) -> None:
        # "1" collides with the positional-index fallback of the second entry,
        # which has no doc_id and sits at index 1.
        stub = _stub_returning([(0, 1.0)])
        corpus: list[dict[str, Any]] = [{"doc_id": "1", "text": "alpha"}, {"text": "beta"}]
        with pytest.raises(ValueError, match="duplicate doc_id"):
            stub._retrieve("alpha", corpus, top_k=1)


class TestCalculateFeatureLimits:
    """The family answers one query per run; a multi-feature set raises."""

    def test_more_than_one_feature_raises(self) -> None:
        stub = _stub_returning([(0, 1.0)])
        features = MagicMock()
        features.features = [MagicMock(), MagicMock()]
        with pytest.raises(ValueError, match="one query per run"):
            stub.calculate_feature([], features)


class TestTopKParsing:
    """A non-integer ``top_k`` option raises naming the key and the value."""

    def test_garbage_top_k_raises(self) -> None:
        options = Options(context={BaseRetrieveConnector.TOP_K: "not-an-int"})
        with pytest.raises(ValueError, match="top_k.*not-an-int"):
            BaseRetrieveConnector._get_top_k(options)


class TestCorpusSource:
    """``corpus_source`` declares the upstream feature and validates what it produced."""

    _SOURCE = "scoped_corpus"

    @classmethod
    def _context(cls, **extra: Any) -> dict[str, Any]:
        return {
            BaseRetrieveConnector.RETRIEVE_BACKEND: "misbehaving_stub",
            BaseRetrieveConnector.QUERY_TEXT: "alpha",
            BaseRetrieveConnector.CORPUS_SOURCE: cls._SOURCE,
            "tenant": "acme",
            **extra,
        }

    @staticmethod
    def _feature_set(options: Options) -> Any:
        feature = MagicMock()
        feature.options = options
        features = MagicMock()
        features.features = [feature]
        return features

    def test_no_corpus_source_stays_root(self) -> None:
        stub = _stub_returning([])
        options = Options(context={BaseRetrieveConnector.RETRIEVE_BACKEND: "misbehaving_stub"})
        assert stub().input_features(options, FeatureName(stub.ROOT_FEATURE_NAME)) is None

    @pytest.mark.parametrize("category", ["context", "group"])
    def test_declares_source_forwarding_options_without_family_keys(self, category: str) -> None:
        stub = _stub_returning([])
        options = Options(context=self._context()) if category == "context" else Options(group=self._context())
        inputs = stub().input_features(options, FeatureName(stub.ROOT_FEATURE_NAME))
        assert inputs is not None
        (feature,) = inputs
        assert str(feature.name) == self._SOURCE
        assert getattr(feature.options, category).get("tenant") == "acme"
        for family_key in stub.FAMILY_OPTION_KEYS:
            assert feature.options.get(family_key) is None
        assert feature.forward_group_exclude == stub.FAMILY_OPTION_KEYS

    def test_source_with_inline_corpus_raises(self) -> None:
        stub = _stub_returning([(0, 1.0)])
        options = Options(context=self._context(**{BaseRetrieveConnector.CORPUS: _corpus()}))
        with pytest.raises(ValueError, match="one corpus only"):
            stub.calculate_feature({}, self._feature_set(options))

    def test_source_without_upstream_row_raises(self) -> None:
        stub = _stub_returning([(0, 1.0)])
        with pytest.raises(ValueError, match="produced no row"):
            stub.calculate_feature({}, self._feature_set(Options(context=self._context())))

    @pytest.mark.parametrize(
        "rows",
        [[{"doc_id": "d0", "text": "alpha"}], [_corpus()[:1], _corpus()[1:]]],
        ids=["non_list_payload", "multiple_rows"],
    )
    def test_source_not_one_list_row_raises(self, rows: list[Any]) -> None:
        stub = _stub_returning([(0, 1.0)])
        data = {self._SOURCE: rows}
        with pytest.raises(ValueError, match="single row holding a list"):
            stub.calculate_feature(data, self._feature_set(Options(context=self._context())))

    def test_source_with_empty_corpus_returns_no_passages(self) -> None:
        stub = _stub_returning([])
        data: dict[str, Any] = {self._SOURCE: [[]]}
        result = stub.calculate_feature(data, self._feature_set(Options(context=self._context())))
        assert result == [{stub.ROOT_FEATURE_NAME: []}]
