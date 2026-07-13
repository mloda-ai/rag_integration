"""Inheritable contract-test suite for the ``retrieve`` connector family.

Mirrors open-kgo's ``KgConnectorContractBase``: a concrete backend's test class
subclasses :class:`RetrieveConnectorContractBase`, implements five small adapter
methods, and inherits the whole body of contract assertions for free.

The base is intentionally NOT named ``Test*`` so pytest does not collect it
directly (it has abstract adapters and no backend). Concrete subclasses named
``Test<Backend>`` are collected and run every assertion below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from unittest.mock import MagicMock

import pytest

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import columnar_to_rows
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector


class RetrieveConnectorContractBase(ABC):
    """Contract every retrieve-connector backend must satisfy."""

    # -- Adapter methods (a concrete test implements these five) --------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> Type[BaseRetrieveConnector]:
        """Return the concrete ``BaseRetrieveConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``retrieve_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def sample_corpus(cls) -> List[Dict[str, Any]]:
        """Return a small corpus of ``{doc_id, text}`` dicts.

        Craft it so the query has one determinate best match. For a lexical
        backend the query must share literal tokens with the intended top doc,
        and the distractors must share none.
        """

    @classmethod
    @abstractmethod
    def sample_query(cls) -> str:
        """Return a query whose best match in ``sample_corpus`` is determinate."""

    @classmethod
    @abstractmethod
    def expected_top_doc_id(cls) -> str:
        """Return the ``doc_id`` that must rank first for ``sample_query``."""

    # -- Default fixtures (overridable, shared by all backends) ----------------

    @classmethod
    def matching_query(cls) -> str:
        """Return a query that every doc in :meth:`matching_corpus` matches."""
        return "zebra"

    @classmethod
    def matching_corpus(cls) -> List[Dict[str, Any]]:
        """Return a corpus in which every doc positively matches
        :meth:`matching_query`.

        Every doc shares the literal token ``zebra``, so any lexical or
        vector-space backend scores all of them positively. Tests about corpus
        coverage and the default ``top_k`` use this fixture because the family
        returns only positively scoring passages, which makes the regular
        distractor-heavy ``sample_corpus`` unsuitable for them. The corpus is
        deliberately larger than ``DEFAULT_TOP_K``.
        """
        return [{"doc_id": f"m{i}", "text": f"zebra fact number {i} from the zebra herd"} for i in range(7)]

    # -- Helpers --------------------------------------------------------------

    @classmethod
    def _retrieve(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return cls.connector_class()._retrieve(query, corpus, top_k)

    @classmethod
    def _options(cls, query: str, corpus: List[Dict[str, Any]], top_k: Optional[int]) -> Options:
        """Build the family Options; ``top_k=None`` omits the key (default applies)."""
        connector = cls.connector_class()
        context: Dict[str, Any] = {
            connector.RETRIEVE_BACKEND: cls.backend_value(),
            connector.QUERY_TEXT: query,
            connector.CORPUS: corpus,
        }
        if top_k is not None:
            context[connector.TOP_K] = top_k
        return Options(context=context)

    @classmethod
    def _feature_set(cls, options: Options) -> Any:
        """Build a minimal FeatureSet stand-in holding one feature with ``options``."""
        feature = MagicMock()
        feature.options = options
        features = MagicMock()
        features.features = [feature]
        return features

    @classmethod
    def _run_all(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        connector = cls.connector_class()
        feature = Feature(connector.ROOT_FEATURE_NAME, options=cls._options(query, corpus, top_k))
        result = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        for partition in result:
            for row in columnar_to_rows(partition):
                if connector.ROOT_FEATURE_NAME in row:
                    passages: List[Dict[str, Any]] = row[connector.ROOT_FEATURE_NAME]
                    return passages
        raise AssertionError(f"run_all returned no '{connector.ROOT_FEATURE_NAME}' row: {result!r}")

    # -- Matching / honest surface --------------------------------------------

    def test_matches_root_feature_for_declared_backend(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.RETRIEVE_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is True

    def test_does_not_match_other_feature_name(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.RETRIEVE_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria("docs", opts) is False

    def test_unknown_backend_does_not_match(self) -> None:
        """Honest surface: the connector does not claim a backend it cannot serve."""
        connector = self.connector_class()
        opts = Options(context={connector.RETRIEVE_BACKEND: "definitely_not_a_backend_xyz"})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is False

    def test_backend_declared_in_supported_set(self) -> None:
        connector = self.connector_class()
        assert self.backend_value() in connector.RETRIEVE_BACKENDS

    # -- Output contract ------------------------------------------------------

    def test_returns_ranked_passage_shape(self) -> None:
        passages = self._retrieve(self.sample_query(), self.sample_corpus(), top_k=len(self.sample_corpus()))
        assert isinstance(passages, list)
        assert passages, "canonical query returned no passages; contract assertions would be vacuous"
        for passage in passages:
            assert set(passage) >= {"doc_id", "text", "score", "rank"}
            assert isinstance(passage["doc_id"], str)
            assert isinstance(passage["text"], str)
            assert isinstance(passage["score"], float)
            assert isinstance(passage["rank"], int)

    def test_scores_non_increasing_and_ranks_ascending(self) -> None:
        passages = self._retrieve(self.sample_query(), self.sample_corpus(), top_k=len(self.sample_corpus()))
        ranks = [p["rank"] for p in passages]
        scores = [p["score"] for p in passages]
        assert ranks == list(range(len(passages)))
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranked_first(self) -> None:
        """Not-a-stub proof: the crafted relevant doc must rank #1, by a margin.

        The strict ``score`` separation rules out a backend that ignores the
        query and happens to return the corpus in an order (e.g. alphabetical)
        that puts the expected doc first, and a backend that returns all-equal
        scores.
        """
        passages = self._retrieve(self.sample_query(), self.sample_corpus(), top_k=len(self.sample_corpus()))
        assert passages[0]["doc_id"] == self.expected_top_doc_id()
        assert len(passages) >= 2, "contract corpus must have >=2 docs to prove score separation"
        assert passages[0]["score"] > passages[1]["score"]

    def test_passage_text_matches_corpus(self) -> None:
        """Guards the base assembly: each passage's text is the corpus text for
        its doc_id (catches a doc_id<->text pairing regression in the base, not
        a wrong-index backend, which ``test_relevant_doc_ranked_first`` covers)."""
        corpus = self.sample_corpus()
        text_by_doc_id = {str(doc["doc_id"]): str(doc["text"]) for doc in corpus}
        passages = self._retrieve(self.sample_query(), corpus, top_k=len(corpus))
        for passage in passages:
            assert passage["text"] == text_by_doc_id[passage["doc_id"]]

    def test_doc_ids_unique_and_cover_corpus(self) -> None:
        """No silent drop or duplicate: with an all-matching corpus and
        ``top_k >= corpus size`` the returned doc_ids are unique and cover
        exactly the corpus.

        A backend that drops or duplicates a doc (e.g. returns ``[(2,..),(2,..),
        (0,..)]``) passes ranking/score/mapping checks but fails here. This is
        the assertion that keeps the silent-corruption class out of every
        sibling family that copies this suite. Uses ``matching_corpus`` because
        the family drops zero-scored passages, so only a corpus where every doc
        matches can be covered in full.
        """
        corpus = self.matching_corpus()
        passages = self._retrieve(self.matching_query(), corpus, top_k=len(corpus))
        returned = [p["doc_id"] for p in passages]
        assert len(returned) == len(set(returned)), "ranking returned duplicate doc_ids"
        assert set(returned) == {str(doc["doc_id"]) for doc in corpus}

    def test_top_k_respected(self) -> None:
        passages = self._retrieve(self.sample_query(), self.sample_corpus(), top_k=1)
        assert len(passages) == 1
        assert passages[0]["doc_id"] == self.expected_top_doc_id()

    def test_top_k_clamped_to_corpus(self) -> None:
        corpus = self.matching_corpus()
        passages = self._retrieve(self.matching_query(), corpus, top_k=len(corpus) + 50)
        assert len(passages) == len(corpus)

    def test_top_k_zero_returns_empty(self) -> None:
        assert self._retrieve(self.sample_query(), self.sample_corpus(), top_k=0) == []

    def test_top_k_negative_returns_empty(self) -> None:
        assert self._retrieve(self.sample_query(), self.sample_corpus(), top_k=-3) == []

    def test_default_top_k_is_five(self) -> None:
        """With ``top_k`` absent from the options, ``DEFAULT_TOP_K`` (5) applies.

        The fixture proves it: ``matching_corpus`` has more than 5 docs that
        all positively match, so exactly 5 coming back can only be the default
        at work (the only-positive-scores rule cannot have trimmed the list).
        """
        connector = self.connector_class()
        corpus = self.matching_corpus()
        assert len(corpus) > connector.DEFAULT_TOP_K, "fixture must exceed the default to prove the cut"
        options = self._options(self.matching_query(), corpus, top_k=None)
        result = connector.calculate_feature([], self._feature_set(options))
        passages = result[0][connector.ROOT_FEATURE_NAME]
        assert len(passages) == connector.DEFAULT_TOP_K

    def test_empty_corpus_returns_empty(self) -> None:
        assert self._retrieve(self.sample_query(), [], top_k=5) == []

    def test_degenerate_query_returns_empty(self) -> None:
        """Family rule: only positively scoring passages are returned, so a
        query sharing no terms with the corpus yields no passages (instead of
        ``top_k`` arbitrary zero-scored ones)."""
        assert self._retrieve("zzzz qqqq", self.sample_corpus(), top_k=3) == []

    def test_missing_query_text_raises(self) -> None:
        connector = self.connector_class()
        options = Options(
            context={
                connector.RETRIEVE_BACKEND: self.backend_value(),
                connector.CORPUS: self.sample_corpus(),
            }
        )
        with pytest.raises(ValueError, match=connector.QUERY_TEXT):
            connector.calculate_feature([], self._feature_set(options))

    def test_missing_corpus_raises(self) -> None:
        connector = self.connector_class()
        options = Options(
            context={
                connector.RETRIEVE_BACKEND: self.backend_value(),
                connector.QUERY_TEXT: self.sample_query(),
            }
        )
        with pytest.raises(ValueError, match=connector.CORPUS):
            connector.calculate_feature([], self._feature_set(options))

    def test_idempotent(self) -> None:
        corpus = self.sample_corpus()
        first = self._retrieve(self.sample_query(), corpus, top_k=len(corpus))
        second = self._retrieve(self.sample_query(), corpus, top_k=len(corpus))
        assert first == second

    # -- End to end -----------------------------------------------------------

    def test_end_to_end_run_all(self) -> None:
        corpus = self.sample_corpus()
        passages = self._run_all(self.sample_query(), corpus, top_k=len(corpus))
        assert passages, "run_all produced no passages"
        assert passages[0]["doc_id"] == self.expected_top_doc_id()
