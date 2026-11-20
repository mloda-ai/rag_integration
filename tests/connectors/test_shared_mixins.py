"""Shared connector mixins and error types: every error is a ``ValueError``,
``query_text`` / ``top_k`` run values parse uniformly across the families, and
an upstream ``{doc_id, text}`` source is consumed the same way in each family that takes one."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest
from mloda.user import FeatureName, Options

from rag_integration.feature_groups.connectors.errors import (
    ConnectorError,
    DuplicateDocIdError,
    GroundingError,
    InvalidOptionError,
    MissingOptionError,
    RankingContractError,
    SqlSafetyError,
)
from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector
from rag_integration.feature_groups.connectors.generate.extractive_responder import ExtractiveResponder
from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.connectors.mixins import OptionsMixin, TopKMixin
from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.connectors.rerank.lexical_reranker import LexicalReranker
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.tfidf_retriever import TfidfRetriever


class TestErrorHierarchy:
    """Every connector error is a ``ValueError`` so existing callers keep catching it."""

    @pytest.mark.parametrize(
        "error_cls",
        [
            ConnectorError,
            MissingOptionError,
            InvalidOptionError,
            DuplicateDocIdError,
            RankingContractError,
            GroundingError,
            SqlSafetyError,
        ],
    )
    def test_subclasses_value_error(self, error_cls: type[Exception]) -> None:
        assert issubclass(error_cls, ValueError)


_TOPK_FAMILIES = [BaseRetrieveConnector, BaseRerankConnector, BaseGraphRagConnector, BaseOrchestratorConnector]
_QUERY_FAMILIES = [*_TOPK_FAMILIES, BaseGenerateConnector]


def _run(values: dict[str, Any], runtime: bool = True) -> tuple[Any, Any]:
    """A FeatureSet stand-in whose run values are ``values`` (or plain options when not ``runtime``)."""
    feature = MagicMock()
    feature.options = Options(context=values)
    features = MagicMock()
    features.runtime_parameters = ("query_text", "top_k") if runtime else ()
    features.runtime_parameter.side_effect = lambda key, feat: values.get(key)
    return features, feature


class TestSharedTopK:
    """``TopKMixin._run_top_k`` parses uniformly across every family that mixes it in."""

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_garbage_top_k_raises_naming_key_and_value(self, base: type[TopKMixin]) -> None:
        with pytest.raises(InvalidOptionError, match="top_k.*not-an-int"):
            base._run_top_k(*_run({base.TOP_K: "not-an-int"}))

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_non_coercible_type_raises_value_error(self, base: type[TopKMixin]) -> None:
        # A list cannot become an int; before the refactor the non-validating
        # families surfaced a bare TypeError here, now it is a loud ValueError.
        with pytest.raises(ValueError):
            base._run_top_k(*_run({base.TOP_K: [1, 2]}))

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    def test_absent_top_k_falls_back_to_default(self, base: type[TopKMixin]) -> None:
        assert base._run_top_k(*_run({})) == base.DEFAULT_TOP_K

    @pytest.mark.parametrize("base", _TOPK_FAMILIES)
    @pytest.mark.parametrize("runtime", [True, False])
    def test_string_integer_is_coerced(self, base: type[TopKMixin], runtime: bool) -> None:
        assert base._run_top_k(*_run({base.TOP_K: "3"}, runtime)) == 3


class TestSharedRunQuery:
    """``OptionsMixin._run_query`` reads ``query_text`` uniformly across every query-taking family."""

    @pytest.mark.parametrize("base", _QUERY_FAMILIES)
    @pytest.mark.parametrize("runtime", [True, False])
    def test_reads_the_run_value(self, base: type[OptionsMixin], runtime: bool) -> None:
        assert base._run_query(*_run({"query_text": "engine oil"}, runtime), "query_text") == "engine oil"

    @pytest.mark.parametrize("base", _QUERY_FAMILIES)
    def test_missing_query_names_both_sources(self, base: type[OptionsMixin]) -> None:
        with pytest.raises(MissingOptionError, match=r"query_text.*run\(parameters=\.\.\.\)"):
            base._run_query(*_run({}), "query_text")

    @pytest.mark.parametrize("base", _QUERY_FAMILIES)
    def test_non_string_query_raises(self, base: type[OptionsMixin]) -> None:
        with pytest.raises(InvalidOptionError, match="query_text.*must be a string, got int"):
            base._run_query(*_run({"query_text": 42}), "query_text")


_SOURCE = "upstream_docs"
_DOCS: list[dict[str, Any]] = [
    {"doc_id": "d0", "text": "alpha"},
    {"doc_id": "d1", "text": "beta"},
    {"doc_id": "d2", "text": "gamma"},
]
# (connector, backend selector, inline key, source key, empty result)
_SOURCE_FAMILIES = [
    pytest.param(TfidfRetriever, {"retrieve_backend": "tfidf"}, "corpus", "corpus_source", [], id="retrieve"),
    pytest.param(LexicalReranker, {"rerank_backend": "lexical"}, "candidates", "candidates_source", [], id="rerank"),
    pytest.param(
        ExtractiveResponder,
        {"generate_backend": "extractive"},
        "passages",
        "passages_source",
        {"answer": "", "citations": []},
        id="generate",
    ),
]


def _source_options(
    backend: dict[str, str], source_key: str, category: str = "context", extra: dict[str, Any] | None = None
) -> Options:
    values = {**backend, "query_text": "alpha", source_key: _SOURCE, "tenant": "acme", **(extra or {})}
    return Options(context=values) if category == "context" else Options(group=values)


def _feature_set(options: Options) -> Any:
    feature = MagicMock()
    feature.options = options
    features = MagicMock()
    features.features = [feature]
    features.runtime_parameter.side_effect = lambda key, feat: feat.options.get(key)
    features.runtime_parameters = ("query_text", "top_k")
    return features


@pytest.mark.parametrize(("connector", "backend", "inline_key", "source_key", "empty"), _SOURCE_FAMILIES)
class TestSharedDocSource:
    """A ``*_source`` option declares the upstream feature and validates what it produced."""

    def test_no_source_stays_root(
        self, connector: Any, backend: dict[str, str], inline_key: str, source_key: str, empty: Any
    ) -> None:
        assert connector().input_features(Options(context=backend), FeatureName(connector.ROOT_FEATURE_NAME)) is None

    @pytest.mark.parametrize("category", ["context", "group"])
    def test_declares_source_forwarding_options_without_family_keys(
        self, connector: Any, backend: dict[str, str], inline_key: str, source_key: str, empty: Any, category: str
    ) -> None:
        options = _source_options(backend, source_key, category)
        inputs = connector().input_features(options, FeatureName(connector.ROOT_FEATURE_NAME))
        assert inputs is not None
        (feature,) = inputs
        assert str(feature.name) == _SOURCE
        assert getattr(feature.options, category).get("tenant") == "acme"
        assert {source_key, inline_key, "query_text", *backend} <= connector.FAMILY_OPTION_KEYS
        for family_key in connector.FAMILY_OPTION_KEYS:
            assert feature.options.get(family_key) is None
        assert feature.forward_group_exclude == connector.FAMILY_OPTION_KEYS

    def test_source_with_inline_raises(
        self, connector: Any, backend: dict[str, str], inline_key: str, source_key: str, empty: Any
    ) -> None:
        options = _source_options(backend, source_key, extra={inline_key: _DOCS})
        with pytest.raises(InvalidOptionError, match=f"one {inline_key} only"):
            connector.calculate_feature({}, _feature_set(options))

    def test_source_without_upstream_row_raises(
        self, connector: Any, backend: dict[str, str], inline_key: str, source_key: str, empty: Any
    ) -> None:
        with pytest.raises(InvalidOptionError, match="produced no row"):
            connector.calculate_feature({}, _feature_set(_source_options(backend, source_key)))

    @pytest.mark.parametrize(
        ("rows", "detail", "absent"),
        [
            ([{"doc_id": "d0", "text": "alpha"}], re.escape("{'doc_id': 'd0', 'text': 'alpha'}"), None),
            ([{"doc_id": "d0", "text": "z" * 500}], re.escape("..."), "z" * 100),
            ([_DOCS[:1], _DOCS[1:]], "got 2 rows", "alpha"),
            ([["alpha", "beta"]], re.escape("['alpha', 'beta']"), None),
        ],
        ids=["non_list_payload", "large_payload_truncated", "multiple_rows", "non_dict_entries"],
    )
    def test_source_not_one_list_row_raises(
        self,
        connector: Any,
        backend: dict[str, str],
        inline_key: str,
        source_key: str,
        empty: Any,
        rows: list[Any],
        detail: str,
        absent: str | None,
    ) -> None:
        options = _source_options(backend, source_key)
        with pytest.raises(InvalidOptionError, match="single row holding a list") as exc_info:
            connector.calculate_feature({_SOURCE: rows}, _feature_set(options))
        assert re.search(detail, str(exc_info.value))
        assert absent is None or absent not in str(exc_info.value)

    def test_source_with_empty_list_returns_empty(
        self, connector: Any, backend: dict[str, str], inline_key: str, source_key: str, empty: Any
    ) -> None:
        result = connector.calculate_feature({_SOURCE: [[]]}, _feature_set(_source_options(backend, source_key)))
        assert result == [{connector.ROOT_FEATURE_NAME: empty}]
