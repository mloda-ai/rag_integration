"""Shared connector mixins and error types: every error is a ``ValueError``, and
``query_text`` / ``top_k`` run values parse uniformly across the families."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from mloda.user import Options

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
from rag_integration.feature_groups.connectors.graph_rag.base import BaseGraphRagConnector
from rag_integration.feature_groups.connectors.mixins import OptionsMixin, TopKMixin
from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.rerank.base import BaseRerankConnector
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector


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
