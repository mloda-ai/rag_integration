"""Inheritable contract-test suite for the ``generate`` connector family.

A concrete backend's test implements six adapter methods and inherits every
assertion. The base is not named ``Test*`` so pytest does not collect it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import columnar_to_rows
from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector


class GenerateConnectorContractBase(ABC):
    """Contract every generate-connector backend must satisfy."""

    # -- Adapter methods ------------------------------------------------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> Type[BaseGenerateConnector]:
        """Return the concrete ``BaseGenerateConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``generate_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def sample_passages(cls) -> List[Dict[str, Any]]:
        """Return supporting passages (``{doc_id, text}``) with at least one clearly relevant doc.

        ``expected_citation_doc_id`` names the one that must be cited; a backend
        whose distinguishing behaviour is multi-passage citation may make several
        passages relevant (the contract only requires that doc to be among them).
        """

    @classmethod
    @abstractmethod
    def sample_query(cls) -> str:
        """Return a query answerable from ``sample_passages`` with a determinate
        ``expected_citation_doc_id`` (other passages may also be relevant)."""

    @classmethod
    @abstractmethod
    def expected_citation_doc_id(cls) -> str:
        """Return the ``doc_id`` that must be cited for ``sample_query``."""

    @classmethod
    @abstractmethod
    def expected_answer_substring(cls) -> str:
        """Return a distinctive substring of the relevant passage that the
        grounded answer must contain (proves the answer is drawn from the
        passage, not invented)."""

    # -- Helpers --------------------------------------------------------------

    @classmethod
    def _answer(cls, query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        return cls.connector_class()._answer(query, passages)

    @classmethod
    def _run_all(cls, query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        connector = cls.connector_class()
        feature = Feature(
            connector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    connector.GENERATE_BACKEND: cls.backend_value(),
                    connector.QUERY_TEXT: query,
                    connector.PASSAGES: passages,
                }
            ),
        )
        result = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        for partition in result:
            for row in columnar_to_rows(partition):
                if connector.ROOT_FEATURE_NAME in row:
                    answer: Dict[str, Any] = row[connector.ROOT_FEATURE_NAME]
                    return answer
        raise AssertionError(f"run_all returned no '{connector.ROOT_FEATURE_NAME}' row: {result!r}")

    # -- Matching / honest surface --------------------------------------------

    def test_matches_root_feature_for_declared_backend(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.GENERATE_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is True

    def test_does_not_match_other_feature_name(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.GENERATE_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria("docs", opts) is False

    def test_unknown_backend_does_not_match(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.GENERATE_BACKEND: "definitely_not_a_backend_xyz"})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is False

    def test_backend_declared_in_supported_set(self) -> None:
        connector = self.connector_class()
        assert self.backend_value() in connector.GENERATE_BACKENDS

    # -- Output contract ------------------------------------------------------

    def test_answer_object_shape(self) -> None:
        result = self._answer(self.sample_query(), self.sample_passages())
        assert set(result) >= {"answer", "citations"}
        assert isinstance(result["answer"], str)
        assert isinstance(result["citations"], list)
        assert all(isinstance(c, str) for c in result["citations"])

    def test_answer_nonempty_for_canonical_query(self) -> None:
        result = self._answer(self.sample_query(), self.sample_passages())
        assert result["answer"].strip(), "canonical query produced an empty answer; assertions would be vacuous"

    def test_citations_are_grounded(self) -> None:
        """Every citation is one of the supplied passage doc_ids (no invented sources).

        Uses the same positional-fallback rule as the base validator so a
        passage without an explicit ``doc_id`` does not crash this assertion.
        """
        passages = self.sample_passages()
        known = {str(p.get("doc_id", str(i))) for i, p in enumerate(passages)}
        result = self._answer(self.sample_query(), passages)
        assert set(result["citations"]) <= known

    def test_nonempty_answer_is_cited(self) -> None:
        """Grounded by construction: the canonical query yields a non-empty
        answer, and a non-empty answer must cite >=1 passage."""
        result = self._answer(self.sample_query(), self.sample_passages())
        assert result["answer"].strip(), "canonical query produced an empty answer"
        assert result["citations"], "non-empty answer returned no citations"

    def test_relevant_passage_cited(self) -> None:
        """Not-a-stub proof: the relevant passage is cited."""
        result = self._answer(self.sample_query(), self.sample_passages())
        assert self.expected_citation_doc_id() in result["citations"]

    def test_answer_grounded_in_passage(self) -> None:
        """Not-a-stub proof: the answer contains a distinctive substring of the
        relevant passage, so it is drawn from the source rather than invented."""
        passages = self.sample_passages()
        cited = [p for i, p in enumerate(passages) if str(p.get("doc_id", str(i))) == self.expected_citation_doc_id()]
        assert cited, "expected_citation_doc_id is not among sample_passages"
        assert self.expected_answer_substring() in str(cited[0].get("text", "")), (
            "expected_answer_substring must occur in the expected-citation passage's text"
        )
        result = self._answer(self.sample_query(), passages)
        assert self.expected_answer_substring() in result["answer"]

    def test_empty_passages_returns_empty(self) -> None:
        result = self._answer(self.sample_query(), [])
        assert result == {"answer": "", "citations": []}

    def test_idempotent(self) -> None:
        passages = self.sample_passages()
        first = self._answer(self.sample_query(), passages)
        second = self._answer(self.sample_query(), passages)
        assert first == second

    # -- End to end -----------------------------------------------------------

    def test_end_to_end_run_all(self) -> None:
        result = self._run_all(self.sample_query(), self.sample_passages())
        assert self.expected_citation_doc_id() in result["citations"]
        assert self.expected_answer_substring() in result["answer"]
