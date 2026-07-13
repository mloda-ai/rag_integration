"""Inheritable contract-test suite for the ``orchestrator`` connector family.

A concrete backend's test implements six adapter methods and inherits every
assertion. The base is not named ``Test*`` so pytest does not collect it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import pytest
from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.rows import as_rows


class OrchestratorConnectorContractBase(ABC):
    """Contract every orchestrator backend must satisfy."""

    # -- Adapter methods ------------------------------------------------------

    @classmethod
    @abstractmethod
    def connector_class(cls) -> Type[BaseOrchestratorConnector]:
        """Return the concrete ``BaseOrchestratorConnector`` subclass under test."""

    @classmethod
    @abstractmethod
    def backend_value(cls) -> str:
        """Return the ``orchestrator_backend`` value that selects this concrete."""

    @classmethod
    @abstractmethod
    def sample_corpus(cls) -> List[Dict[str, Any]]:
        """Return a corpus (``{doc_id, text}``) with one clearly relevant doc."""

    @classmethod
    @abstractmethod
    def sample_query(cls) -> str:
        """Return a query whose best match is determinate."""

    @classmethod
    @abstractmethod
    def expected_top_doc_id(cls) -> str:
        """Return the doc_id the pipeline must surface first."""

    @classmethod
    @abstractmethod
    def expected_answer_substring(cls) -> str:
        """Return a distinctive substring the answer must contain (drawn from the top doc)."""

    # -- Helpers --------------------------------------------------------------

    @classmethod
    def _answer(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        return cls.connector_class()._answer(query, corpus, top_k)

    @classmethod
    def _run_all(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        connector = cls.connector_class()
        feature = Feature(
            connector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    connector.ORCHESTRATOR_BACKEND: cls.backend_value(),
                    connector.QUERY_TEXT: query,
                    connector.CORPUS: corpus,
                    connector.TOP_K: top_k,
                }
            ),
        )
        result = mlodaAPI.run_all(
            [feature],
            compute_frameworks={PythonDictFramework},
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        for partition in result:
            for row in as_rows(partition):
                if connector.ROOT_FEATURE_NAME in row:
                    answer: Dict[str, Any] = row[connector.ROOT_FEATURE_NAME]
                    return answer
        raise AssertionError(f"run_all returned no '{connector.ROOT_FEATURE_NAME}' row: {result!r}")

    # -- Matching / honest surface --------------------------------------------

    def test_matches_root_feature_for_declared_backend(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.ORCHESTRATOR_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is True

    def test_does_not_match_other_feature_name(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.ORCHESTRATOR_BACKEND: self.backend_value()})
        assert connector.match_feature_group_criteria("docs", opts) is False

    def test_unknown_backend_does_not_match(self) -> None:
        connector = self.connector_class()
        opts = Options(context={connector.ORCHESTRATOR_BACKEND: "definitely_not_a_backend_xyz"})
        assert connector.match_feature_group_criteria(connector.ROOT_FEATURE_NAME, opts) is False

    def test_backend_declared_in_supported_set(self) -> None:
        connector = self.connector_class()
        assert self.backend_value() in connector.ORCHESTRATOR_BACKENDS

    # -- Output contract ------------------------------------------------------

    def test_answer_object_shape(self) -> None:
        result = self._answer(self.sample_query(), self.sample_corpus(), top_k=len(self.sample_corpus()))
        assert set(result) >= {"answer", "documents"}
        assert isinstance(result["answer"], str)
        assert isinstance(result["documents"], list)
        for document in result["documents"]:
            assert set(document) >= {"doc_id", "text", "score"}
            assert isinstance(document["doc_id"], str)
            assert isinstance(document["text"], str)
            assert isinstance(document["score"], float)

    def test_documents_are_grounded(self) -> None:
        """Every surfaced document came from the supplied corpus (no fabrication)."""
        corpus = self.sample_corpus()
        known = {str(doc["doc_id"]) for doc in corpus}
        result = self._answer(self.sample_query(), corpus, top_k=len(corpus))
        assert {document["doc_id"] for document in result["documents"]} <= known

    def test_relevant_doc_surfaced_first(self) -> None:
        """Not-a-stub proof: the pipeline ranks the relevant doc first."""
        result = self._answer(self.sample_query(), self.sample_corpus(), top_k=len(self.sample_corpus()))
        assert result["documents"], "pipeline surfaced no documents"
        assert result["documents"][0]["doc_id"] == self.expected_top_doc_id()

    def test_answer_drawn_from_top_document(self) -> None:
        """The answer contains a distinctive substring of the relevant document."""
        result = self._answer(self.sample_query(), self.sample_corpus(), top_k=len(self.sample_corpus()))
        assert self.expected_answer_substring() in result["answer"]

    def test_top_k_respected(self) -> None:
        result = self._answer(self.sample_query(), self.sample_corpus(), top_k=1)
        assert len(result["documents"]) == 1
        assert result["documents"][0]["doc_id"] == self.expected_top_doc_id()

    def test_empty_corpus_returns_empty(self) -> None:
        result = self._answer(self.sample_query(), [], top_k=5)
        assert result == {"answer": "", "documents": []}

    def test_empty_query_returns_empty(self) -> None:
        """An empty/whitespace query yields no answer and no documents (no framework error leak)."""
        result = self._answer("   ", self.sample_corpus(), top_k=len(self.sample_corpus()))
        assert result["answer"] == ""
        assert result["documents"] == []

    def test_nonpositive_top_k_returns_empty(self) -> None:
        """A non-positive top_k yields no answer and no documents (no framework error leak)."""
        result = self._answer(self.sample_query(), self.sample_corpus(), top_k=0)
        assert result["answer"] == ""
        assert result["documents"] == []

    def test_duplicate_doc_id_raises(self) -> None:
        """Duplicate corpus doc_ids are rejected uniformly by the base (no silent dedup)."""
        corpus = [
            {"doc_id": "dup", "text": "first entry"},
            {"doc_id": "dup", "text": "second entry"},
        ]
        with pytest.raises(ValueError, match="duplicate doc_id"):
            self._answer(self.sample_query(), corpus, top_k=2)

    def test_positional_default_doc_id_collision_raises(self) -> None:
        """An entry without doc_id defaults to its index, so it collides with an explicit id '1'."""
        corpus = [
            {"doc_id": "1", "text": "explicit id one"},
            {"text": "no doc_id: defaults to positional index '1'"},
        ]
        with pytest.raises(ValueError, match="duplicate doc_id"):
            self._answer(self.sample_query(), corpus, top_k=2)

    def test_idempotent(self) -> None:
        corpus = self.sample_corpus()
        first = self._answer(self.sample_query(), corpus, top_k=len(corpus))
        second = self._answer(self.sample_query(), corpus, top_k=len(corpus))
        assert first == second

    # -- End to end -----------------------------------------------------------

    def test_end_to_end_run_all(self) -> None:
        corpus = self.sample_corpus()
        result = self._run_all(self.sample_query(), corpus, top_k=len(corpus))
        assert result["documents"][0]["doc_id"] == self.expected_top_doc_id()
        assert self.expected_answer_substring() in result["answer"]
