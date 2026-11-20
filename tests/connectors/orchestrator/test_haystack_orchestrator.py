"""Contract test for :class:`HaystackOrchestrator`.

Runs a real Haystack in-memory BM25 pipeline (zero-download, so it runs on CI).
Skips cleanly when the ``orchestrator`` extra is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Feature, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector
from rag_integration.feature_groups.connectors.orchestrator.haystack_orchestrator import (
    HaystackOrchestrator,
    _cached_pipeline,
)
from tests.connectors.orchestrator.orchestrator_contract import OrchestratorConnectorContractBase

# Clean skip (not an error) when the `orchestrator` extra is not installed.
pytest.importorskip("haystack")


class TestHaystackOrchestrator(OrchestratorConnectorContractBase):
    @classmethod
    def connector_class(cls) -> type[BaseOrchestratorConnector]:
        return HaystackOrchestrator

    @classmethod
    def backend_value(cls) -> str:
        return "haystack"

    @classmethod
    def sample_corpus(cls) -> list[dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "A cat is an independent and curious pet."},
            {"doc_id": "d2", "text": "Dogs are loyal and energetic companions."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        return "cat pet"

    @classmethod
    def expected_top_doc_id(cls) -> str:
        return "d1"

    @classmethod
    def expected_answer_substring(cls) -> str:
        return "curious pet"

    def test_prepared_session_reuses_the_pipeline(self) -> None:
        connector = HaystackOrchestrator
        options = Options(context={connector.ORCHESTRATOR_BACKEND: "haystack", connector.CORPUS: self.sample_corpus()})
        session = mloda.prepare(
            [Feature(connector.ROOT_FEATURE_NAME, options=options)],
            [PythonDictFramework],
            plugin_collector=PluginCollector.enabled_feature_groups({connector}),
        )
        connector.clear_pipeline_cache()
        session.run(parameters={connector.QUERY_TEXT: self.sample_query(), connector.TOP_K: 1})
        built = _cached_pipeline.cache_info()
        second = self._answer_row(
            session.run(parameters={connector.QUERY_TEXT: "loyal energetic companions", connector.TOP_K: 3})
        )
        reused = _cached_pipeline.cache_info()
        assert (built.misses, built.hits) == (1, 0)
        assert (reused.misses, reused.hits) == (1, 1)
        assert second["documents"][0]["doc_id"] == "d2"
