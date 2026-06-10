"""Stage <-> connector migration-seam parity tests (issue #36).

Stage and connector emit the same passage / answer row shape under the same
canonical feature name, so migration is an option swap, not a pipeline
rewrite. Verified end to end for retrieve (FAISS stage vs dense connector)
and generate (llm_response stage vs extractive connector).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import pytest

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector
from rag_integration.feature_groups.connectors.generate.extractive_responder import ExtractiveResponder
from rag_integration.feature_groups.connectors.retrieve.base import BaseRetrieveConnector
from rag_integration.feature_groups.connectors.retrieve.faiss_retriever import FaissDenseRetriever
from rag_integration.feature_groups.rag_pipeline.embedding.hash_embed import HashEmbedder
from rag_integration.feature_groups.rag_pipeline.llm_response.base import BaseLLMResponse
from rag_integration.feature_groups.rag_pipeline.retrieval.base import BaseRetriever
from rag_integration.feature_groups.rag_pipeline.retrieval.faiss_retriever import FaissRetriever
from tests.integration.helpers import flatten_result

# Hash-embedder friendly (whitespace tokens, no punctuation): the query shares
# two tokens with d2, one with d1, none with the distractors.
CORPUS = [
    {"doc_id": "d0", "text": "the mat lay flat on the floor by the window"},
    {"doc_id": "d1", "text": "a dog can be a loyal and energetic pet"},
    {"doc_id": "d2", "text": "a cat is an independent and curious pet"},
    {"doc_id": "d3", "text": "cars need regular engine oil and maintenance"},
]
QUERY = "cat pet"
TOP_K = 2
EMBED_DIM = 384


class _StubLLMResponse(BaseLLMResponse):
    """Deterministic offline stand-in for an LLM stage backend."""

    LLM_METHODS = {"stub": "Deterministic stub for tests"}

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Any,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        # Gate on the stub's selector so unrelated tests never match it.
        if options.get(cls.LLM_METHOD) != "stub":
            return False
        return bool(super().match_feature_group_criteria(feature_name, options, data_access_collection))

    @classmethod
    def _generate(cls, query: str, context: str, system_prompt: str, options: Options) -> str:
        return f"stub answer to: {query}"


def _run_one(feature: Feature, groups: set[type[FeatureGroup]], key: str) -> Any:
    result = mlodaAPI.run_all(
        [feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(groups),
    )
    for row in flatten_result(result):
        if key in row:
            return row[key]
    raise AssertionError(f"run_all returned no '{key}' row: {result!r}")


def _build_stage_index(tmp_path: Path) -> tuple[str, str]:
    """Build the on-disk FAISS index + metadata sidecar the stage consumes."""
    texts = [str(doc["text"]) for doc in CORPUS]
    vectors = HashEmbedder._embed_texts(texts, EMBED_DIM, "default")
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(vectors, dtype=np.float32))

    index_path = str(tmp_path / "parity_index.faiss")
    faiss.write_index(index, index_path)

    metadata = {"texts": texts, "doc_ids": [str(doc["doc_id"]) for doc in CORPUS]}
    metadata_path = str(tmp_path / "parity_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    FaissRetriever._index_cache = None
    FaissRetriever._metadata_cache = None
    return index_path, metadata_path


def _assert_passage_shape(passages: List[Dict[str, Any]]) -> None:
    assert isinstance(passages, list)
    assert passages, "no passages came back; parity assertions would be vacuous"
    for rank, passage in enumerate(passages):
        assert set(passage) == {"doc_id", "text", "score", "rank"}
        assert isinstance(passage["doc_id"], str)
        assert isinstance(passage["text"], str)
        assert isinstance(passage["score"], float)
        assert passage["rank"] == rank
    scores = [p["score"] for p in passages]
    assert scores == sorted(scores, reverse=True)


class TestCanonicalFeatureNames:
    def test_stage_keys_match_connector_root_feature_names(self) -> None:
        """The stage-side literals must stay equal to the connector contract names."""
        assert BaseRetriever.PASSAGES_KEY == BaseRetrieveConnector.ROOT_FEATURE_NAME
        assert BaseLLMResponse.ANSWER_KEY == BaseGenerateConnector.ROOT_FEATURE_NAME

    def test_stage_yield_gates_match_connector_selector_keys(self) -> None:
        """The selector literals the stage gates yield on must stay the family selectors."""
        assert BaseRetrieveConnector.RETRIEVE_BACKEND == "retrieve_backend"
        assert BaseGenerateConnector.GENERATE_BACKEND == "generate_backend"

    def test_stage_yields_canonical_name_to_explicit_connector_backend(self) -> None:
        """With mixed options (half-finished migration) only the connector claims the request."""
        mixed_retrieve = Options(context={"index_path": "some/index.faiss", "retrieve_backend": "faiss"})
        assert FaissRetriever.match_feature_group_criteria(BaseRetriever.PASSAGES_KEY, mixed_retrieve) is False
        assert FaissDenseRetriever.match_feature_group_criteria(BaseRetrieveConnector.ROOT_FEATURE_NAME, mixed_retrieve)

        mixed_generate = Options(context={"query": "q", "llm_method": "stub", "generate_backend": "extractive"})
        assert _StubLLMResponse.match_feature_group_criteria(BaseLLMResponse.ANSWER_KEY, mixed_generate) is False
        assert ExtractiveResponder.match_feature_group_criteria(BaseGenerateConnector.ROOT_FEATURE_NAME, mixed_generate)


class TestRetrieveSeamParity:
    def test_stage_and_connector_emit_same_passage_rows(self, tmp_path: Path) -> None:
        index_path, metadata_path = _build_stage_index(tmp_path)

        stage_feature = Feature(
            BaseRetriever.PASSAGES_KEY,
            options=Options(
                context={
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                    "query_text": QUERY,
                    "embedding_method": "hash",
                    "top_k": TOP_K,
                }
            ),
        )
        stage_passages = _run_one(stage_feature, {FaissRetriever}, BaseRetriever.PASSAGES_KEY)

        connector_feature = Feature(
            BaseRetrieveConnector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    "retrieve_backend": "faiss",
                    "query_text": QUERY,
                    "corpus": CORPUS,
                    "top_k": TOP_K,
                }
            ),
        )
        connector_passages = _run_one(connector_feature, {FaissDenseRetriever}, BaseRetrieveConnector.ROOT_FEATURE_NAME)

        _assert_passage_shape(stage_passages)
        _assert_passage_shape(connector_passages)

        # Same vectors: same documents, same order, same cosine scores.
        assert [p["doc_id"] for p in stage_passages] == [p["doc_id"] for p in connector_passages]
        assert [p["text"] for p in stage_passages] == [p["text"] for p in connector_passages]
        for stage_passage, connector_passage in zip(stage_passages, connector_passages):
            assert stage_passage["score"] == pytest.approx(connector_passage["score"], abs=1e-5)
        assert stage_passages[0]["doc_id"] == "d2"

    def test_no_match_query_returns_empty_on_both_paths(self, tmp_path: Path) -> None:
        """A query relevant to nothing yields no passages on either path."""
        index_path, metadata_path = _build_stage_index(tmp_path)

        stage_feature = Feature(
            BaseRetriever.PASSAGES_KEY,
            options=Options(
                context={
                    "index_path": index_path,
                    "metadata_path": metadata_path,
                    "query_text": "zzzz qqqq",
                    "embedding_method": "hash",
                    "top_k": TOP_K,
                }
            ),
        )
        stage_passages = _run_one(stage_feature, {FaissRetriever}, BaseRetriever.PASSAGES_KEY)

        connector = FaissDenseRetriever()
        connector_passages = connector._retrieve("zzzz qqqq", CORPUS, TOP_K)

        assert stage_passages == []
        assert connector_passages == []


class TestGenerateSeamParity:
    def test_stage_and_connector_emit_same_answer_shape(self) -> None:
        query = "what is a cat"

        stage_feature = Feature(
            BaseLLMResponse.ANSWER_KEY,
            options=Options(
                context={
                    "query": query,
                    "context": [str(doc["text"]) for doc in CORPUS],
                    "llm_method": "stub",
                }
            ),
        )
        stage_answer = _run_one(stage_feature, {_StubLLMResponse}, BaseLLMResponse.ANSWER_KEY)

        connector_feature = Feature(
            BaseGenerateConnector.ROOT_FEATURE_NAME,
            options=Options(
                context={
                    "generate_backend": "extractive",
                    "query_text": query,
                    "passages": [{"doc_id": "d2", "text": "a cat is an independent and curious pet"}],
                }
            ),
        )
        connector_answer = _run_one(connector_feature, {ExtractiveResponder}, BaseGenerateConnector.ROOT_FEATURE_NAME)

        for answer in (stage_answer, connector_answer):
            assert set(answer) == {"answer", "citations"}
            assert isinstance(answer["answer"], str)
            assert answer["answer"], "answer must be non-empty for a shape comparison that means anything"
            assert isinstance(answer["citations"], list)
            assert all(isinstance(c, str) for c in answer["citations"])
