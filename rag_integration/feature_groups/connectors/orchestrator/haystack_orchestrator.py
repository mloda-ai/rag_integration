"""Haystack orchestrator backend.

Canonical concrete for the ``orchestrator`` family: runs a real Haystack
pipeline (``InMemoryDocumentStore`` + ``InMemoryBM25Retriever``) entirely
in-memory. Zero-download (BM25 needs no model and no API) and deterministic, so
it anchors the CI contract suite while exercising a genuine external framework.
Behind the ``orchestrator`` extra. Haystack telemetry is disabled (via
``HAYSTACK_TELEMETRY_ENABLED``, set before the lazy import) to keep runs
offline and deterministic.
"""

from __future__ import annotations

import functools
import os
from typing import Any, ClassVar

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector


class HaystackOrchestrator(BaseOrchestratorConnector):
    """Whole-pipeline retrieval via Haystack (``orchestrator_backend="haystack"``).

    Builds an in-memory document store and BM25 retrieval pipeline once per
    corpus, cached across runs. The answer (no LLM) is the top document's
    content; the surfaced documents carry the pipeline's BM25 scores.
    """

    ORCHESTRATOR_BACKENDS: ClassVar = {
        "haystack": "Haystack in-memory BM25 pipeline",
    }

    PROPERTY_MAPPING: ClassVar = {
        BaseOrchestratorConnector.ORCHESTRATOR_BACKEND: property_spec(
            "Use 'haystack' for a Haystack BM25 pipeline", context=False
        ),
        BaseOrchestratorConnector.QUERY_TEXT: BaseOrchestratorConnector.QUERY_TEXT_SPEC,
        BaseOrchestratorConnector.TOP_K: BaseOrchestratorConnector.TOP_K_SPEC,
        BaseOrchestratorConnector.CORPUS: property_spec(
            "Inline corpus: a list of {doc_id, text} dicts", context=False, default=None
        ),
    }

    @classmethod
    def _run(cls, query: str, corpus: list[dict[str, Any]], top_k: int) -> tuple[str, list[dict[str, Any]]]:
        entries = [(str(doc.get("doc_id", str(i))), str(doc.get("text", ""))) for i, doc in enumerate(corpus)]

        # Nothing rankable -> empty result (rather than leaking a framework
        # error): an empty/whitespace query, an all-empty-text corpus (BM25
        # divides by the average document length), or a non-positive top_k.
        effective_k = min(top_k, len(entries))
        if not query.strip() or not any(text.strip() for _, text in entries) or effective_k <= 0:
            return "", []

        result = _cached_pipeline(cls, tuple(entries)).run({"retriever": {"query": query, "top_k": effective_k}})

        # document.score is float | None; the BM25 retriever filters out
        # non-positive scores, so every surfaced document carries a float score.
        documents = [
            {"doc_id": document.id, "text": document.content, "score": float(document.score)}
            for document in result["retriever"]["documents"]
        ]
        answer = documents[0]["text"] if documents else ""
        return answer, documents

    @classmethod
    def _build_pipeline(cls, entries: tuple[tuple[str, str], ...]) -> Any:
        """A BM25 pipeline over ``entries``; ``Pipeline.run`` keeps per-run state local, so runs may share it."""
        # Haystack evaluates telemetry at first import (and Pipeline.run() would
        # otherwise POST a PostHog event and write ~/.haystack/config.yaml), so
        # opt out before the lazy import to keep runs offline and deterministic.
        os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")

        from haystack import Document, Pipeline
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        # Not shared: a shared store lives in process-global storage and outlives its cache entry.
        store = InMemoryDocumentStore(shared=False)
        store.write_documents([Document(id=doc_id, content=text) for doc_id, text in entries])
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
        return pipeline

    @staticmethod
    def clear_pipeline_cache() -> None:
        """Drop every cached pipeline (the cache holds corpus text in process memory)."""
        _cached_pipeline.cache_clear()


# Up to PIPELINE_CACHE_SIZE (backend, corpus) pipelines stay cached process-wide, least recently used
# evicted first. A concurrent miss may build the same pipeline twice.
PIPELINE_CACHE_SIZE = 8


@functools.lru_cache(maxsize=PIPELINE_CACHE_SIZE)
def _cached_pipeline(connector: type[HaystackOrchestrator], entries: tuple[tuple[str, str], ...]) -> Any:
    return connector._build_pipeline(entries)
