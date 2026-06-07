"""Concurrency tests for the lock-guarded class-level caches (issue: thread safety).

Each test fires many threads at a cache whose underlying builder is mocked to be
slow, and asserts the builder ran exactly once (no double initialization).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List

import pytest


def _run_concurrently(target: Callable[[], Any], threads: int = 8) -> List[Any]:
    """Run target() in many threads released simultaneously; return their results."""
    barrier = threading.Barrier(threads)
    results: List[Any] = [None] * threads
    errors: List[BaseException] = []

    def worker(idx: int) -> None:
        try:
            barrier.wait()
            results[idx] = target()
        except BaseException as exc:  # noqa: BLE001 - surfaced via assert below
            errors.append(exc)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    assert not errors, f"worker errors: {errors}"
    return results


class _CountingBuilder:
    """Callable that counts invocations and is deliberately slow."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> object:
        self.calls += 1
        time.sleep(0.02)  # widen the race window
        return object()


def test_sentence_transformer_model_built_once() -> None:
    pytest.importorskip("sentence_transformers")
    from rag_integration.feature_groups.rag_pipeline.embedding.sentence_transformer import (
        SentenceTransformerEmbedder,
    )

    SentenceTransformerEmbedder._model_cache = None
    builder = _CountingBuilder()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sentence_transformers.SentenceTransformer", builder)
            results = _run_concurrently(lambda: SentenceTransformerEmbedder._get_model("m"))
        assert builder.calls == 1
        assert all(r is results[0] for r in results)
    finally:
        SentenceTransformerEmbedder._model_cache = None


def test_semantic_chunker_model_built_once() -> None:
    pytest.importorskip("sentence_transformers")
    from rag_integration.feature_groups.rag_pipeline.chunking.semantic import SemanticChunker

    SemanticChunker._model_cache = None
    builder = _CountingBuilder()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("sentence_transformers.SentenceTransformer", builder)
            results = _run_concurrently(lambda: SemanticChunker._get_model("m"))
        assert builder.calls == 1
        assert all(r is results[0] for r in results)
    finally:
        SemanticChunker._model_cache = None


def test_presidio_analyzer_built_once() -> None:
    pytest.importorskip("presidio_analyzer")
    from rag_integration.feature_groups.rag_pipeline.pii_redaction.presidio import PresidioPIIRedactor

    PresidioPIIRedactor._analyzer = None
    builder = _CountingBuilder()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("presidio_analyzer.AnalyzerEngine", builder)
            results = _run_concurrently(PresidioPIIRedactor._get_analyzer)
        assert builder.calls == 1
        assert all(r is results[0] for r in results)
    finally:
        PresidioPIIRedactor._analyzer = None


def test_faiss_index_loaded_once() -> None:
    faiss = pytest.importorskip("faiss")
    from rag_integration.feature_groups.rag_pipeline.retrieval.faiss_retriever import FaissRetriever

    FaissRetriever._index_cache = None
    builder = _CountingBuilder()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(faiss, "read_index", builder)
            results = _run_concurrently(lambda: FaissRetriever._load_index("some/path"))
        assert builder.calls == 1
        assert all(r is results[0] for r in results)
    finally:
        FaissRetriever._index_cache = None
