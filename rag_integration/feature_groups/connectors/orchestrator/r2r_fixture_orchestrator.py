"""R2R fixture-stub orchestrator backend.

Second concrete for the ``orchestrator`` family, and a different *integration
mode* from the in-process ``HaystackOrchestrator``: it models a server-shaped
RAG tool (R2R) over a static JSON fixture instead of running a library
in-process. There is no server and no network: the fixture holds canned R2R
``/rag``-style responses (a generated answer plus ranked source doc_ids), keyed
by query, exactly as the open-kgo ``rest_public`` file-fixture connectors model
a REST API from local files.

The honest-surface mechanism is **narrowing**: the corpus passed to the family
is treated as the documents ingested into R2R, and the stub surfaces only the
canned doc_ids that are actually in that corpus (with the corpus's own text), so
nothing is fabricated. A query with no canned response yields ``("", [])`` (the
server has nothing indexed for it). The canned answer is surfaced only when the
document it is drawn from (``answer_doc_id``) is among the documents actually
surfaced: it is suppressed when that document is dropped either by corpus
narrowing or by ``top_k`` truncation. In both cases the result is retrieve-only
(the surviving documents, an empty answer), so the answer always rests on the
surfaced documents it was drawn from.

Zero-download, zero-dependency (stdlib ``json``), deterministic; a CI anchor
alongside the Haystack backend.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.orchestrator.base import BaseOrchestratorConnector

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "r2r_responses.json"


class R2RFixtureOrchestrator(BaseOrchestratorConnector):
    """R2R-shaped fixture-stub orchestrator (``orchestrator_backend="r2r"``).

    Answers from a bundled JSON fixture of canned R2R responses, narrowed to the
    supplied corpus. The fixture is loaded once and cached at class level; the
    read is deterministic, so repeated calls are idempotent.
    """

    ORCHESTRATOR_BACKENDS = {
        "r2r": "R2R-shaped server stub over a static JSON fixture (honest-surface narrowing)",
    }

    PROPERTY_MAPPING = {
        BaseOrchestratorConnector.ORCHESTRATOR_BACKEND: property_spec(
            "Use 'r2r' for the R2R fixture-stub pipeline", context=False
        ),
        BaseOrchestratorConnector.QUERY_TEXT: property_spec(
            "The query to look up in the canned R2R responses", context=False
        ),
        BaseOrchestratorConnector.TOP_K: property_spec(
            f"Number of documents to surface (default {BaseOrchestratorConnector.DEFAULT_TOP_K})", context=False
        ),
        BaseOrchestratorConnector.CORPUS: property_spec(
            "Inline corpus (the documents ingested into R2R)", context=False
        ),
    }

    _responses: Dict[str, Any] | None = None
    _cache_lock = threading.Lock()

    @classmethod
    def _get_responses(cls) -> Dict[str, Any]:
        """Load and cache the canned-response table from the bundled fixture.

        The returned table is the shared cache and must be treated as read-only;
        ``_run`` only reads from it and emits fresh dicts for surfaced documents,
        so the cache is never mutated through a result.
        """
        responses = cls._responses
        if responses is not None:
            return responses
        with cls._cache_lock:
            if cls._responses is None:
                try:
                    with _FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
                        payload = json.load(fixture_file)
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"{cls.__name__}: failed to load bundled R2R fixture {_FIXTURE_PATH}: {exc}"
                    ) from exc
                cls._responses = dict(payload.get("responses", {}))
            return cls._responses

    @classmethod
    def _run(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
        effective_k = min(top_k, len(corpus))
        if not query.strip() or effective_k <= 0:
            return "", []

        response = cls._get_responses().get(query.strip().lower())
        if response is None:
            # The server has no canned response for this query (honest surface).
            return "", []

        text_by_doc_id = {str(doc.get("doc_id", str(i))): str(doc.get("text", "")) for i, doc in enumerate(corpus)}

        # Narrowing: keep only canned doc_ids that are in the ingested corpus,
        # surfacing the corpus's own text (never the fixture's), so a surfaced
        # document is always grounded in what was actually supplied.
        documents: List[Dict[str, Any]] = []
        for entry in response.get("documents", []):
            doc_id = str(entry.get("doc_id"))
            if doc_id in text_by_doc_id:
                documents.append(
                    {"doc_id": doc_id, "text": text_by_doc_id[doc_id], "score": float(entry.get("score", 0.0))}
                )
            if len(documents) >= effective_k:
                break

        # If narrowing removed every document, there is nothing to ground an
        # answer on, so return an empty result rather than an ungrounded answer.
        if not documents:
            return "", []

        # The canned answer is drawn from one source document (answer_doc_id).
        # Surface the answer only if that document is among the SURFACED
        # documents: it may have been dropped by corpus narrowing or by top_k
        # truncation, and in either case we return a retrieve-only result (the
        # surviving documents, no answer) rather than an answer whose support
        # was not surfaced.
        answer_doc_id = str(response.get("answer_doc_id", ""))
        surfaced_ids = {document["doc_id"] for document in documents}
        answer = str(response.get("answer", "")) if answer_doc_id in surfaced_ids else ""
        return answer, documents
