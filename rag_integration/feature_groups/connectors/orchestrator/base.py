"""Base class for the ``orchestrator`` connector family.

Contract: ``query_text + corpus + top_k -> answer + documents`` (internals opaque).

An orchestrator connector wraps a whole external RAG framework (LlamaIndex,
Haystack, txtai, ...) as a single connector: you hand it a query and a corpus,
it runs the framework's own pipeline, and you get an answer plus the documents
the pipeline surfaced. Unlike the retrieve/rerank/generate families, the
internals are the framework's, not ours; this family is about the *integration
surface* (bring your existing pipeline), not the algorithm.

It is a ROOT FeatureGroup: the corpus is passed inline through ``Options`` and
the framework runs fully in-memory, so the family is self-contained and
contract-testable without a server.

Output (single row, keyed by the root feature name)::

    {"orchestrated_answer": {"answer": "...", "documents": [{"doc_id": ..., "text": ..., "score": ...}, ...]}}

The base owns option extraction, single-row assembly, and validation that every
returned document came from the supplied corpus (no fabricated sources). A
backend implements only :meth:`_run` (driving its framework's pipeline).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mloda.provider import DataCreator, FeatureGroup, ComputeFramework, FeatureSet
from mloda.user import Options, FeatureName
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.errors import DuplicateDocIdError, GroundingError
from rag_integration.feature_groups.connectors.mixins import DocCollectionMixin, OptionsMixin, TopKMixin


class BaseOrchestratorConnector(OptionsMixin, TopKMixin, DocCollectionMixin, FeatureGroup):
    """Root FeatureGroup for orchestrator connector backends.

    A concrete backend declares its selector value in ``ORCHESTRATOR_BACKENDS``
    and implements :meth:`_run`; selection is via
    :meth:`match_feature_group_criteria`, gating on
    ``orchestrator_backend in cls.ORCHESTRATOR_BACKENDS``.
    """

    ROOT_FEATURE_NAME = "orchestrated_answer"

    # Option keys. ``TOP_K`` / ``DEFAULT_TOP_K`` come from ``TopKMixin``.
    ORCHESTRATOR_BACKEND = "orchestrator_backend"
    QUERY_TEXT = "query_text"
    CORPUS = "corpus"

    ORCHESTRATOR_BACKENDS: Dict[str, str] = {}

    PROPERTY_MAPPING = {
        ORCHESTRATOR_BACKEND: {"explanation": "Which orchestrator (external framework) backend to use"},
        QUERY_TEXT: {"explanation": "The query to run through the framework pipeline"},
        TopKMixin.TOP_K: {
            "explanation": f"Number of documents the pipeline should surface (default {TopKMixin.DEFAULT_TOP_K})"
        },
        CORPUS: {"explanation": "Inline corpus: a list of {doc_id, text} dicts"},
    }

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.ROOT_FEATURE_NAME})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: Union[FeatureName, str],
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match the root feature name only for a backend this concrete declares."""
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.ORCHESTRATOR_BACKEND)
        return backend in cls.ORCHESTRATOR_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (the corpus arrives via Options)."""
        return None

    @classmethod
    @abstractmethod
    def _run(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
        """Run the framework's pipeline for ``query`` over ``corpus``.

        Returns ``(answer, documents)`` where ``documents`` is a list of
        ``{doc_id, text, score}`` dicts the pipeline surfaced (best first) and
        ``answer`` is the framework's answer. Each document's ``doc_id`` must be
        one of the supplied corpus ids (the base validates this). The base
        short-circuits an empty corpus before dispatching, so :meth:`_run` is
        never called with one.
        """
        ...

    @classmethod
    def _validate_unique_doc_ids(cls, corpus: List[Dict[str, Any]]) -> None:
        """Reject duplicate effective doc_ids, uniformly across backends.

        An entry without ``doc_id`` defaults to its positional index, so an
        explicit ``doc_id`` ``"1"`` collides with a missing ``doc_id`` at
        index 1; the check runs on the effective ids.
        """
        duplicate = cls._find_duplicate_doc_id(corpus)
        if duplicate is not None:
            raise DuplicateDocIdError(f"{cls.__name__}: duplicate doc_id {duplicate!r} in corpus; ids must be unique.")

    @classmethod
    def _validate_documents(cls, documents: List[Dict[str, Any]], corpus: List[Dict[str, Any]]) -> None:
        """Reject any surfaced document whose doc_id is not in the supplied corpus."""
        known = cls._known_doc_ids(corpus)
        for document in documents:
            if str(document.get("doc_id")) not in known:
                raise GroundingError(
                    f"{cls.__name__}._run surfaced document {document.get('doc_id')!r}, "
                    f"which is not in the supplied corpus."
                )

    @classmethod
    def _answer(cls, query: str, corpus: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
        """Assemble the answer contract around the backend's :meth:`_run`."""
        if not corpus:
            return {"answer": "", "documents": []}
        cls._validate_unique_doc_ids(corpus)
        answer, documents = cls._run(query, corpus, top_k)
        cls._validate_documents(documents, corpus)
        # A non-empty answer must rest on surfaced documents. An empty answer
        # with no documents is a valid retrieve-only / no-match result; an empty
        # answer alongside documents is fine too (retrieve-only pipeline).
        if answer.strip() and not documents:
            raise GroundingError(f"{cls.__name__}._run returned a non-empty answer with no supporting documents.")
        return {"answer": answer, "documents": documents}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Run the framework pipeline, return the answer object."""
        for feature in features.features:
            options = feature.options
            query = cls._require_option(options, cls.QUERY_TEXT)
            corpus = cls._require_doc_list(options, cls.CORPUS)
            top_k = cls._get_top_k(options)
            return [{cls.ROOT_FEATURE_NAME: cls._answer(str(query), corpus, top_k)}]
        return []
