"""Base class for the ``generate`` connector family.

Contract: ``query_text + passages -> answer + citations``.

A generate connector takes a query and supporting passages (e.g. from the
retrieve or rerank families) and produces a grounded answer plus the passage
ids it drew from. It is a ROOT FeatureGroup here: passages are passed inline
through ``Options`` so the family is self-contained and contract-testable
without a network or an LLM.

Output (single row, keyed by the root feature name)::

    {"generated_answer": {"answer": "...", "citations": ["doc_id", ...]}}

The canonical concrete is deterministic and offline. LLM-backed generators are
pedigree backends that belong behind their own extra. The contract enforces
that the answer is *grounded*: every citation is one of the supplied passages.

This mirrors the retrieve/rerank families (selector-gated matching, a single
abstract hook, single-row output) but the output is an answer object, not a
ranked-passage list, so it copies the pattern rather than subclassing them.
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
from rag_integration.feature_groups.connectors.mixins import DocCollectionMixin, OptionsMixin, SingleQueryPerRunMixin


class BaseGenerateConnector(SingleQueryPerRunMixin, OptionsMixin, DocCollectionMixin, FeatureGroup):
    """Root FeatureGroup for generate-connector backends.

    A concrete backend declares its selector value in ``GENERATE_BACKENDS`` and
    implements :meth:`_generate`; the base owns option extraction, the
    single-row assembly, and validation that every returned citation is one of
    the supplied passages (no hallucinated sources). Selection is via
    :meth:`match_feature_group_criteria`, gating on
    ``generate_backend in cls.GENERATE_BACKENDS``.
    """

    ROOT_FEATURE_NAME = "generated_answer"

    # Option keys.
    GENERATE_BACKEND = "generate_backend"
    QUERY_TEXT = "query_text"
    PASSAGES = "passages"

    # Filled per concrete; empty on the base so it never matches.
    GENERATE_BACKENDS: Dict[str, str] = {}

    # Declarative option documentation only; selection is via
    # ``match_feature_group_criteria`` (not the FeatureChainParser).
    PROPERTY_MAPPING = {
        GENERATE_BACKEND: {"explanation": "Which generate-connector backend to use"},
        QUERY_TEXT: {"explanation": "The question to answer"},
        PASSAGES: {"explanation": "Supporting passages: a list of {doc_id, text} dicts"},
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
        backend = options.get(cls.GENERATE_BACKEND)
        return backend in cls.GENERATE_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> None:
        """Root feature: no input features (passages arrive via Options)."""
        return None

    @classmethod
    def _get_passages(cls, options: Options) -> List[Dict[str, Any]]:
        passages = cls._require_doc_list(options, cls.PASSAGES)
        duplicate = cls._find_duplicate_doc_id(passages)
        if duplicate is not None:
            raise DuplicateDocIdError(
                f"{cls.__name__} received duplicate passage doc_id '{duplicate}'; doc_ids must be unique."
            )
        return passages

    @classmethod
    @abstractmethod
    def _generate(cls, query: str, passages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        """Answer ``query`` from ``passages``.

        Returns ``(answer, citations)`` where ``answer`` is the answer text and
        ``citations`` is the list of ``doc_id``s the answer draws from. Each
        citation must be the ``doc_id`` of one of the supplied passages (the
        base validates this). The base handles empty passages itself, so this
        hook is never called with an empty list.
        """
        ...

    @classmethod
    def _validate_citations(cls, citations: List[str], passages: List[Dict[str, Any]]) -> None:
        """Reject any citation that is not one of the supplied passage doc_ids, or cited twice."""
        known = cls._known_doc_ids(passages)
        for citation in citations:
            if citation not in known:
                raise GroundingError(
                    f"{cls.__name__}._generate cited '{citation}', which is not among the supplied passages."
                )
        if len(citations) != len(set(citations)):
            raise GroundingError(
                f"{cls.__name__}._generate returned duplicate citations; each doc_id may be cited once."
            )

    @classmethod
    def _answer(cls, query: str, passages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assemble the answer contract around the backend's :meth:`_generate`."""
        if not passages:
            return {"answer": "", "citations": []}
        answer, citations = cls._generate(query, passages)
        cls._validate_citations(citations, passages)
        # Grounded by construction, in both directions: a non-empty answer must
        # cite its source(s), and citations without an answer are meaningless.
        if answer.strip() and not citations:
            raise GroundingError(
                f"{cls.__name__}._generate returned a non-empty answer with no citations; "
                f"a grounded answer must cite at least one supplied passage."
            )
        if not answer.strip() and citations:
            raise GroundingError(
                f"{cls.__name__}._generate returned citations with an empty answer; "
                f"citations are only valid for a non-empty answer."
            )
        if not answer.strip():
            # Normalize a whitespace-only answer so the empty shape is always
            # exactly {"answer": "", "citations": []}.
            return {"answer": "", "citations": []}
        return {"answer": answer, "citations": citations}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> List[Dict[str, Any]]:
        """Generate an answer from the passages, return the answer object."""
        cls._assert_single_feature(features)
        for feature in features.features:
            options = feature.options
            query = cls._require_option(options, cls.QUERY_TEXT)
            passages = cls._get_passages(options)
            return [{cls.ROOT_FEATURE_NAME: cls._answer(str(query), passages)}]
        return []
