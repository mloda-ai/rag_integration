"""Base class for the ``retrieve`` connector family.

Contract: ``query_text + corpus + top_k -> ranked passages with scores``.

By default a retrieve connector is a ROOT FeatureGroup: it takes an inline
corpus and a query through ``Options`` and returns the passages ranked
best-first. Alternatively, ``corpus_source`` names an upstream feature whose
single row carries the corpus (a list of ``{doc_id, text}`` dicts), declared
as the connector's input. Ranking and output are identical on both paths.
Concrete backends (lexical, dense, hybrid, late-interaction) differ only in
the ranking they apply behind this one contract; they declare their selector
value in ``RETRIEVE_BACKENDS`` and implement :meth:`_rank`.

Output (single row, keyed by the root feature name)::

    {"retrieved_passages": [{"doc_id": ..., "text": ..., "score": ..., "rank": ...}, ...]}

``score`` is higher-is-more-relevant; ``rank`` is 0-based, ascending, best
first. Backends return at most ``top_k`` passages and only those with a
positive score, so a degenerate query (empty, or sharing no terms with the
corpus) yields no passages. ``PythonDictFramework`` slices the result to the
requested feature, so the ranked-passage list is the whole contract.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

from mloda.provider import ComputeFramework, DataCreator, FeatureGroup, FeatureSet, property_spec
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import columnar_to_rows

from rag_integration.feature_groups.connectors.errors import (
    DuplicateDocIdError,
    InvalidOptionError,
    RankingContractError,
)
from rag_integration.feature_groups.connectors.mixins import (
    DocCollectionMixin,
    OptionsMixin,
    RankingValidationMixin,
    SingleQueryPerRunMixin,
    TopKMixin,
)


class BaseRetrieveConnector(
    SingleQueryPerRunMixin, OptionsMixin, TopKMixin, DocCollectionMixin, RankingValidationMixin, FeatureGroup
):
    """FeatureGroup for retrieve-connector backends.

    A concrete backend declares its selector value in ``RETRIEVE_BACKENDS`` and
    implements :meth:`_rank` (the only per-backend logic); the base owns the
    empty-corpus / ``top_k`` clamping and the passage-assembly contract so every
    backend returns an identically shaped result.

    Selection is unambiguous and done entirely by
    :meth:`match_feature_group_criteria`, which gates on
    ``retrieve_backend in cls.RETRIEVE_BACKENDS``. Because each backend declares
    a disjoint selector value and mloda raises when more than one feature group
    matches, at most one backend ever claims a given ``Options``. The base keeps
    ``RETRIEVE_BACKENDS`` empty so it never matches.

    Reuse note for sibling families (rerank, generate, ...): the carry-over is
    the *shape* (the ``RETRIEVE_BACKENDS`` selector dict, the
    ``match_feature_group_criteria`` gating, and the ``_rank`` hoist), not this
    class. The root/``DataCreator``/corpus handling below is
    retrieve-specific: rerank consumes candidate passages as input features
    rather than an inline corpus, so a sibling family copies this pattern, it
    does not subclass ``BaseRetrieveConnector``.
    """

    ROOT_FEATURE_NAME = "retrieved_passages"

    # Option keys. ``TOP_K`` / ``DEFAULT_TOP_K`` come from ``TopKMixin``.
    RETRIEVE_BACKEND = "retrieve_backend"
    QUERY_TEXT = "query_text"
    CORPUS = "corpus"
    CORPUS_SOURCE = "corpus_source"

    # Family option keys, kept off the corpus-source feature in input_features.
    FAMILY_OPTION_KEYS = frozenset({RETRIEVE_BACKEND, CORPUS_SOURCE, QUERY_TEXT, TopKMixin.TOP_K, CORPUS})

    # Filled per concrete: {backend_value: human-readable description}. The base
    # stays empty so it never matches a feature. Values must be disjoint across
    # backends (see the class docstring).
    RETRIEVE_BACKENDS: ClassVar[dict[str, str]] = {}

    # Shared so each concrete's PROPERTY_MAPPING advertises the same spec.
    CORPUS_SOURCE_SPEC = property_spec(
        "Name of an upstream feature whose single row carries the corpus as a list of {doc_id, text} dicts."
        " Optional: replaces the inline corpus",
        context=False,
        default=None,
    )

    # Selection is via ``match_feature_group_criteria`` on the backend key. The other
    # keys declare ``default=None`` (mloda treats a key without a default as required) and are
    # validated in ``calculate_feature``.
    PROPERTY_MAPPING: ClassVar = {
        RETRIEVE_BACKEND: property_spec("Which retrieve-connector backend to use", context=False),
        QUERY_TEXT: property_spec("Raw text query to search the corpus", context=False, default=None),
        TopKMixin.TOP_K: property_spec(
            f"Number of passages to return (default {TopKMixin.DEFAULT_TOP_K})", context=False, default=None
        ),
        CORPUS: property_spec("Inline corpus: a list of {doc_id, text} dicts", context=False, default=None),
        CORPUS_SOURCE: CORPUS_SOURCE_SPEC,
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({cls.ROOT_FEATURE_NAME})

    @classmethod
    def match_feature_group_criteria(
        cls,
        feature_name: FeatureName | str,
        options: Options,
        data_access_collection: Any = None,
    ) -> bool:
        """Match the root feature name only for a backend this concrete declares.

        Gating on ``retrieve_backend`` (rather than name alone) is what keeps
        concrete backends mutually exclusive, so enabling several at once is
        unambiguous. An unknown backend matches nothing (honest surface: the
        connector does not silently claim a backend it cannot serve).
        """
        if str(feature_name) != cls.ROOT_FEATURE_NAME:
            return False
        backend = options.get(cls.RETRIEVE_BACKEND)
        return backend in cls.RETRIEVE_BACKENDS

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        """Declare the corpus-source feature as input when ``CORPUS_SOURCE`` is set.

        Without it this is a root feature (corpus arrives via Options). With
        it, the source gets the parent's group and context options minus
        ``FAMILY_OPTION_KEYS``: context keys do not propagate on their own, and
        ``forward_group_exclude`` keeps the engine's default group forwarding
        from re-adding query-specific keys to the source.
        """
        source = options.get(self.CORPUS_SOURCE)
        if source is None:
            return None
        forwarded_group = {key: value for key, value in options.group.items() if key not in self.FAMILY_OPTION_KEYS}
        forwarded_context = {key: value for key, value in options.context.items() if key not in self.FAMILY_OPTION_KEYS}
        return {
            Feature(
                str(source),
                options=Options(group=forwarded_group, context=forwarded_context),
                forward_group_exclude=self.FAMILY_OPTION_KEYS,
            )
        }

    @classmethod
    def _corpus_from_source(cls, data: Any, source_name: str) -> list[dict[str, Any]]:
        """Read the corpus list the corpus-source feature produced in its single row."""
        for row in columnar_to_rows(data):
            if isinstance(row, dict) and source_name in row:
                payload = row[source_name]
                if not isinstance(payload, (list, tuple)):
                    raise InvalidOptionError(
                        f"{cls.__name__} corpus source '{source_name}' must produce a single row holding a "
                        f"list of {{doc_id, text}} dicts, got {payload!r}."
                    )
                return list(payload)
        raise InvalidOptionError(f"{cls.__name__} corpus source '{source_name}' produced no row.")

    @classmethod
    @abstractmethod
    def _rank(cls, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
        """Rank ``texts`` against ``query``.

        Returns at most ``top_k`` ``(corpus_index, score)`` pairs, ordered
        best-first, where ``score`` is higher-is-more-relevant. The unified
        family rule: only pairs with a positive score are returned, so a
        degenerate query (empty, or sharing no terms with the corpus) yields no
        pairs. Requirements the base relies on and enforces (see
        :meth:`_validate_ranking`): indices are in range
        (``0 <= corpus_index < len(texts)``) and unique, at most ``top_k``
        pairs come back, and scores are non-increasing. ``top_k`` is already
        clamped to ``1 <= top_k <= len(texts)``, so backends need not re-check
        it. The base does not re-sort, so returning best-first is a hard
        requirement.
        """
        ...

    @classmethod
    def _validate_ranking(cls, ranked: list[tuple[int, float]], corpus_size: int, top_k: int) -> None:
        """Enforce the four :meth:`_rank` requirements (count is retrieve-specific; rest shared)."""
        if len(ranked) > top_k:
            raise RankingContractError(f"{cls.__name__}._rank returned {len(ranked)} pairs for top_k={top_k}.")
        cls._validate_rank_indices(ranked, corpus_size, f"a corpus of size {corpus_size}", non_increasing=True)

    @classmethod
    def _retrieve(
        cls,
        query: str,
        corpus: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Assemble the ranked-passage contract around the backend's :meth:`_rank`.

        Owns the cross-backend invariants: empty corpus and non-positive
        ``top_k`` return ``[]``; ``top_k`` is clamped to the corpus size;
        ``doc_id``/``text`` are read from the corpus; ``rank`` is assigned
        0-based ascending; ``score`` is coerced to ``float``. Corpus entries
        must be dicts, and the effective ``doc_id`` values (after ``str()``
        coercion and the positional-index fallback for a missing ``doc_id``)
        must be unique; either violation raises ``ValueError``. A missing
        ``text`` key stays lenient and coerces to ``""``. The ranking a backend
        returns is validated against the :meth:`_rank` contract, so a buggy
        ``_rank`` fails loudly here (for any input) instead of silently
        dropping or duplicating a passage.
        """
        if not corpus:
            return []

        for i, doc in enumerate(corpus):
            if not isinstance(doc, dict):
                raise InvalidOptionError(
                    f"{cls.__name__} corpus entry at index {i} is not a dict: {doc!r}. "
                    f"Each entry must be a {{doc_id, text}} dict."
                )

        duplicate = cls._find_duplicate_doc_id(corpus)
        if duplicate is not None:
            raise DuplicateDocIdError(
                f"{cls.__name__} corpus contains duplicate doc_id {duplicate!r} "
                f"(after str() coercion and the positional-index fallback)."
            )
        doc_ids = cls._effective_doc_ids(corpus)

        effective_k = min(top_k, len(corpus))
        if effective_k <= 0:
            return []

        texts = [str(doc.get("text", "")) for doc in corpus]

        ranked = cls._rank(query, texts, effective_k)
        cls._validate_ranking(ranked, len(corpus), effective_k)

        passages: list[dict[str, Any]] = []
        for rank, (corpus_idx, score) in enumerate(ranked):
            passages.append(
                {
                    "doc_id": doc_ids[corpus_idx],
                    "text": texts[corpus_idx],
                    "score": float(score),
                    "rank": rank,
                }
            )
        return passages

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> list[dict[str, Any]]:
        """Rank the corpus against the query, return ranked passages."""
        cls._assert_single_feature(features)
        for feature in features.features:
            options = feature.options
            query = cls._require_option(options, cls.QUERY_TEXT)
            source = options.get(cls.CORPUS_SOURCE)
            if source is not None:
                if options.get(cls.CORPUS) is not None:
                    raise InvalidOptionError(
                        f"{cls.__name__} got both '{cls.CORPUS_SOURCE}' and inline '{cls.CORPUS}'; pass one corpus only."
                    )
                corpus = cls._corpus_from_source(data, str(source))
            else:
                corpus = cls._require_doc_list(options, cls.CORPUS)
            top_k = cls._get_top_k(options)
            passages = cls._retrieve(str(query), corpus, top_k)
            return [{cls.ROOT_FEATURE_NAME: passages}]
        return []
