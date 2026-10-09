"""Cross-cutting property mixins shared by the connector-family bases.

Each mixin hoists one concern duplicated inline: option parsing (``OptionsMixin``),
top_k cut-off (``TopKMixin``), doc collection / doc_id bookkeeping
(``DocCollectionMixin``), ranking validation (``RankingValidationMixin``), the
one-query-per-run guard (``SingleQueryPerRunMixin``), and reading an upstream
source's single row (``SingleRowSourceMixin``). They are plain classes listed
ahead of ``FeatureGroup`` in a base, so mloda discovery still sees only the
``FeatureGroup`` leaves; ``cls.__name__`` keeps messages naming the backend.
"""

from __future__ import annotations

import reprlib
from collections.abc import Callable, Sequence
from typing import Any

from mloda.user import Options
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import columnar_to_rows

from rag_integration.feature_groups.connectors.errors import (
    InvalidOptionError,
    MissingOptionError,
    RankingContractError,
)


class SingleQueryPerRunMixin:
    """Guard: each connector family answers exactly one query per run.

    All family bases mix this in so that a FeatureSet carrying two features
    with the same root feature name fails loudly instead of silently dropping
    every feature after the first.
    """

    @classmethod
    def _assert_single_feature(cls, features: Any) -> None:
        feature_list = list(features.features)
        if len(feature_list) > 1:
            raise ValueError(
                f"{cls.__name__} answers one query per run, but the FeatureSet contains {len(feature_list)} features."
            )


class SingleRowSourceMixin:
    """Read the one payload an upstream source feature produced (``corpus_source``, ``graph_source``)."""

    @classmethod
    def _single_row_payload(
        cls,
        data: Any,
        source_name: str,
        *,
        label: str,
        expected: str,
        is_valid: Callable[[Any], bool],
    ) -> Any:
        """Return the source's single payload; errors report a row count, never every payload."""
        payloads = [row[source_name] for row in columnar_to_rows(data) if isinstance(row, dict) and source_name in row]
        prefix = f"{cls.__name__} {label} source '{source_name}'"
        if not payloads:
            raise InvalidOptionError(f"{prefix} produced no row.")
        if len(payloads) > 1:
            raise InvalidOptionError(
                f"{prefix} must produce a single row holding {expected}, got {len(payloads)} rows."
            )
        if not is_valid(payloads[0]):
            # reprlib bounds the message for a large malformed payload.
            raise InvalidOptionError(
                f"{prefix} must produce a single row holding {expected}, got {reprlib.repr(payloads[0])}."
            )
        return payloads[0]


class OptionsMixin:
    """Read required values out of ``Options``."""

    @classmethod
    def _require_option(cls, options: Options, key: str) -> Any:
        value = options.get(key)
        if value is None:
            raise MissingOptionError(f"{cls.__name__} requires '{key}' in options.")
        return value


class TopKMixin:
    """The ``top_k`` cut-off (retrieve, rerank, graph_rag, orchestrator)."""

    TOP_K = "top_k"
    DEFAULT_TOP_K = 5

    @classmethod
    def _get_top_k(cls, options: Options) -> int:
        val = options.get(cls.TOP_K)
        if val is None:
            return cls.DEFAULT_TOP_K
        try:
            return int(val)
        except (ValueError, TypeError) as exc:
            raise InvalidOptionError(f"{cls.__name__} option '{cls.TOP_K}' must be an integer, got {val!r}.") from exc


class DocCollectionMixin:
    """A ``{doc_id, text}`` collection and its ``doc_id`` bookkeeping.

    Effective ``doc_id`` is the explicit value coerced to ``str``, else the
    positional index.
    """

    @staticmethod
    def _effective_doc_id(item: dict[str, Any], index: int) -> str:
        return str(item.get("doc_id", str(index)))

    @classmethod
    def _effective_doc_ids(cls, items: Sequence[dict[str, Any]]) -> list[str]:
        return [cls._effective_doc_id(item, i) for i, item in enumerate(items)]

    @classmethod
    def _known_doc_ids(cls, items: Sequence[dict[str, Any]]) -> set[str]:
        return set(cls._effective_doc_ids(items))

    @classmethod
    def _find_duplicate_doc_id(cls, items: Sequence[dict[str, Any]]) -> str | None:
        """Return the first repeated effective ``doc_id``, or ``None``."""
        seen: set[str] = set()
        for i, item in enumerate(items):
            doc_id = cls._effective_doc_id(item, i)
            if doc_id in seen:
                return doc_id
            seen.add(doc_id)
        return None

    @classmethod
    def _require_doc_list(cls, options: Options, key: str) -> list[dict[str, Any]]:
        value = options.get(key)
        if value is None:
            raise MissingOptionError(f"{cls.__name__} requires '{key}' in options: a list of {{doc_id, text}} dicts.")
        return list(value)


class RankingValidationMixin:
    """Validate the ``(index, score)`` pairs a backend ``_rank`` returns."""

    @classmethod
    def _validate_rank_indices(
        cls,
        ranked: list[tuple[int, float]],
        count: int,
        extent: str,
        *,
        non_increasing: bool = False,
    ) -> None:
        """Reject out-of-range / duplicate indices, and (if ``non_increasing``) rising scores.

        ``extent`` is the population label used in the out-of-range message
        (e.g. ``"3 candidates"``).
        """
        seen: set[int] = set()
        previous_score: float | None = None
        for idx, score in ranked:
            if not 0 <= idx < count:
                raise RankingContractError(f"{cls.__name__}._rank returned out-of-range index {idx} for {extent}.")
            if idx in seen:
                raise RankingContractError(f"{cls.__name__}._rank returned duplicate index {idx}.")
            seen.add(idx)
            if non_increasing:
                if previous_score is not None and score > previous_score:
                    raise RankingContractError(
                        f"{cls.__name__}._rank returned scores out of order: {score} after {previous_score} "
                        f"(scores must be non-increasing, best-first)."
                    )
                previous_score = score
