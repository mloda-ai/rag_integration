"""Tests for the shared BaseRowDeduplicator scaffolding.

The keep-strategy filtering and group-representative selection
(`calculate_feature`, `_keep_largest_per_group`, `_item_size`) are shared by the text
and image deduplication bases. The per-implementation test suites only exercise
`_find_duplicates`, and the integration tests only use the default ``"first"`` strategy,
so the keep-largest path ("longest" for text, "largest" for image) would otherwise be
untested. These tests pin that behavior down directly on the shared base.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from mloda.user import Feature

from rag_integration.feature_groups.deduplication_base import BaseRowDeduplicator


class _LenDeduplicator(BaseRowDeduplicator):
    """Minimal concrete deduplicator: items live under the ``"item"`` key and rows are
    grouped as duplicates when they share a first element (so duplicates can differ in size).
    """

    @classmethod
    def _extract_items(cls, data: List[Dict[str, Any]], feature: Feature) -> List[Any]:
        return [row["item"] for row in data]

    @classmethod
    def _find_duplicates(cls, items: List[Any], threshold: float) -> List[Optional[int]]:
        first_seen: Dict[Any, int] = {}
        result: List[Optional[int]] = []
        for index, item in enumerate(items):
            key = item[:1]
            if key in first_seen:
                result.append(first_seen[key])
            else:
                first_seen[key] = index
                result.append(None)
        return result


class _LargestLenDeduplicator(_LenDeduplicator):
    """Same as ``_LenDeduplicator`` but keeps the largest item via the image-style value."""

    KEEP_LARGEST_STRATEGY = "largest"


def _features(keep_strategy: str) -> Any:
    """Build a minimal FeatureSet stand-in with a single feature and the given keep strategy."""
    feature = MagicMock()
    feature.name = "items__deduped"
    feature.options.get.side_effect = lambda key: keep_strategy if key == BaseRowDeduplicator.KEEP_STRATEGY else None
    features = MagicMock()
    features.features = [feature]
    return features


def _features_named(*names: str) -> Any:
    """Build a FeatureSet stand-in holding one "first"-strategy feature per given name.

    Repeating a name models the framework placing the same feature in the set more than once
    (e.g. requested directly and again as a downstream input).
    """
    feature_objs = []
    for name in names:
        feature = MagicMock()
        feature.name = name
        feature.options.get.side_effect = lambda key: "first" if key == BaseRowDeduplicator.KEEP_STRATEGY else None
        feature_objs.append(feature)
    features = MagicMock()
    features.features = feature_objs
    return features


class TestBaseRowDeduplicatorKeepStrategies:
    """Cover keep-strategy filtering and representative selection on the shared base."""

    def test_keep_longest_selects_largest_item_per_group(self) -> None:
        """Strategy "longest" keeps the longest string from each duplicate group."""
        data = [{"item": "aa"}, {"item": "aaaa"}, {"item": "bb"}, {"item": "b"}]
        result = _LenDeduplicator.calculate_feature(data, _features("longest"))
        assert [row["items__deduped"] for row in result] == ["aaaa", "bb"]

    def test_keep_largest_uses_length_for_bytes_and_custom_strategy_value(self) -> None:
        """A subclass with KEEP_LARGEST_STRATEGY="largest" keeps the largest bytes per group."""
        data = [{"item": b"aa"}, {"item": b"aaaa"}, {"item": b"bb"}]
        result = _LargestLenDeduplicator.calculate_feature(data, _features("largest"))
        assert [row["items__deduped"] for row in result] == [b"aaaa", b"bb"]

    def test_keep_first_keeps_only_non_duplicate_rows(self) -> None:
        """Strategy "first" keeps the first occurrence of each group and drops the duplicates."""
        data = [{"item": "aa"}, {"item": "aaaa"}, {"item": "bb"}, {"item": "b"}]
        result = _LenDeduplicator.calculate_feature(data, _features("first"))
        assert [row["items__deduped"] for row in result] == ["aa", "bb"]

    def test_all_unique_keeps_all_rows_with_duplicate_metadata(self) -> None:
        """Strategy "all_unique" keeps every row but still annotates duplicate metadata."""
        data = [{"item": "aa"}, {"item": "aaaa"}, {"item": "bb"}, {"item": "b"}]
        result = _LenDeduplicator.calculate_feature(data, _features("all_unique"))
        assert [row["items__deduped"] for row in result] == ["aa", "aaaa", "bb", "b"]
        assert [(row["is_duplicate"], row["duplicate_of"]) for row in result] == [
            (False, None),
            (True, 0),
            (False, None),
            (True, 2),
        ]

    def test_empty_feature_set_returns_data_unchanged(self) -> None:
        """An empty FeatureSet is a no-op: the input rows are returned as-is."""
        data = [{"item": "aa"}, {"item": "bb"}]
        result = _LenDeduplicator.calculate_feature(data, _features_named())
        assert result is data

    def test_repeated_same_feature_is_processed_once(self) -> None:
        """The same feature appearing multiple times (a framework duplicate) collapses by name."""
        data = [{"item": "aa"}, {"item": "aaaa"}, {"item": "bb"}]
        result = _LenDeduplicator.calculate_feature(data, _features_named("items__deduped", "items__deduped"))
        # "first" strategy: one row survives per duplicate group, under the single feature name.
        assert [row["items__deduped"] for row in result] == ["aa", "bb"]

    def test_distinct_features_raise_instead_of_silently_dropping(self) -> None:
        """Deduplication is undefined for >1 distinct feature (row-filtering + shared metadata)."""
        data = [{"item": "aa"}, {"item": "aaaa"}]
        with pytest.raises(ValueError, match="distinct features"):
            _LenDeduplicator.calculate_feature(data, _features_named("a__deduped", "b__deduped"))
