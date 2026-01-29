"""Tests for MyFeatureGroup."""

from placeholder.feature_groups.my_plugin import MyFeatureGroup
from mloda.provider import FeatureGroup


def test_extends_base() -> None:
    """MyFeatureGroup should extend FeatureGroup."""
    assert issubclass(MyFeatureGroup, FeatureGroup)


def test_calculate_feature() -> None:
    """calculate_feature should return example data."""
    result = MyFeatureGroup.calculate_feature(None, None)
    assert result == {"example": "data"}
