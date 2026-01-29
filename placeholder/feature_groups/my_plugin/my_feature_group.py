"""Example FeatureGroup implementation."""

from typing import Any

from mloda.provider import FeatureGroup


class MyFeatureGroup(FeatureGroup):
    """Example FeatureGroup - rename and customize for your use case."""

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> Any:
        """Calculate and return feature data."""
        return {"example": "data"}
