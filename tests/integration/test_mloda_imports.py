"""Tests to verify mloda dependencies can be imported."""


def test_mloda_provider_imports() -> None:
    """Verify mloda.provider module imports work."""
    from mloda.provider import ComputeFramework, FeatureGroup

    assert FeatureGroup is not None
    assert ComputeFramework is not None


def test_mloda_steward_imports() -> None:
    """Verify the public mloda.steward Extender import works."""
    from mloda.steward import Extender

    assert Extender is not None


def test_mloda_testing_imports() -> None:
    """Verify mloda.testing module imports work."""
    from mloda.testing.base import FeatureGroupTestBase

    assert FeatureGroupTestBase is not None
