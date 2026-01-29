"""Tests to verify mloda dependencies can be imported."""


def test_mloda_provider_imports() -> None:
    """Verify mloda.provider module imports work."""
    from mloda.provider import FeatureGroup, ComputeFramework

    assert FeatureGroup is not None
    assert ComputeFramework is not None


def test_mloda_core_imports() -> None:
    """Verify mloda.core module imports work."""
    from mloda.core.abstract_plugins.function_extender import Extender

    assert Extender is not None


def test_mloda_testing_imports() -> None:
    """Verify mloda.testing module imports work."""
    from mloda.testing import FeatureGroupTestBase

    assert FeatureGroupTestBase is not None
