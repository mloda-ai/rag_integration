"""Tests for MyComputeFramework."""

from placeholder.compute_frameworks.my_plugin import MyComputeFramework
from mloda.provider import ComputeFramework


def test_extends_base() -> None:
    """MyComputeFramework should extend ComputeFramework."""
    assert issubclass(MyComputeFramework, ComputeFramework)


def test_instantiation() -> None:
    """MyComputeFramework should instantiate with no arguments."""
    instance = MyComputeFramework()
    assert instance is not None
