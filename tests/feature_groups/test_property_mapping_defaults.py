"""Repo-wide invariant: every PROPERTY_MAPPING value must be a valid PropertySpec.

mloda enforces this at class-definition time (``FeatureChainParser.validate_property_mapping_defaults``
in ``FeatureGroup.__init_subclass__``), so a violation is already an import error. The sweep still
earns its place: it pins that every feature group is reachable and actually reached, which is what
stops the invariant from passing vacuously.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

import pytest
from mloda.provider import FeatureChainParser, FeatureGroup, PropertySpec

import rag_integration.feature_groups


def _validate(owner_name: str, property_mapping: dict[str, Any] | None) -> list[str]:
    """Run the upstream validator; returns violation messages."""
    try:
        FeatureChainParser.validate_property_mapping_defaults(owner_name, property_mapping)
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def _all_feature_groups() -> list[type[FeatureGroup]]:
    """Import every feature_groups module and collect this package's FeatureGroup subclasses.

    Any import failure fails the test: a module that cannot import is a module whose
    PROPERTY_MAPPING this invariant silently skips.
    """
    import_failures: list[str] = []
    for module_info in pkgutil.walk_packages(
        rag_integration.feature_groups.__path__,
        prefix="rag_integration.feature_groups.",
        onerror=lambda name: import_failures.append(f"{name}: failed during package walk"),
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # noqa: BLE001 - any import failure must fail the assert below
            import_failures.append(f"{module_info.name}: {exc!r}")
    assert not import_failures, "feature_groups modules failed to import:\n" + "\n".join(import_failures)

    collected: list[type[FeatureGroup]] = []
    stack: list[type[FeatureGroup]] = list(FeatureGroup.__subclasses__())
    seen: set[type[FeatureGroup]] = set()
    while stack:
        candidate = stack.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        stack.extend(candidate.__subclasses__())
        if candidate.__module__.startswith("rag_integration."):
            collected.append(candidate)
    return sorted(collected, key=lambda c: f"{c.__module__}.{c.__name__}")


def test_validator_catches_raw_dict_spec() -> None:
    """Guard against reverting to the retired dict form: validate_property_mapping_defaults must catch it."""
    bad_mapping: dict[str, Any] = {"mode": {"allowed_values": {"fast": "fast mode"}}}
    assert _validate("DummyOwner", bad_mapping)


def test_property_spec_rejects_default_outside_allowed_values() -> None:
    """Guard against a default outside allowed_values silently passing; PropertySpec.__post_init__ must flag it."""
    with pytest.raises(ValueError):
        PropertySpec(
            "mode",
            allowed_values={"fast": "fast mode", "slow": "slow mode"},
            default="turbo",
            strict_validation=True,
        )


def test_property_spec_rejects_default_failing_element_validator() -> None:
    with pytest.raises(ValueError):
        PropertySpec(
            "size",
            default=-1,
            strict_validation=True,
            element_validator=lambda value: isinstance(value, int) and value > 0,
        )


def test_group_context_split_is_stable() -> None:
    """Connector options resolve the backend, so they are group; every other family is context.

    mloda reads a missing ``context`` key as group, while ``property_spec`` defaults it to True, so a
    spec authored without an explicit ``context=False`` silently flips a connector option to context
    and changes feature-group resolution and hashing. Pin the split so that flip cannot land quietly.
    """
    misplaced: list[str] = []
    for feature_group in _all_feature_groups():
        is_connector = ".connectors." in feature_group.__module__
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        for key, spec in (feature_group.PROPERTY_MAPPING or {}).items():
            assert isinstance(spec, PropertySpec), f"{owner}.{key} is a {type(spec).__name__}, not a PropertySpec"
            is_context = spec.context
            if is_context is is_connector:
                expected = "group (pass context=False)" if is_connector else "context"
                misplaced.append(f"{owner}.{key}: expected {expected}")

    assert not misplaced, "PROPERTY_MAPPING options on the wrong side of the group/context split:\n" + "\n".join(
        misplaced
    )


def test_all_property_mapping_values_are_property_specs() -> None:
    feature_groups = _all_feature_groups()
    # 74 feature groups exist today; lower this only when groups are deliberately removed.
    assert len(feature_groups) >= 74, f"feature group discovery looks broken, found {len(feature_groups)}"

    violations: list[str] = []
    for feature_group in feature_groups:
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        violations.extend(_validate(owner, feature_group.PROPERTY_MAPPING))

    assert not violations, "PROPERTY_MAPPING values that are not valid PropertySpec objects:\n" + "\n".join(violations)
