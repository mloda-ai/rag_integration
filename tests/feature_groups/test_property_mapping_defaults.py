"""Repo-wide invariant: every PROPERTY_MAPPING default must be an accepted value.

mloda enforces this at class-definition time (``FeatureChainParser.validate_property_mapping_defaults``
in ``FeatureGroup.__init_subclass__``), so a violation is already an import error. The sweep still
earns its place: it pins that every feature group is reachable and actually reached, which is what
stops the invariant from passing vacuously.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from mloda.provider import DefaultOptionKeys, FeatureChainParser, FeatureGroup

import rag_integration.feature_groups


def _validate(owner_name: str, property_mapping: Optional[Dict[str, Any]]) -> List[str]:
    """Run the upstream validator; returns violation messages."""
    try:
        FeatureChainParser.validate_property_mapping_defaults(owner_name, property_mapping)
    except (ValueError, TypeError) as exc:
        return [str(exc)]
    return []


def _all_feature_groups() -> List[Type[FeatureGroup]]:
    """Import every feature_groups module and collect this package's FeatureGroup subclasses.

    Any import failure fails the test: a module that cannot import is a module whose
    PROPERTY_MAPPING this invariant silently skips.
    """
    import_failures: List[str] = []
    for module_info in pkgutil.walk_packages(
        rag_integration.feature_groups.__path__,
        prefix="rag_integration.feature_groups.",
        onerror=lambda name: import_failures.append(f"{name}: failed during package walk"),
    ):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:
            import_failures.append(f"{module_info.name}: {exc!r}")
    assert not import_failures, "feature_groups modules failed to import:\n" + "\n".join(import_failures)

    collected: List[Type[FeatureGroup]] = []
    stack: List[Type[FeatureGroup]] = list(FeatureGroup.__subclasses__())
    seen: set[Type[FeatureGroup]] = set()
    while stack:
        candidate = stack.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        stack.extend(candidate.__subclasses__())
        if candidate.__module__.startswith("rag_integration."):
            collected.append(candidate)
    return sorted(collected, key=lambda c: f"{c.__module__}.{c.__name__}")


def test_validator_catches_bad_default() -> None:
    """Guard against a vacuously green check: a default outside the accepted values must be flagged."""
    bad_mapping: Dict[str, Any] = {
        "mode": {
            DefaultOptionKeys.allowed_values: {"fast": "fast mode", "slow": "slow mode"},
            DefaultOptionKeys.default: "turbo",
            DefaultOptionKeys.strict_validation: True,
        }
    }
    assert _validate("DummyOwner", bad_mapping)


def test_validator_catches_default_rejected_by_element_validator() -> None:
    bad_mapping: Dict[str, Any] = {
        "size": {
            "explanation": "positive size",
            DefaultOptionKeys.default: -1,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.element_validator: lambda value: isinstance(value, int) and value > 0,
        }
    }
    assert _validate("DummyOwner", bad_mapping)


def test_all_property_mapping_defaults_are_accepted_values() -> None:
    feature_groups = _all_feature_groups()
    # 74 feature groups exist today; lower this only when groups are deliberately removed.
    assert len(feature_groups) >= 74, f"feature group discovery looks broken, found {len(feature_groups)}"

    violations: List[str] = []
    for feature_group in feature_groups:
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        violations.extend(_validate(owner, feature_group.PROPERTY_MAPPING))

    assert not violations, "PROPERTY_MAPPING defaults outside accepted values:\n" + "\n".join(violations)
