"""Repo-wide invariant: every PROPERTY_MAPPING default must be an accepted value.

Upstream mloda will enforce this at class-definition time via
``FeatureChainParser.validate_property_mapping_defaults`` in ``FeatureGroup.__init_subclass__``.
Enforce it here first: delegate to that validator when the installed mloda has it,
otherwise replicate it on the ``FeatureChainParser`` helpers present in 0.8.x.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from mloda.provider import DefaultOptionKeys, FeatureChainParser, FeatureGroup

import rag_integration.feature_groups


def _validate_with_replica(owner_name: str, property_mapping: Optional[Dict[str, Any]]) -> List[str]:
    """Replicate upstream validate_property_mapping_defaults; returns violation messages."""
    violations: List[str] = []
    if property_mapping is None:
        return violations
    for key, spec in property_mapping.items():
        if not isinstance(spec, dict):
            continue
        if DefaultOptionKeys.default not in spec:
            continue
        default = spec[DefaultOptionKeys.default]
        if default is None:
            continue
        validation_function = FeatureChainParser._get_validation_function(spec)
        if validation_function is not None:
            if not FeatureChainParser._is_strict_validation(spec):
                continue
            try:
                verdict = validation_function(default)
            except Exception as exc:
                violations.append(
                    f"{owner_name}.PROPERTY_MAPPING['{key}'] default {default!r}: validation_function raised {exc!r}"
                )
                continue
            if not verdict:
                violations.append(
                    f"{owner_name}.PROPERTY_MAPPING['{key}'] default {default!r}: rejected by validation_function"
                )
            continue
        accepted = FeatureChainParser._extract_property_values(spec)
        try:
            FeatureChainParser._validate_property_value(default, accepted, key, spec)
        except (ValueError, TypeError):
            violations.append(
                f"{owner_name}.PROPERTY_MAPPING['{key}'] default {default!r}: "
                f"not in accepted values {sorted(accepted, key=repr)}"
            )
    return violations


def _validate(owner_name: str, property_mapping: Optional[Dict[str, Any]]) -> List[str]:
    """Delegate to the upstream validator when available, else use the replica."""
    upstream = getattr(FeatureChainParser, "validate_property_mapping_defaults", None)
    if upstream is None:
        return _validate_with_replica(owner_name, property_mapping)
    try:
        upstream(owner_name, property_mapping)
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
            "fast": "fast mode",
            "slow": "slow mode",
            DefaultOptionKeys.default: "turbo",
            DefaultOptionKeys.strict_validation: True,
        }
    }
    assert _validate("DummyOwner", bad_mapping)


def test_validator_catches_default_rejected_by_validation_function() -> None:
    bad_mapping: Dict[str, Any] = {
        "size": {
            "explanation": "positive size",
            DefaultOptionKeys.default: -1,
            DefaultOptionKeys.strict_validation: True,
            DefaultOptionKeys.validation_function: lambda value: isinstance(value, int) and value > 0,
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
