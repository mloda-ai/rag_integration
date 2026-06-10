"""Repo-wide invariant: every PROPERTY_MAPPING default must be an accepted value.

Upstream mloda is adding ``FeatureChainParser.validate_property_mapping_defaults``,
called from ``FeatureGroup.__init_subclass__``, which rejects at class-definition
time any strict default that is not in the key's accepted values. Once that mloda
release ships, a violating feature group fails on import. This test enforces the
same invariant now so violations are caught here first. When the installed mloda
already exposes the validator, the test delegates to it; otherwise it replicates
the upstream logic on top of the ``FeatureChainParser`` helpers present in 0.8.x.
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
    """Import every feature_groups module and collect this package's FeatureGroup subclasses."""
    for module_info in pkgutil.walk_packages(
        rag_integration.feature_groups.__path__, prefix="rag_integration.feature_groups."
    ):
        try:
            importlib.import_module(module_info.name)
        except ImportError:
            # Optional-dependency modules (e.g. model backends) are exercised by their own tests.
            continue

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


def test_all_property_mapping_defaults_are_accepted_values() -> None:
    feature_groups = _all_feature_groups()
    assert len(feature_groups) >= 70, "feature group discovery looks broken"

    violations: List[str] = []
    for feature_group in feature_groups:
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        violations.extend(_validate(owner, feature_group.PROPERTY_MAPPING))

    assert not violations, "PROPERTY_MAPPING defaults outside accepted values:\n" + "\n".join(violations)
