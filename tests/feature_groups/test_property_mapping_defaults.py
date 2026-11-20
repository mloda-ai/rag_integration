"""Repo-wide sweep over every feature group: PROPERTY_MAPPING invariants, mloda's shape check, subclass scope.

mloda enforces valid PropertySpecs at class-definition time, so that violation is already an import
error. The sweep still earns its place: it pins that every feature group is reachable and actually
reached, which is what stops the invariants from passing vacuously.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

import pytest

# mloda 0.20 ships these checks under the private _testing package only.
from mloda._testing.feature_group_shape import check_feature_group_shape
from mloda._testing.subclass_scope import check_subclass_scope
from mloda.provider import FeatureChainParser, FeatureChainParserMixin, FeatureGroup, PropertySpec, is_no_default
from mloda.user import Options

import rag_integration.feature_groups


def _validate(owner_name: str, property_mapping: Mapping[str, Any] | None) -> list[str]:
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


def test_plain_groups_require_only_backend_keys() -> None:
    """mloda requires no-default keys when a plain group matches; only connector backends may be required.

    The other options are validated in ``calculate_feature``, so a key without ``default=None`` would make
    the group stop matching instead of raising a clear option error.
    """
    required: list[str] = []
    for feature_group in _all_feature_groups():
        if issubclass(feature_group, FeatureChainParserMixin):
            continue
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        for key, spec in (feature_group.PROPERTY_MAPPING or {}).items():
            if is_no_default(spec.default) and not str(key).endswith("_backend"):
                required.append(f"{owner}.{key}")

    assert not required, "plain feature group keys missing default=None:\n" + "\n".join(required)


def test_all_property_mapping_values_are_property_specs() -> None:
    feature_groups = _all_feature_groups()
    # 74 feature groups exist today; lower this only when groups are deliberately removed.
    assert len(feature_groups) >= 74, f"feature group discovery looks broken, found {len(feature_groups)}"

    violations: list[str] = []
    for feature_group in feature_groups:
        owner = f"{feature_group.__module__}.{feature_group.__name__}"
        violations.extend(_validate(owner, feature_group.PROPERTY_MAPPING))

    assert not violations, "PROPERTY_MAPPING values that are not valid PropertySpec objects:\n" + "\n".join(violations)


# Groups matching fixed names (and the options their name needs), keyed by family base.
_FIXED_CASES: dict[str, tuple[list[str], dict[str, Any]]] = {
    "BaseDocumentSource": (["docs"], {}),
    "BaseImageSource": (["image_docs"], {}),
    "BaseTextDatasetSource": (["eval_docs"], {}),
    "BaseImageDatasetSource": (["eval_images"], {}),
    "BaseLLMResponse": (["generated_answer", "llm_response"], {"query": "q"}),
    "BaseRetriever": (["retrieved", "retrieved_passages"], {"index_path": "index"}),
}

_INERT = ("names:", "required:")
# {group: (reason, prefixes every failure must start with)}; a waived group that passes is stale.
_SHAPE_WAIVERS: dict[str, tuple[str, tuple[str, ...]]] = {
    **{
        name: ("inert connector base: empty backend set, matches nothing by design", _INERT)
        for name in (
            "BaseGenerateConnector",
            "BaseGraphRagConnector",
            "BaseKnowledgeGraphSource",
            "BaseOrchestratorConnector",
            "BaseRerankConnector",
            "BaseRetrieveConnector",
            "BaseStructuredConnector",
        )
    },
    "BaseRowDeduplicator": ("shared abstract base without a pattern", ("names:", "inputs:")),
}

# Siblings the scope check cannot cover, with why.
_SCOPE_WAIVERS: dict[str, str] = {
    "DictDocumentSource": "takes no options; a pipeline enables one document source",
    "FileDocumentSource": "takes no options; a pipeline enables one document source",
    "DictImageSource": "takes no options; a pipeline enables one image source",
    "FileImageSource": "takes no options; a pipeline enables one image source",
    "ScifactDatasetSource": "the only text dataset",
    "Flickr30kDatasetSource": "the only image dataset",
    "BaseDeduplicator": "intermediate base, its siblings are checked against it",
    "BaseImageDeduplicator": "intermediate base, its siblings are checked against it",
    "FaissFlatIndexer": "family default: serves docs__indexed when index_method is unset",
    "ClaudeCliResponse": "the only LLM stage; llm_method is optional and generated_answer needs a query option",
    "FaissRetriever": "the only retrieval stage; retrieval_method is optional and retrieved_passages needs index_path",
    **{
        f"FullCoverage{method}PIIRedactor": "takes over full pii_types coverage by design (match_guard)"
        for method in ("Pattern", "Presidio", "Regex", "Simple")
    },
}


def _backend_key(fg: type[FeatureGroup]) -> str | None:
    if ".connectors." not in fg.__module__:
        return None
    return next((str(k) for k in fg.PROPERTY_MAPPING or {} if str(k).endswith("_backend")), None)


def _backends(fg: type[FeatureGroup], key: str) -> Mapping[str, str]:
    backends: Mapping[str, str] = getattr(fg, key.upper() + "S")
    return backends


def _requests(fg: type[FeatureGroup]) -> list[str] | None:
    match = re.fullmatch(r"\.\*(__\w+)\$", getattr(fg, "PREFIX_PATTERN", None) or "")
    return [f"docs{match.group(1)}"] if match else None


def _shape_case(fg: type[FeatureGroup]) -> tuple[list[str] | None, Options | None]:
    """The request names and base options a group's shape check needs."""
    for base in fg.__mro__:
        if base.__name__ in _FIXED_CASES:
            names, group = _FIXED_CASES[base.__name__]
            return names, Options(group=group) if group else None
    key = _backend_key(fg)
    if key is not None:
        backends = _backends(fg, key)
        options = Options(group={key: next(iter(backends))}) if backends else None
        return sorted(fg.feature_names_supported()), options
    return _requests(fg), None


def test_every_feature_group_passes_the_shape_check() -> None:
    failures: list[str] = []
    stale: list[str] = []
    for feature_group in _all_feature_groups():
        requests, base = _shape_case(feature_group)
        result = check_feature_group_shape(feature_group, base, requests=requests)
        waiver = _SHAPE_WAIVERS.get(feature_group.__name__)
        if waiver is None:
            failures += [f"{feature_group.__name__}: {f}" for f in result.failures]
        elif not result.failures or not all(f.startswith(waiver[1]) for f in result.failures):
            stale.append(f"{feature_group.__name__}: {list(result.failures) or 'passes'}")

    assert not failures, "feature groups failing mloda's shape check:\n" + "\n".join(failures)
    assert not stale, "shape waivers that no longer describe the failure:\n" + "\n".join(stale)


def test_backend_siblings_stay_in_scope() -> None:
    """Each sibling takes over only requests carrying its own discriminator value.

    A ``deferred_binding`` selector is exempt on the name path by design, so a lone enabled sibling
    also serves a request without it; that one probe is tolerated, every other-value probe is not.
    """
    groups = _all_feature_groups()
    inside = set(groups)
    failures: list[str] = []
    waived: set[str] = set()
    checked = 0
    for sub in groups:
        parent = next((b for b in sub.__mro__[1:] if b in inside), None)
        if parent is None or _backend_key(sub) is not None:
            continue
        if sub.__name__ in _SCOPE_WAIVERS:
            waived.add(sub.__name__)
            continue
        wider = parent.PROPERTY_MAPPING or {}
        selector = {
            key: spec
            for key, spec in (sub.PROPERTY_MAPPING or {}).items()
            if spec.allowed_values and key in wider and set(spec.allowed_values) != set(wider[key].allowed_values or ())
        }
        if not selector:
            failures.append(f"{sub.__name__} < {parent.__name__}: no discriminator value and no waiver")
            continue
        group = {k: next(iter(s.allowed_values or ())) for k, s in selector.items() if not s.context}
        context = {k: next(iter(s.allowed_values or ())) for k, s in selector.items() if s.context}
        requests = _requests(parent) or sorted(parent.feature_names_supported())
        result = check_subclass_scope(sub, parent, Options(group=group, context=context), requests)
        tolerated = {
            f"steals: '{name}' without {sorted(selector)} resolves to {sub.__name__}"
            for name in requests
            if all(spec.deferred_binding for spec in selector.values())
        }
        failures += [f"{sub.__name__} < {parent.__name__}: {f}" for f in result.failures if f not in tolerated]
        checked += 1

    assert not failures, "siblings claiming requests outside their discriminator:\n" + "\n".join(failures)
    assert waived == set(_SCOPE_WAIVERS), f"scope waivers for unknown groups: {sorted(set(_SCOPE_WAIVERS) - waived)}"
    # 29 sibling pairs are checked today; lower this only when siblings are deliberately removed.
    assert checked >= 29, f"subclass scope discovery looks broken, checked {checked}"


def test_connector_backend_values_are_disjoint() -> None:
    """Connector siblings are selected by ``backend in BACKENDS``, so disjoint value sets keep them exclusive."""
    families: dict[type[FeatureGroup], Counter[str]] = {}
    for feature_group in _all_feature_groups():
        key = _backend_key(feature_group)
        if key is None:
            continue
        base = next(b for b in reversed(feature_group.__mro__) if key in (getattr(b, "PROPERTY_MAPPING", None) or {}))
        if base is feature_group:
            assert not _backends(feature_group, key), f"{feature_group.__name__}: a family base must stay inert"
            families.setdefault(feature_group, Counter())
            continue
        backends = _backends(feature_group, key)
        assert backends, f"{feature_group.__name__} declares no {key} value"
        families.setdefault(base, Counter()).update(list(backends))

    shared = [f"{base.__name__}: {value}" for base, counts in families.items() for value, n in counts.items() if n > 1]
    assert not shared, "backend values claimed by more than one sibling:\n" + "\n".join(shared)
    assert len(families) >= 7, f"connector family discovery looks broken, found {len(families)}"
