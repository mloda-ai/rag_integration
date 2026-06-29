#!/usr/bin/env python3
"""Swap-backends demo (issue #34).

The open-kgo promise applied to RAG: swapping one connector backend for another
is an edit to the options dict, never a pipeline rewrite. One fixed set of
connector FeatureGroups (:data:`CONNECTORS`) is enabled once, and every run goes
through one invariant call shape (:func:`run_connector`): build a Feature on the
family's root name, hand ``mlodaAPI.run_all`` the options, read the single result
row. Neither the enabled providers nor the calling code changes between runs. A
swap changes only the options dict:

  * within a family: the ``<family>_backend`` selector value (e.g.
    ``retrieve_backend="bm25s"`` -> ``"tfidf"``);
  * across families:  the selector key and the root feature name, with the same
    ``query_text`` / ``corpus`` / ``top_k`` inputs (e.g. a ``retrieve`` connector
    returning ranked passages vs an ``orchestrator`` connector returning an
    answer, over identical inputs).

Selection is unambiguous because each backend gates on its own
``<family>_backend`` value: of all the enabled connectors, exactly one claims a
given options dict (an unknown value claims nothing). That gating is what makes
"enable everything, vary only options" safe.

Run with::

    python -m cli.swap_demo

The pure-Python backends (``tfidf``, ``extractive``, ``template``) always run.
The ``bm25s`` retrieve backend needs ``uv sync --extra connectors``; the
``haystack`` orchestrator backend needs ``uv sync --extra orchestrator``. An
enabled-but-uninstalled backend is harmless until selected (its library imports
lazily), so a missing extra only skips the run that would select it, never
fatal, and the swap story stays visible on any install.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda.provider import FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.connectors.generate import ExtractiveResponder, TemplateResponder
from rag_integration.feature_groups.connectors.orchestrator import HaystackOrchestrator
from rag_integration.feature_groups.connectors.retrieve import Bm25sRetriever, TfidfRetriever

# A tiny corpus where the query shares terms with the pet docs and nothing else,
# so every lexical backend ranks the same two documents above the distractors.
CORPUS: List[Dict[str, str]] = [
    {"doc_id": "d0", "text": "the mat lay flat on the floor by the window"},
    {"doc_id": "d1", "text": "a dog can be a loyal and energetic pet"},
    {"doc_id": "d2", "text": "a cat is an independent and curious pet"},
    {"doc_id": "d3", "text": "cars need regular engine oil and maintenance"},
]
QUERY = "cat pet"
TOP_K = 3

# The inputs that the retrieve and orchestrator families share verbatim. The
# across-family swap adds only a selector key on top of these.
SHARED_INPUTS: Dict[str, Any] = {"query_text": QUERY, "corpus": CORPUS, "top_k": TOP_K}

# Every backend the demo can select, enabled together for every run. The fixed
# set is the point: swapping a backend never touches this, only the options. An
# enabled backend whose library is absent stays dormant (lazy import) unless a
# run actually selects it, which :func:`_run_or_skip` guards.
CONNECTORS: Set[Type[FeatureGroup]] = {
    Bm25sRetriever,
    TfidfRetriever,
    ExtractiveResponder,
    TemplateResponder,
    HaystackOrchestrator,
}


def run_connector(root_feature: str, options: Dict[str, Any]) -> Any:
    """The invariant call shape every connector family shares.

    Build one Feature on the family's root name, run it through ``mlodaAPI`` with
    the whole :data:`CONNECTORS` set enabled, and return the single result row's
    value. Nothing here knows which family or backend it drives: the
    ``<family>_backend`` selector in ``options`` routes the request to exactly
    one enabled connector. That is the swap promise made literal: only ``options``
    changes between calls.
    """
    feature = Feature(root_feature, options=Options(context=dict(options)))
    result = mlodaAPI.run_all(
        [feature],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(CONNECTORS),
    )
    rows: List[Any] = list(result[0]) if result and isinstance(result[0], list) else list(result)
    # Some frameworks nest one level deeper ([[{...}]]); unwrap to match the
    # sibling demos (cli/rag_demo.py, cli/eval_demo.py).
    if rows and isinstance(rows[0], list):
        rows = rows[0]
    for row in rows:
        if isinstance(row, dict) and root_feature in row:
            return row[root_feature]
    raise AssertionError(f"run_all returned no '{root_feature}' row: {result!r}")


def _format_passages(passages: List[Dict[str, Any]]) -> str:
    """One line per ranked passage: ``rank. doc_id (score) text``."""
    if not passages:
        return "    (no passages)"
    lines = [f"    {p['rank']}. {p['doc_id']} ({p['score']:.4f}) {p['text']}" for p in passages]
    return "\n".join(lines)


def _format_answer(answer: Dict[str, Any]) -> str:
    """Render a ``{answer, citations}`` (generate) result."""
    if not answer.get("answer"):
        return "    (no answer)"
    return f"    answer:    {answer['answer']}\n    citations: {answer['citations']}"


def _format_orchestrated(answer: Dict[str, Any]) -> str:
    """Render a ``{answer, documents}`` (orchestrator) result."""
    return f"    answer:    {answer['answer']}\n    documents: {[d['doc_id'] for d in answer['documents']]}"


def _run_or_skip(
    label: str,
    options: Dict[str, Any],
    root_feature: str,
    render: Callable[[Any], str],
    requires: Optional[str] = None,
) -> None:
    """Run one swap, skipping cleanly if its optional extra is not importable.

    ``requires`` names the top-level module the selected backend needs (e.g.
    ``bm25s``, ``haystack``). The probe is a deterministic :func:`find_spec`, not
    a ``try/except ImportError``: a backend imports its library lazily inside
    ``mlodaAPI.run_all``, which wraps the failure as a generic error, so catching
    ``ImportError`` here would miss it and the demo would abort.
    """
    if requires is not None and find_spec(requires) is None:
        print(f"  {label}\n    (skipped: optional extra not installed, no '{requires}' module)")
        return
    print(f"  {label}\n{render(run_connector(root_feature, options))}")


def demo_within_family_retrieve() -> None:
    """Swap the retrieve backend in place: only ``retrieve_backend`` changes."""
    print("\n[A] Within the retrieve family: swap retrieve_backend, identical call shape")
    _run_or_skip(
        'retrieve_backend="bm25s"',
        {"retrieve_backend": "bm25s", **SHARED_INPUTS},
        "retrieved_passages",
        _format_passages,
        requires="bm25s",
    )
    _run_or_skip(
        'retrieve_backend="tfidf"',
        {"retrieve_backend": "tfidf", **SHARED_INPUTS},
        "retrieved_passages",
        _format_passages,
    )


def demo_within_family_generate() -> None:
    """Swap the generate backend in place: only ``generate_backend`` changes."""
    print("\n[B] Within the generate family: swap generate_backend, identical call shape")
    passages = [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in CORPUS]
    _run_or_skip(
        'generate_backend="extractive"',
        {"generate_backend": "extractive", "query_text": QUERY, "passages": passages},
        "generated_answer",
        _format_answer,
    )
    _run_or_skip(
        'generate_backend="template"',
        {"generate_backend": "template", "query_text": QUERY, "passages": passages},
        "generated_answer",
        _format_answer,
    )


def demo_across_families() -> None:
    """Swap across families: same inputs, only the selector key and root name change."""
    print("\n[C] Across families: same query/corpus/top_k, swap selector + root feature name")
    _run_or_skip(
        "retrieve connector     -> retrieved_passages",
        {"retrieve_backend": "tfidf", **SHARED_INPUTS},
        "retrieved_passages",
        _format_passages,
    )
    _run_or_skip(
        "orchestrator connector -> orchestrated_answer",
        {"orchestrator_backend": "haystack", **SHARED_INPUTS},
        "orchestrated_answer",
        _format_orchestrated,
        requires="haystack",
    )


DEMOS: Tuple[Callable[[], None], ...] = (
    demo_within_family_retrieve,
    demo_within_family_generate,
    demo_across_families,
)


def main() -> None:
    print(__doc__.splitlines()[0] if __doc__ else "Swap-backends demo")
    print(f"query={QUERY!r}  top_k={TOP_K}  corpus={len(CORPUS)} docs")
    print(f"enabled connectors (fixed for every run): {sorted(c.__name__ for c in CONNECTORS)}")
    for demo in DEMOS:
        demo()
    print("\nEvery result above came from the same run_connector(...) call over the same enabled")
    print("connectors. Only the options dict changed. A swap is data, not code.")


if __name__ == "__main__":
    main()
