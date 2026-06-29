"""Honest-surface lock for :class:`FlashRankReranker`.

Imports the class only (no FlashRank model, no network), so it always runs on
CI: it pins the backend's honest surface, that it advertises no model option it
would then ignore. The model-running contract test lives in
``test_flashrank_reranker.py`` and is CI-skipped on model download.
"""

from __future__ import annotations

from rag_integration.feature_groups.connectors.rerank.flashrank_reranker import FlashRankReranker


def test_no_model_option_advertised() -> None:
    """The model is fixed to ``DEFAULT_MODEL``; no model option is exposed, so the
    backend never advertises a knob it cannot honor (the base ``_rank`` contract
    passes no options)."""
    advertised = {key for key in FlashRankReranker.PROPERTY_MAPPING if "model" in key.lower()}
    assert not advertised, f"FlashRankReranker must not advertise a model option: {advertised}"
    assert FlashRankReranker.DEFAULT_MODEL == "ms-marco-TinyBERT-L-2-v2"
