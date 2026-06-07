"""Shared test fixtures and markers."""

from __future__ import annotations

import os

import pytest


def _spacy_model_available(model_name: str) -> bool:
    """Check if a spaCy model is installed and loadable."""
    try:
        import spacy

        spacy.load(model_name)
        return True
    except Exception:
        return False


requires_spacy_model = pytest.mark.skipif(
    not _spacy_model_available("en_core_web_lg"),
    reason="Requires en_core_web_lg spaCy model (install with: uv pip install -r requirements-models.txt)",
)


# Skip integration tests that download sentence-transformers models from the Hugging Face Hub
# at runtime when running on CI. The Hub rate-limits with HTTP 429, which makes those tests
# flaky on CI runners. GitHub Actions sets CI=true (forwarded into tox via passenv); locally
# CI is unset, so the tests run against the cached model.
RUNNING_ON_CI = os.environ.get("CI", "").lower() == "true"

requires_sentence_transformer_model = pytest.mark.skipif(
    RUNNING_ON_CI,
    reason="Skipped on CI: downloads a sentence-transformers model from the Hugging Face Hub (rate-limited, HTTP 429).",
)
