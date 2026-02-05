"""Shared test fixtures and markers."""

from __future__ import annotations

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
    reason="Requires en_core_web_lg spaCy model (install with: pip install 'rag-integration[models]')",
)
