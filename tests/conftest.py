"""Shared test fixtures and markers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_integration.feature_groups.rag_pipeline import artifact_storage


@pytest.fixture(autouse=True)
def _isolated_default_artifact_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep artifacts saved without artifact_storage_path out of the real user cache."""
    default_dir: Path = tmp_path_factory.mktemp("default_artifacts")
    monkeypatch.setattr(artifact_storage, "default_storage_dir", lambda: default_dir)


def _spacy_model_available(model_name: str) -> bool:
    """Check if a spaCy model is installed and loadable."""
    try:
        import spacy

        spacy.load(model_name)
        return True
    except Exception:  # noqa: BLE001 - any failure means the model is unavailable
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

requires_flashrank_model = pytest.mark.skipif(
    RUNNING_ON_CI,
    reason="Skipped on CI: FlashRank downloads its (~4 MB) ONNX model from the network on first use.",
)
