"""Tests for PresidioPIIRedactor option wiring.

The Presidio analyzer is mocked so these tests stay fast and offline; they
exercise the configurable language option (issue: promote hardcoded values).
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import patch

from mloda.user import Feature, Options

from rag_integration.feature_groups.rag_pipeline.pii_redaction.presidio import PresidioPIIRedactor


class _RecordingAnalyzer:
    """Stand-in for a Presidio AnalyzerEngine that records the language used."""

    def __init__(self) -> None:
        self.languages: List[str] = []

    def analyze(self, text: str, entities: List[str], language: str) -> List[Any]:
        self.languages.append(language)
        return []


class TestLanguageOption:
    """Tests for the configurable analyzer language."""

    def test_language_default(self) -> None:
        feature = Feature("docs__pii_redacted", options=Options())
        assert PresidioPIIRedactor._get_language(feature) == PresidioPIIRedactor.DEFAULT_LANGUAGE

    def test_language_from_options(self) -> None:
        feature = Feature("docs__pii_redacted", options=Options(context={"language": "de"}))
        assert PresidioPIIRedactor._get_language(feature) == "de"

    def test_redact_texts_for_feature_threads_language(self) -> None:
        feature = Feature("docs__pii_redacted", options=Options(context={"language": "es"}))
        with patch.object(PresidioPIIRedactor, "_redact_pii", return_value=["x"]) as mock_redact:
            result = PresidioPIIRedactor._redact_texts_for_feature(["hello"], feature)

        assert result == ["x"]
        assert mock_redact.call_args.kwargs["language"] == "es"

    def test_redact_pii_default_language(self) -> None:
        analyzer = _RecordingAnalyzer()
        with patch.object(PresidioPIIRedactor, "_get_analyzer", return_value=analyzer):
            PresidioPIIRedactor._redact_pii(["hello world"], ["ALL"], "mask")
        assert analyzer.languages == ["en"]

    def test_redact_pii_custom_language(self) -> None:
        analyzer = _RecordingAnalyzer()
        with patch.object(PresidioPIIRedactor, "_get_analyzer", return_value=analyzer):
            PresidioPIIRedactor._redact_pii(["hola mundo"], ["ALL"], "mask", language="es")
        assert analyzer.languages == ["es"]

    def test_empty_text_skips_analysis(self) -> None:
        analyzer = _RecordingAnalyzer()
        with patch.object(PresidioPIIRedactor, "_get_analyzer", return_value=analyzer):
            result = PresidioPIIRedactor._redact_pii([""], ["ALL"], "mask")
        assert result == [""]
        assert analyzer.languages == []
