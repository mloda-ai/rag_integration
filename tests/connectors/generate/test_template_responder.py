"""Contract test for :class:`TemplateResponder` (zero-download CI anchor).

Inherits the whole generate contract suite, then adds a backend-specific proof:
unlike the single-citation extractive responder, this backend cites *every*
passage it drew a sentence from. The fixture is crafted so the top sentences
span two passages, so the multi-citation behaviour is exercised.
"""

from __future__ import annotations

from typing import Any, Dict, List, Type

from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector
from rag_integration.feature_groups.connectors.generate.template_responder import TemplateResponder
from tests.connectors.generate.generate_contract import GenerateConnectorContractBase


class TestTemplateResponder(GenerateConnectorContractBase):
    @classmethod
    def connector_class(cls) -> Type[BaseGenerateConnector]:
        return TemplateResponder

    @classmethod
    def backend_value(cls) -> str:
        return "template"

    @classmethod
    def sample_passages(cls) -> List[Dict[str, Any]]:
        return [
            {"doc_id": "d0", "text": "Cars need regular engine oil and maintenance."},
            {"doc_id": "d1", "text": "A cat needs fresh water every day. The cat also needs a clean box."},
            {"doc_id": "d2", "text": "Good cat food keeps a cat strong."},
        ]

    @classmethod
    def sample_query(cls) -> str:
        # Deliberately relevant to both d1 and d2 (not just one passage) so the
        # multi-citation behaviour below is exercised.
        return "cat water food"

    @classmethod
    def expected_citation_doc_id(cls) -> str:
        return "d1"

    @classmethod
    def expected_answer_substring(cls) -> str:
        return "fresh water"

    # -- Backend-specific proof: multi-passage citation -----------------------

    def test_cites_every_contributing_passage(self) -> None:
        """The distinguishing behaviour vs the extractive responder: when the
        top sentences span several passages, each is cited (not just one)."""
        result = self._answer(self.sample_query(), self.sample_passages())
        assert set(result["citations"]) == {"d1", "d2"}, result["citations"]
        assert len(result["citations"]) > 1, "template responder must cite every contributing passage"

    def test_answer_uses_fixed_template_and_cited_sources(self) -> None:
        """The answer is exactly the fixed template lead-in followed by the
        deterministic best-first sentence selection, verbatim. An exact-equality
        check on the residual body, so no invented text can sneak in."""
        result = self._answer(self.sample_query(), self.sample_passages())
        prefix = "Based on the retrieved passages: "
        assert result["answer"].startswith(prefix)
        body = result["answer"].removeprefix(prefix)
        # Score 2 sentences first (passage order: d1 before d2), then score 1.
        assert body == (
            "A cat needs fresh water every day. Good cat food keeps a cat strong. The cat also needs a clean box."
        )

    def test_no_relevant_sentence_returns_empty(self) -> None:
        """Passages present but no sentence shares a query token: the responder
        returns an empty answer with no citations (never a bare template), so the
        base's 'non-empty answer requires citations' guard is never tripped."""
        result = self._answer("zzz nonmatching query", self.sample_passages())
        assert result == {"answer": "", "citations": []}

    def test_caps_answer_at_max_sentences(self) -> None:
        """Honest surface: the answer draws at most ``MAX_SENTENCES`` sentences,
        even when more passages are relevant. Four single-sentence passages all
        match equally, but only the first three (by passage order) are selected,
        so the fourth is neither answered nor cited."""
        passages = [
            {"doc_id": "p0", "text": "A cat naps."},
            {"doc_id": "p1", "text": "A cat purrs."},
            {"doc_id": "p2", "text": "A cat hunts."},
            {"doc_id": "p3", "text": "A cat climbs."},
        ]
        result = self._answer("cat", passages)
        assert TemplateResponder.MAX_SENTENCES == 3
        assert result["citations"] == ["p0", "p1", "p2"], result["citations"]
        assert "climbs" not in result["answer"]
