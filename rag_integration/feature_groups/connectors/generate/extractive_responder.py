"""Extractive (no-LLM) responder.

Canonical concrete for the ``generate`` family: zero-download, zero-dependency
(pure Python stdlib), deterministic. Selects the passage sentence most relevant
to the query (by token overlap) and returns it verbatim as the answer, citing
the passage it came from. A grounded-by-construction baseline that anchors the
CI contract suite; LLM-backed generators are pedigree backends for later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.generate._text import SENTENCE_RE, tokenize
from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector


class ExtractiveResponder(BaseGenerateConnector):
    """Extractive responder (``generate_backend="extractive"``).

    Splits each passage into sentences, scores every sentence by the number of
    distinct query tokens it contains, and returns the single best sentence as
    the answer (cited to its passage). Ties are broken by passage then sentence
    order, so the output is stable and deterministic. If no sentence shares any
    token with the query, the answer is empty with no citations (the responder
    does not invent an answer).

    Baseline limitations (acceptable for a zero-dependency CI anchor): the
    tokenizer matches ``[a-z0-9]+``, so it is English/ASCII-only (accented or
    non-Latin text scores low or zero); the sentence splitter is punctuation
    based (``[.!?]``), so it over-splits abbreviations and keeps embedded
    newlines verbatim. There is no stopword handling, so sentences echoing the
    question's function words can outrank the substantive answer. A
    higher-fidelity or multilingual responder would be a separate backend.
    """

    GENERATE_BACKENDS = {
        "extractive": "Extractive sentence selection (pure Python, no LLM)",
    }

    PROPERTY_MAPPING = {
        BaseGenerateConnector.GENERATE_BACKEND: property_spec(
            "Use 'extractive' for no-LLM sentence extraction", context=False
        ),
        BaseGenerateConnector.QUERY_TEXT: property_spec("The question to answer", context=False),
        BaseGenerateConnector.PASSAGES: property_spec(
            "Supporting passages: a list of {doc_id, text} dicts", context=False
        ),
    }

    @classmethod
    def _generate(cls, query: str, passages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        query_tokens = tokenize(query)

        best_score = 0
        best_sentence = ""
        best_doc_id = ""
        for i, passage in enumerate(passages):
            doc_id = str(passage.get("doc_id", str(i)))
            for raw_sentence in SENTENCE_RE.findall(str(passage.get("text", ""))):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                score = len(query_tokens & tokenize(sentence))
                # Strictly greater keeps the first (earliest) best on ties.
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
                    best_doc_id = doc_id

        if best_score == 0:
            return "", []
        return best_sentence, [best_doc_id]
