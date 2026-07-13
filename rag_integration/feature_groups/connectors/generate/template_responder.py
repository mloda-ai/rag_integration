"""Template (no-LLM) responder.

Second concrete for the ``generate`` family: zero-download, zero-dependency
(pure Python stdlib), deterministic. Where :class:`ExtractiveResponder` returns
a single best sentence cited to one passage, this backend selects the top-N
query-relevant sentences *across* passages, joins them into a fixed template,
and cites **every** passage it drew from (multi-citation). Grounded by
construction: the answer is a fixed lead-in plus verbatim source sentences, and
each contributing passage is cited.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mloda.provider import property_spec

from rag_integration.feature_groups.connectors.generate._text import SENTENCE_RE, tokenize
from rag_integration.feature_groups.connectors.generate.base import BaseGenerateConnector

# Fixed lead-in the selected sentences are joined onto. The answer is this
# template plus verbatim source sentences, so it stays grounded.
_TEMPLATE_PREFIX = "Based on the retrieved passages: "


class TemplateResponder(BaseGenerateConnector):
    """Multi-sentence template responder (``generate_backend="template"``).

    Splits every passage into sentences, scores each by the number of distinct
    query tokens it contains, and keeps the top ``MAX_SENTENCES`` sentences with
    a non-zero score. Those sentences are joined (in best-first order) onto a
    fixed template, and every passage that contributed a sentence is cited.

    Ties are broken by passage order then sentence order, so the selection,
    answer text, and citation order are all stable and deterministic. If no
    sentence shares a token with the query, the answer is empty with no
    citations (the responder does not invent an answer).

    Baseline limitations (shared with :class:`ExtractiveResponder` via the same
    tokenizer and sentence splitter): English/ASCII-only matching and
    punctuation-based splitting. There is no stopword handling, so sentences
    echoing the question's function words can outrank the substantive answer.
    """

    # How many sentences the answer may draw together. A handful is enough to
    # surface multi-passage support while keeping the answer focused.
    MAX_SENTENCES = 3

    GENERATE_BACKENDS = {
        "template": "Top-N sentence templating with multi-passage citation (pure Python, no LLM)",
    }

    PROPERTY_MAPPING = {
        BaseGenerateConnector.GENERATE_BACKEND: property_spec(
            "Use 'template' for multi-sentence templated answers", context=False
        ),
        BaseGenerateConnector.QUERY_TEXT: property_spec("The question to answer", context=False),
        BaseGenerateConnector.PASSAGES: property_spec(
            "Supporting passages: a list of {doc_id, text} dicts", context=False
        ),
    }

    @classmethod
    def _generate(cls, query: str, passages: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
        query_tokens = tokenize(query)

        # (score, passage_index, sentence_index, sentence, doc_id) for every
        # sentence that shares at least one distinct query token.
        scored: List[Tuple[int, int, int, str, str]] = []
        for passage_index, passage in enumerate(passages):
            doc_id = str(passage.get("doc_id", str(passage_index)))
            for sentence_index, raw_sentence in enumerate(SENTENCE_RE.findall(str(passage.get("text", "")))):
                sentence = raw_sentence.strip()
                if not sentence:
                    continue
                score = len(query_tokens & tokenize(sentence))
                if score > 0:
                    scored.append((score, passage_index, sentence_index, sentence, doc_id))

        if not scored:
            return "", []

        # Best score first; ties broken by passage then sentence order for a
        # stable, deterministic selection.
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = scored[: cls.MAX_SENTENCES]

        answer = _TEMPLATE_PREFIX + " ".join(item[3] for item in selected)
        # Cite every passage a selected sentence came from, de-duplicated and in
        # first-appearance order.
        citations = list(dict.fromkeys(item[4] for item in selected))
        return answer, citations
