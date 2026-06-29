"""The swap-backends demo is the runnable proof for issue #34, item 1.

Every swap below goes through the demo's single :func:`run_connector` helper,
which enables the whole fixed :data:`CONNECTORS` set and varies only the options
dict. So the test pins the actual claim: swapping a backend is a change to the
options dict, not to the calling code or the enabled providers. Because all
backends are enabled at once, the asserts also exercise selector routing: the
distinct per-backend scores prove the ``<family>_backend`` value, not the call
site, chose the backend, and an unknown value claims nothing.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from cli.swap_demo import CONNECTORS, CORPUS, QUERY, SHARED_INPUTS, TOP_K, run_connector

PASSAGES: List[Dict[str, str]] = [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in CORPUS]


def _assert_passage_shape(passages: List[Dict[str, Any]]) -> None:
    assert passages, "expected a non-empty ranking for the shape assertions to mean anything"
    for rank, passage in enumerate(passages):
        assert set(passage) == {"doc_id", "text", "score", "rank"}
        assert passage["rank"] == rank
    scores = [p["score"] for p in passages]
    assert scores == sorted(scores, reverse=True)


class TestWithinFamilyRetrieveSwap:
    """retrieve_backend="bm25s" -> "tfidf": same call, same enabled set, only the selector value moves."""

    def test_only_the_selector_value_differs_between_the_two_runs(self) -> None:
        pytest.importorskip("bm25s")  # bm25s is behind the connectors extra
        bm25s_passages = run_connector("retrieved_passages", {"retrieve_backend": "bm25s", **SHARED_INPUTS})
        tfidf_passages = run_connector("retrieved_passages", {"retrieve_backend": "tfidf", **SHARED_INPUTS})

        _assert_passage_shape(bm25s_passages)
        _assert_passage_shape(tfidf_passages)
        # Two different lexical mechanisms, but the same relevant documents in the
        # same order: the swap is safe, the downstream contract is unchanged.
        assert [p["doc_id"] for p in bm25s_passages] == ["d2", "d1"]
        assert [p["doc_id"] for p in tfidf_passages] == ["d2", "d1"]
        # Routing proof: with both retrievers enabled, the selector (not the call
        # site) picks the backend, so each run carries its own backend's score.
        # bm25s (BM25) and tfidf (cosine) score the top doc differently.
        assert bm25s_passages[0]["score"] != pytest.approx(tfidf_passages[0]["score"])


class TestWithinFamilyGenerateSwap:
    """generate_backend="extractive" -> "template": same call, only the selector value moves."""

    def test_both_backends_answer_under_one_contract(self) -> None:
        extractive = run_connector(
            "generated_answer", {"generate_backend": "extractive", "query_text": QUERY, "passages": PASSAGES}
        )
        template = run_connector(
            "generated_answer", {"generate_backend": "template", "query_text": QUERY, "passages": PASSAGES}
        )

        known = {p["doc_id"] for p in PASSAGES}
        for answer in (extractive, template):
            assert set(answer) == {"answer", "citations"}
            assert answer["answer"], "a no-LLM responder still answers when the corpus is on-topic"
            # Grounded by construction: every citation is one of the supplied passages.
            assert answer["citations"], "a non-empty answer must cite its source"
            assert all(c in known for c in answer["citations"])
        # Routing proof: the two backends produce genuinely different answers, so
        # the selector (not the call site) chose between them.
        assert extractive["answer"] != template["answer"]


class TestAcrossFamilySwap:
    """retrieve <-> orchestrator: identical inputs and enabled set, only the selector key and root name change."""

    def test_the_two_families_share_their_inputs_verbatim(self) -> None:
        retrieve_options = {"retrieve_backend": "tfidf", **SHARED_INPUTS}
        orchestrator_options = {"orchestrator_backend": "haystack", **SHARED_INPUTS}

        # The only differences are the selector key and (downstream) the root
        # feature name; query/corpus/top_k are the same objects on both sides.
        shared_keys = {"query_text", "corpus", "top_k"}
        for key in shared_keys:
            assert retrieve_options[key] == orchestrator_options[key]
        assert set(retrieve_options) - shared_keys == {"retrieve_backend"}
        assert set(orchestrator_options) - shared_keys == {"orchestrator_backend"}

    def test_each_family_returns_its_own_shape_from_the_same_inputs(self) -> None:
        pytest.importorskip("haystack")  # the orchestrator backend is behind the orchestrator extra
        passages = run_connector("retrieved_passages", {"retrieve_backend": "tfidf", **SHARED_INPUTS})
        answer = run_connector("orchestrated_answer", {"orchestrator_backend": "haystack", **SHARED_INPUTS})

        _assert_passage_shape(passages)
        assert set(answer) == {"answer", "documents"}
        assert answer["answer"], "the orchestrator pipeline answers an on-topic query"
        corpus_ids = {doc["doc_id"] for doc in CORPUS}
        assert all(doc["doc_id"] in corpus_ids for doc in answer["documents"])


class TestSelectorGating:
    """The gating that makes "enable everything, vary only options" safe."""

    def test_unknown_backend_value_claims_nothing(self) -> None:
        # An unknown selector matches no enabled connector, so even with every
        # retriever enabled mloda resolves no feature group for the request. This
        # is the honest surface: a connector never claims a backend it cannot
        # serve, which is exactly what makes the all-enabled swap unambiguous.
        with pytest.raises(ValueError, match="No feature groups found"):
            run_connector("retrieved_passages", {"retrieve_backend": "does-not-exist", **SHARED_INPUTS})

    def test_every_enabled_connector_is_a_feature_group(self) -> None:
        # The fixed enabled set is the invariant the swap promise rests on: it
        # never changes between runs, only the options do.
        from mloda.provider import FeatureGroup

        assert CONNECTORS
        assert all(issubclass(c, FeatureGroup) for c in CONNECTORS)


def test_top_k_constant_is_within_corpus_size() -> None:
    """Guard the demo's own fixture so the swaps above stay non-degenerate."""
    assert 0 < TOP_K <= len(CORPUS)
