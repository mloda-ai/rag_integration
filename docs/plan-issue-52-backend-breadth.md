# Plan: backend breadth for the existing connector families (issue #52)

Status: planning document, no code yet. Implementation is gated on issue #51
(evaluation harness), which is intentionally ordered first so every backend
below lands with a metrics comparison row instead of only a smoke test.

Scope guard, restated from the issue: **no new contracts, no new families**.
Pure breadth behind the existing honest-surface discipline. Every candidate
below slots into an existing `Base<Family>Connector` as one more concrete
backend gated by the family's selector option, inherits the family's contract
test suite, and adds its own locked narrowing tests.

## Ordering and dependencies

1. #51 evaluation harness (open at the time of writing): provides the demo
   that turns each new backend into a comparison row.
2. This plan (#52): backend breadth plus cross-family RRF blending.
3. #53 agentic RAG: after this.

If a backend PR from this plan is ready before #51 lands, it merges with its
contract and narrowing tests only, and its eval row is backfilled in a small
follow-up once the harness exists. The reverse order is preferred.

## Ground rules carried over from the existing concretes

- **No-Docker CI policy.** Real in-memory libraries run in CI. Backends that
  download a model at first use live behind an extra and are CI-skipped the
  way `FlashRankReranker` is: `pytest.importorskip("<package>")` for a clean
  skip without the extra, plus a `requires_<backend>_model` marker in
  `tests/conftest.py` for the model-download skip (reusing an existing marker
  when one already fits). Server-shaped tools get
  file-fixture stubs like `R2RFixtureOrchestrator`, never a live daemon.
- **Honest surface.** A concrete advertises only what it honors. Narrowing
  lives in behaviour and is locked by a test, never in a `SUPPORTED_VALUES`
  attribute. Each backend below lists its expected narrowing row up front.
- **Selector gating.** Selection gates on the family's `<family>_backend`
  context option in `match_feature_group_criteria`; disjoint selector values,
  unknown value matches nothing (`test_unknown_backend_does_not_match` per
  family). New backends follow the repo's established context-option pattern.
- **Contract suites.** Each backend's test subclasses the family's inheritable
  suite in `tests/connectors/<family>/<family>_contract.py` and implements the
  handful of adapter methods (`connector_class`, `backend_value`, sample data).
- **Docs.** Every backend adds its row to the family map in
  `rag_integration/feature_groups/connectors/README.md`, the family table and
  honest-surface table in `docs/rag-connector-base-classes.md`. The landscape
  survey tables carry pedigree tags, not a built/unbuilt column, so build
  state is expressed by presence in those concrete tables, not by editing the
  survey rows.

## Candidate backends

### 1. `rerank`: sentence-transformers cross-encoder (do first)

The smallest step: it mirrors the existing FlashRank pattern one to one.

- **Class / selector:** `CrossEncoderReranker`, `rerank_backend="cross_encoder"`.
- **Library:** `sentence-transformers` `CrossEncoder` with a fixed small model
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Chosen over BGE / FlagEmbedding
  because sentence-transformers is already a known dependency in the
  `advanced` extra; a BGE reranker remains a survey row.
- **Extra:** new `rerank-st` extra (`sentence-transformers>=2.2.0`). Not folded
  into the existing `rerank` extra: that would force torch onto FlashRank
  users, and FlashRank exists precisely as the no-torch cross-encoder.
- **CI:** model downloads on first use, so `pytest.importorskip` plus the
  existing `requires_sentence_transformer_model` marker in `tests/conftest.py`
  (it already covers Hugging Face Hub downloads by sentence-transformers), the
  FlashRank arrangement with no new marker needed. `LexicalReranker` stays the
  family's always-on CI anchor.
- **Narrowing row:** model fixed, no model option exposed; locked by a
  `test_cross_encoder_surface.py` mirroring `test_flashrank_surface.py`.

### 2. `orchestrator`: LlamaIndex query engine

- **Class / selector:** `LlamaIndexOrchestrator`, `orchestrator_backend="llamaindex"`.
- **Library:** `llama-index-core` only (no integration packages). The survey
  row catalogues the `index.as_query_engine().query(str)` shape; the offline
  cut below wraps the LLM-free retriever path instead, and the query-engine
  synthesizer stays a survey entry.
- **Offline strategy (the main design decision):** LlamaIndex defaults pull an
  OpenAI LLM and an embedding model, both off-limits in CI. Recommended cut:
  build a `SimpleKeywordTableIndex` (regex keyword extraction, no LLM, no
  embeddings), query through its retriever, and synthesize the answer as the
  top source node's text. That is the same visible narrowing the
  `HaystackOrchestrator` anchor already locks (answer drawn from the top
  document, no LLM), so the family stays consistent. Alternative considered:
  `MockLLM` / `MockEmbedding`; rejected because mock output is prompt echo,
  which is a worse honest surface than top-document text.
- **Extra:** new `llamaindex` extra. Zero model download under the recommended
  cut, so its contract test runs in CI (pedigree `real-lib-inmem`).
- **Narrowing row:** answer is the top retrieved node's text (no LLM
  synthesis); empty query / empty corpus / non-positive `top_k` yield an empty
  result, matching the Haystack lock.

### 3. `orchestrator`: txtai

- **Class / selector:** `TxtaiOrchestrator`, `orchestrator_backend="txtai"`.
- **Library:** `txtai` `Embeddings` in keyword mode (BM25 / sparse scoring,
  `keyword=True`), which indexes and searches with no model download. The
  survey row's `embeddings.search(q)` shape; the `rag(q)` LLM path stays a
  survey row.
- **Extra:** new `txtai` extra. Risk: txtai's default install pulls torch and
  transformers, which is heavy for CI install time even though nothing is
  downloaded at runtime. If CI install cost proves prohibitive, fall back to
  local-only tests with a CI skip and note it in the pedigree row; decide in
  the PR against measured install time.
- **Narrowing row:** keyword scoring only (no dense / hybrid / LLM path);
  answer is the top document's text, same lock shape as Haystack.

### 4. `generate`: first LLM-backed generator

The generate contract is `query_text + passages -> answer + citations`,
grounded by construction: the base rejects any citation that is not a supplied
passage `doc_id` (see `_validate_citations`). An LLM backend therefore never
trusts model-claimed citations: the backend cites the passages it actually
placed in the prompt, and the free-form part is only the answer text.

- **Primary class / selector:** `LlamaCppResponder`,
  `generate_backend="llama_cpp"`.
- **Library:** `llama-cpp-python` with a small fixed instruct model in GGUF
  form (candidate: Qwen2.5-0.5B-Instruct-GGUF; final pick in the PR by
  download size and local runtime). Chosen over Ollama because Ollama is a
  server daemon, which under the no-Docker / no-server policy would have to be
  a fixture stub rather than a real backend.
- **Extra:** new `generate-llm` extra. Note: `llama-cpp-python` does not fetch
  anything at `Llama(model_path=...)` construction; the backend must acquire
  the GGUF via `Llama.from_pretrained(repo_id, filename)`, which downloads
  from the Hugging Face Hub and caches locally. That gives the FlashRank
  local-cache story: the contract test runs locally against the cache and is
  CI-skipped (`importorskip` + a new `requires_llama_cpp_model` marker);
  `ExtractiveResponder` remains the family's CI anchor.
- **Determinism:** LLM output is not byte-stable even at temperature 0 across
  library versions. The locked tests assert structural properties only
  (grounding, citation validity, non-empty answer, prompt-passage provenance),
  not exact text. The family contract suite already leans structural, so this
  fits without contract changes.
- **Secondary, cheap row:** a new adapter class `ClaudeCliResponder`
  (`generate_backend="claude_cli"`) that reuses the subprocess CLI-call logic
  from the stage-side `ClaudeCliResponse`
  (`feature_groups/rag_pipeline/llm_response/claude_cli.py`). It is an
  adapter, not a fold-in of the existing FeatureGroup: the stage contract is
  `query + context -> answer string` with no citations, so the connector
  subclasses `BaseGenerateConnector` and constructs citations itself from the
  passages it placed in the prompt. Extends the documented stage-to-connector
  migration seam; skipped whenever the `claude` CLI is absent (CI included).
  Optional; separate small PR.
- **Narrowing rows:** model fixed, no sampling options exposed; citations are
  the prompted passages, not model claims; `claude_cli` additionally requires
  the external CLI and is non-deterministic (locked structurally).

### 5. `retrieve`: late interaction (optional, decide last)

The issue marks this optional if weight cost is prohibitive. Candidates:

- **RAGatouille / ColBERT:** truest late-interaction pedigree, but pulls torch
  plus a several-hundred-MB checkpoint; strictly local-only tests.
- **SPLADE via transformers:** sparse learned weights, research-prototype
  pedigree row per the survey, similar torch weight.

Recommendation: defer the build decision until after backend 1 lands, since
the cross-encoder already establishes the torch-extra weight tolerance in this
repo. If tolerated, build `ColbertRetriever` behind a `colbert` extra with
local-only tests and a `research-prototype` / `real-lib-inmem` pedigree row.
If not, record the decision in the design doc and leave them as survey rows
only. Either outcome satisfies the issue's definition of done.

## Cross-family RRF blending (decided in #46, deferred until now)

Wire `fusion.py`'s `rrf_fuse` to blend `retrieve` and `graph_rag` rankings on
`doc_id`, exactly as the rank-fusion section of the design doc records: reuse
the mechanics, do not grow a new backend or a seventh family, because the
blended in/out shape is no single family's contract.

- **Mechanics:** a small helper next to `rrf_fuse` in `connectors/fusion.py`,
  `blend_ranked_passages(rankings, top_k, k=DEFAULT_RRF_K)`, taking sequences
  of ranked passage dicts (each `{doc_id, text, score, rank}`), fusing on
  `doc_id` keys via `rrf_fuse`, and returning the fused best-first passage
  list with RRF scores and re-assigned ranks. Passage text resolves from the
  first ranking that contains the `doc_id`. Existing `rrf_fuse` guarantees
  carry over for free: only-positive scores, deterministic ties, duplicate
  keys raise.
- **Demo hook:** extend `cli/swap_demo.py` with a blending step that runs the
  `retrieve` anchor and a `graph_rag` backend over the shared demo corpus and
  fuses the two rankings; pinned by `tests/connectors/test_swap_demo.py`.
- **Tests:** `tests/connectors/test_fusion_blend.py` covering consensus beats
  single placement, determinism, top_k truncation, text resolution, duplicate
  `doc_id` inside one ranking raising, and empty-rankings behaviour.
- **Docs:** update the rank-fusion section of the design doc from "not built
  yet" to built, with the helper named.

## Extras summary (proposed additions to `pyproject.toml`)

| Extra | Contents | CI behaviour |
|---|---|---|
| `rerank-st` | `sentence-transformers>=2.2.0` | installed, tests CI-skipped on model download |
| `llamaindex` | `llama-index-core>=0.10` | installed, tests run in CI (no download) |
| `txtai` | `txtai>=7.0` | installed, tests run in CI if install weight acceptable, else local-only |
| `generate-llm` | `llama-cpp-python>=0.2` | installed, tests CI-skipped on model download |
| `colbert` (only if built) | `ragatouille` | local-only tests |

Cross-family blending needs no new extra: `rrf_fuse` is stdlib-typed, and the
demo runs on the zero-download anchors.

## PR slicing and order

One backend per PR, each self-contained (code + contract-suite subclass +
narrowing lock test + README and design-doc rows + eval-harness row once #51
is in):

1. `rerank` cross-encoder (smallest, pattern-mirroring).
2. `orchestrator` LlamaIndex.
3. `orchestrator` txtai.
4. `generate` llama.cpp.
5. Cross-family RRF blending + demo hook + design-doc update.
6. Optional: `generate` claude_cli fold-in.
7. Optional: `retrieve` late interaction, pending the weight decision.

## Definition of done (mapped to the issue)

- At least one new real-library backend in each of `orchestrator` (items 2, 3),
  `rerank` (item 1), and `generate` (item 4): covered.
- `retrieve` late interaction optional: explicit decision point recorded
  (item 7).
- Cross-family RRF blending via the existing `rrf_fuse`, with tests and a demo
  hook: covered (item 5).
- Family-map README and design-doc tables updated with the new pedigree rows:
  part of every backend PR.
- Each backend produces a comparison row in the #51 eval harness demo:
  guaranteed by the ordering rule at the top of this plan.

## Open questions

1. txtai install weight in CI: measure in the PR; fall back to local-only
   tests if prohibitive.
2. LlamaIndex offline synthesis: recommended keyword-table + top-node answer;
   revisit only if the resulting surface feels too narrow next to Haystack.
3. GGUF model choice for `llama_cpp` (size vs quality at 0.5B to 1.5B scale).
4. Extras naming (`rerank-st`, `generate-llm`) vs single-word style of the
   existing extras; settle in the first PR that adds one.
