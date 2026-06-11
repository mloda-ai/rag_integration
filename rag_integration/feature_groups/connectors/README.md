# Connector families

The `connectors/` package wraps whole external open-source RAG tools under one
mloda surface, organized into families by **query-contract shape**. Each family
is a thin `Base<Family>Connector` FeatureGroup plus one or more concrete
backends gated by a per-family selector option (`retrieve_backend`,
`rerank_backend`, `generate_backend`, `graph_backend`, `structured_backend`,
`orchestrator_backend`), with an inheritable contract-test suite so a new
backend's test is a handful of adapter methods.

This sits alongside the build-your-own stage pipeline (`../rag_pipeline/`): the
stages let you assemble a pipeline step by step, the connectors let you drop in
one external tool that subsumes several steps. You swap retrievers, rerankers,
or generators by changing options, not by rewriting a pipeline.

For the design (how families are cut, the full landscape survey, and the base
classes) see [`docs/rag-connector-base-classes.md`](../../../docs/rag-connector-base-classes.md).
The shared cross-cutting mixins and error types live in [`mixins.py`](mixins.py)
and [`errors.py`](errors.py).

## Family map

The canonical concrete per family is the zero-download, deterministic backend
that anchors the CI contract suite. Pedigree tags: `real-lib-inmem` (a real
library running in-process), `fixture-stub` (deterministic stand-in, no model
download or server). The full survey in the design doc also uses
`real-lib-server` and `research-prototype`.

| Family | Reader contract (in -> out) | No-Docker concrete | Other backends | Pedigree of the anchor | Contract suite |
|---|---|---|---|---|---|
| [`retrieve`](retrieve/) | `query_text + corpus + top_k -> ranked passages` (`retrieved_passages: [{doc_id, text, score, rank}]`) | `Bm25sRetriever` (`bm25s`, zero-download lexical) | `TfidfRetriever` (vector-space lexical), `FaissDenseRetriever` (dense FAISS, `faiss` extra), `HybridRrfRetriever` (RRF-fused lexical + dense, `faiss` extra) | real-lib-inmem | [`retrieve_contract.py`](../../../tests/connectors/retrieve/retrieve_contract.py) |
| [`rerank`](rerank/) | `query_text + candidates + top_k -> reordered passages` (`reranked_passages`) | `LexicalReranker` (token overlap, zero-download) | `FlashRankReranker` (ONNX cross-encoder, `rerank` extra, CI-skip on model download) | fixture-stub | [`rerank_contract.py`](../../../tests/connectors/rerank/rerank_contract.py) |
| [`generate`](generate/) | `query_text + passages -> answer + citations` (`generated_answer: {answer, citations}`), grounded by construction | `ExtractiveResponder` (stdlib sentence extraction) | `TemplateResponder` (multi-citation template) | fixture-stub | [`generate_contract.py`](../../../tests/connectors/generate/generate_contract.py) |
| [`graph_rag`](graph_rag/) | `query_text + (nodes + edges, or a `graph_source` feature) + top_k -> ranked passages` (`graph_passages`); query overlap + one-hop neighbour bonus | `AdjacencyGraphRag` (stdlib adjacency map, zero-download) | `NetworkxGraphRag` (`networkx`, `graph` extra); parity test pins identical ranking; `TriplesKnowledgeGraph` KG source feeds either backend | fixture-stub | [`graph_rag_contract.py`](../../../tests/connectors/graph_rag/graph_rag_contract.py) |
| [`structured`](structured/) | `question + table -> SQL -> typed rows` (`structured_rows: {sql, rows}`); in-mem SQLite, single-SELECT `sqlglot` guard | `RuleBasedSql` (rule-based NL->SQL, `structured` extra) | `AggregateSql` (adds avg/min/max/sum intents) | fixture-stub | [`structured_contract.py`](../../../tests/connectors/structured/structured_contract.py) |
| [`orchestrator`](orchestrator/) | `query_text + corpus + top_k -> answer + documents` (internals opaque) (`orchestrated_answer: {answer, documents}`) | `HaystackOrchestrator` (Haystack 2.x BM25, offline, `orchestrator` extra) | `R2RFixtureOrchestrator` (file-fixture REST stub) | real-lib-inmem | [`orchestrator_contract.py`](../../../tests/connectors/orchestrator/orchestrator_contract.py) |

## Families in detail

### `retrieve` -- `query_text + corpus + top_k -> ranked passages`

Holds the vector-store / lexical / late-interaction backends (FAISS, Chroma,
bm25s, ColBERT, ...). The anchor `Bm25sRetriever` (`retrieve_backend="bm25s"`) is
BM25 lexical retrieval via `bm25s`: zero-download, deterministic, numpy/scipy.
`TfidfRetriever` (`retrieve_backend="tfidf"`) ranks the same corpus by TF-IDF
cosine similarity using the repo's deterministic embedder, also zero-download.
`FaissDenseRetriever` (`retrieve_backend="faiss"`, `--extra faiss`) is the
canonical **dense** backend: the same FAISS nearest-neighbor search the stage
pipeline's `retrieval` stage runs, folded in behind this contract (cosine over
the repo's deterministic hash embeddings, in-memory `IndexFlatIP`).
`HybridRrfRetriever` (`retrieve_backend="hybrid_rrf"`, `--extra faiss`) fuses
the lexical and dense rankings with reciprocal-rank fusion; the fusion
mechanics live in the cross-cutting [`fusion.py`](fusion.py), so future
blending of rankings across families (e.g. `retrieve` + `graph_rag`, by
`doc_id`) reuses `rrf_fuse` instead of growing a new backend.

The FAISS `retrieval` stage and this family are one world: the stage serves the
same `retrieved_passages` shape from a pre-built on-disk index, so migrating
between stage and connector is an option swap, not a pipeline rewrite. See
"Relationship to the stage pipeline" in the
[design doc](../../../docs/rag-connector-base-classes.md) and the parity test
in [`tests/integration/test_stage_connector_parity.py`](../../../tests/integration/test_stage_connector_parity.py).

```python
from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from rag_integration.feature_groups.connectors.retrieve import Bm25sRetriever

feature = Feature(
    "retrieved_passages",
    options=Options(context={
        "retrieve_backend": "bm25s",
        "query_text": "cat pet",
        "corpus": [
            {"doc_id": "d1", "text": "A cat is an independent and curious pet."},
            {"doc_id": "d2", "text": "Cars need regular engine oil and maintenance."},
        ],
        "top_k": 3,
    }),
)
results = mlodaAPI.run_all(
    [feature],
    compute_frameworks={PythonDictFramework},
    plugin_collector=PluginCollector.enabled_feature_groups({Bm25sRetriever}),
)
```

### `rerank` -- `query_text + candidates + top_k -> reordered passages`

Takes *candidates* in (already retrieved), not a corpus. `LexicalReranker`
(`rerank_backend="lexical"`) is pure-Python token overlap, zero-download.
`FlashRankReranker` (`rerank_backend="flashrank"`, `--extra rerank`) adds a real
ONNX cross-encoder; its model downloads on first use, so its test runs locally
and is skipped on CI.

### `generate` -- `query_text + passages -> answer + citations`

Returns prose plus citations, grounded by construction (every citation is one of
the supplied passages). `ExtractiveResponder` (`generate_backend="extractive"`)
does pure-Python sentence extraction with a single citation. `TemplateResponder`
(`generate_backend="template"`) selects top query-relevant sentences across
passages into a fixed template and cites every passage it drew from. LLM-backed
generators are pedigree backends for later.

### `graph_rag` -- `query_text + nodes + edges + top_k -> ranked passages`

Scores nodes by query overlap plus a one-hop neighbour bonus: a passage
connected to a relevant one is surfaced even with no query-term overlap.
`AdjacencyGraphRag` (`graph_backend="adjacency"`) applies the scoring over a
hand-built adjacency map with stdlib only. `NetworkxGraphRag`
(`graph_backend="networkx"`, `--extra graph`) does the same over `networkx`; a
parity test pins identical ranking, showing the contract is not tied to one
graph library.

The graph arrives inline (`nodes` + `edges`) or from an upstream
knowledge-graph source: setting `graph_source` to the source's feature name
(e.g. `"knowledge_graph"`) makes the connector declare that feature as its
input and consume an existing graph source instead of duplicating one.
`TriplesKnowledgeGraph` (`kg_backend="triples"`, [`kg_source.py`](graph_rag/kg_source.py))
is the first source: it builds the `{nodes, edges}` payload from
subject-predicate-object triples, pure Python.

### `structured` -- `question + table -> SQL -> typed rows`

Answers a natural-language question over a relational table. `RuleBasedSql`
(`structured_backend="rule_based"`, `--extra structured`) does rule-based NL->SQL
executed on stdlib `sqlite3`, with `sqlglot` validating the generated SQL is a
single top-level `SELECT`. Values are always bound parameters and identifiers
whitelisted. `AggregateSql` (`structured_backend="aggregate"`) adds aggregation
intents (avg/min/max/sum) on top of the count/filter/list intents.

### `orchestrator` -- `query_text + corpus + top_k -> answer + documents`

Wraps a whole external RAG framework as one connector (bring your existing
pipeline); the internals are the framework's. `HaystackOrchestrator`
(`orchestrator_backend="haystack"`, `--extra orchestrator`) runs a real Haystack
2.x in-memory BM25 pipeline, zero-download (no model, no server) so it runs in
CI. `R2RFixtureOrchestrator` (`orchestrator_backend="r2r"`) models a
server-shaped tool over a static JSON fixture, surfacing only canned documents
that are in the supplied corpus. Other server-shaped tools can follow the same
fixture-stub pattern.

## How a backend is selected

Each base gates on its selector option in `match_feature_group_criteria` (named
per family above; note `graph_rag` uses `graph_backend`, not
`graph_rag_backend`); backends declare disjoint selector values, so at most one
ever claims a given `Options`. An unknown backend matches nothing. The
base owns the cross-backend contract (option extraction, validation, assembly);
a concrete backend implements only its one ranking / generation hook.

## Install

```bash
uv sync --extra connectors   # or --extra rerank / graph / structured / orchestrator
uv sync --extra faiss        # dense retrieve backend (FaissDenseRetriever)
```
