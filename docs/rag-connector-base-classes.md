# RAG connector base classes

The `connectors/` package wraps whole external open-source RAG tools under one
mloda surface, organized into families by query-contract shape. Each family is a
thin `Base<Family>Connector` FeatureGroup plus one or more concrete backends
gated by a `<family>_backend` selector, with an inheritable contract-test suite.
A user swaps retrievers, rerankers, or generators by changing options, not by
rewriting a pipeline.

This sits alongside the build-your-own stage pipeline
(`feature_groups/rag_pipeline/`): the stages let a user assemble a pipeline step
by step, the connectors let a user drop in one external tool that subsumes
several steps.

## How families are cut

Families are cut by **query-contract shape**, not by paradigm (lexical / dense /
hybrid / late-interaction) and not by vendor (LlamaIndex / Haystack / ...).

> Same in/out shape -> a **backend** inside an existing family. Different in/out
> shape -> its **own family**.

BM25, dense bi-encoder, hybrid/RRF, and ColBERT late-interaction all share one
contract, `query + top_k -> ranked passages with scores`, so they are four
backends of a single `retrieve` family, not four families. The contract
boundaries are where the in/out signature actually changes: rerank takes
candidates in, generate returns prose plus citations, structured returns typed
rows, graph_rag traverses a subgraph, orchestrator is opaque end-to-end.

## The landscape survey

The open-source RAG systems the families were clustered from, grouped by the
contract cluster they fall into. Per row: the query contract (in -> out), the
state it needs, the no-Docker answer, the family it maps to, and a pedigree tag
(`real-lib-inmem` / `real-lib-server` / `fixture-stub` / `research-prototype`).

Rows count tool *surfaces*, not unique repositories: a few projects appear in
more than one family because they expose more than one contract (LangChain and
LlamaIndex span orchestrator + generate + structured; Haystack spans
orchestrator + generate; ColBERT spans retrieve + rerank). A handful of entries
are hosted or not strictly OSS (Cohere-rerank, Canopy) and are labeled as such.

No-Docker legend: **in-mem** = runs in-process after at most a pip install
(possibly a model download, noted); **fixture** = exercisable only via a static
fixture / REST stub; **server** = genuinely needs a running server or Docker.

### Orchestration frameworks and RAG applications (-> `orchestrator`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| LlamaIndex | `index.as_query_engine().query(str)` -> answer + `source_nodes` | in-mem `VectorStoreIndex`; LLM + embed model | in-mem (basic) | orchestrator | real-lib-inmem |
| Haystack 2.x | `Pipeline.run({"query"})` -> answers + retrieved documents | DocumentStore (InMemory available) | in-mem (basic) | orchestrator | real-lib-inmem |
| txtai | `embeddings.search(q)` / `rag(q)` -> results / answer | in-mem embeddings index (SQLite + ANN) | in-mem | orchestrator | real-lib-inmem |
| LangChain (RAG chains) | `RetrievalQA.invoke({"query"})` -> result + source_documents | in-mem vectorstore (FAISS/Chroma); LLM | in-mem (basic) | orchestrator | real-lib-inmem |
| DSPy | `module(question=...)` -> `Prediction.answer` | retriever + LM clients | in-mem (basic) | orchestrator | real-lib-inmem |
| Embedchain (mem0) | `app.add(src)`; `app.query(q)` -> answer | in-mem Chroma default; LLM API | in-mem (basic) | orchestrator | real-lib-inmem |
| LLMWare | `Query(library).query(text)` -> results; `Prompt` -> answer | library store (SQLite/Mongo); small model | in-mem (basic) | orchestrator | real-lib-inmem |
| LocalGPT | `run_localGPT.py` query -> answer + sources | local Chroma from ingest; local model | in-mem (basic) | orchestrator | real-lib-inmem |
| FlashRAG | `pipeline.run(dataset)` -> answers + eval | corpus + prebuilt index; model downloads | in-mem (basic) | orchestrator | research-prototype |
| AutoRAG | evaluator over QA data -> best pipeline; `query` -> answer | corpus + QA dataset + YAML; model APIs | in-mem (basic) | orchestrator | research-prototype |
| FLARE | `query` -> answer via iterative look-ahead retrieval | retriever + LM | in-mem (basic) | orchestrator | research-prototype |
| Self-RAG | `query` -> answer with retrieval/critique reflection tokens | fine-tuned Llama checkpoint; retriever | in-mem (basic) | generate / orchestrator | research-prototype |
| Canopy (Pinecone) | `ChatEngine.chat` / `query` -> answer + context | Pinecone index (external); OpenAI key | server | orchestrator | real-lib-server |
| Cognita (TrueFoundry) | REST `/query` -> answer + sources | backend server, vector DB, metadata store | server | orchestrator | real-lib-server |
| R2R (SciPhi) | REST `/rag` query -> answer + citations | Postgres + pgvector; Docker compose | server | orchestrator | real-lib-server |
| RAGFlow (InfiniFlow) | REST/UI query -> answer + grounded chunks | Docker stack (deep parse, ES, MySQL, MinIO) | server | orchestrator | real-lib-server |
| Verba (Weaviate) | UI/API query -> answer + context windows | Weaviate instance | server | orchestrator | real-lib-server |
| Quivr | API query -> answer over uploaded docs | Supabase/Postgres backend | server | orchestrator | real-lib-server |
| Danswer / Onyx | API query -> answer + cited sources | Docker stack (Postgres, Vespa, connectors) | server | orchestrator | real-lib-server |
| Khoj | API/chat query -> answer over personal corpus | Django server, embeddings DB | server | orchestrator | real-lib-server |
| PrivateGPT | API/UI query -> answer + source chunks | local LLM + embed; Qdrant/Chroma; FastAPI | server (local) | orchestrator | real-lib-server |
| AnythingLLM | API query (workspace) -> answer + citations | Node server, LanceDB default, LLM provider | server | orchestrator | real-lib-server |
| Open WebUI (RAG) | query + uploaded docs -> answer + refs | Open WebUI server + Ollama/OpenAI | server | orchestrator | real-lib-server |
| Dify | API app query -> answer + retrieved refs | Docker stack (Postgres, Redis, vector DB) | server | orchestrator | real-lib-server |
| Flowise | deployed flow API query -> answer | Node server; vector store; LLM keys | server | orchestrator | real-lib-server |
| kotaemon (Cinnamon) | UI/API query -> answer + citations (GraphRAG opt) | local/server install; vector + doc store | server | orchestrator | real-lib-server |

### Retrieval engines, vector stores, lexical / dense / late-interaction (-> `retrieve`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| FAISS | `query_emb + index -> top_k ids + distances` | prebuilt in-mem index | in-mem | retrieve | real-lib-inmem |
| Chroma | `query_text|emb + collection -> top_k docs + distances` | collection; optional embed fn | in-mem (embedded) / server | retrieve | real-lib-inmem |
| Qdrant | `query_vector(+filter) + collection -> scored points` | collection of vectors + payloads | in-mem (`:memory:`) / server | retrieve | real-lib-inmem |
| Milvus | `query_vectors + collection -> top_k ids + distances` | collection; index built | in-mem (Milvus-Lite) / server | retrieve | real-lib-inmem / server |
| LanceDB | `query_vector + table -> top_k rows + distances` | on-disk Lance table | in-mem (embedded) | retrieve | real-lib-inmem |
| Weaviate | `near_vector|near_text + class -> objects + scores` | schema + vectors; running node | server (embedded opt) | retrieve | real-lib-server |
| pgvector | `query_vector + table -> rows ORDER BY distance` | Postgres table + ANN index | server (Postgres) | retrieve | real-lib-server |
| Vespa | `YQL (ANN+BM25+rank) -> ranked hits` | deployed app package; running node | server | retrieve | real-lib-server |
| Elasticsearch (kNN + BM25) | `knn vector OR BM25 text + index -> ranked hits + _score` | index/mappings; running cluster | server | retrieve | real-lib-server |
| OpenSearch | `knn/neural OR BM25 + index -> ranked hits + _score` | index; running cluster (k-NN plugin) | server | retrieve | real-lib-server |
| Marqo | `query_text|image + index -> ranked docs + scores` | running server + embed models | server | retrieve | real-lib-server |
| Vald | `query_vector via gRPC -> nearest ids + distances` | K8s cluster (agent-NGT, gateway) | server (K8s) | retrieve | real-lib-server |
| Annoy | `query_vector + index -> top_k ids + distances` | prebuilt immutable index | in-mem | retrieve | real-lib-inmem |
| hnswlib | `query_vector + index -> top_k labels + distances` | in-mem HNSW graph | in-mem | retrieve | real-lib-inmem |
| ScaNN | `query_vector + searcher -> top_k ids + scores` | built partition/quantized index | in-mem | retrieve | real-lib-inmem |
| nmslib | `query_vector + index -> top_k ids + distances` | built in-mem index | in-mem | retrieve | real-lib-inmem |
| rank_bm25 | `tokenized_query + corpus -> per-doc score array` | in-mem tokenized corpus | in-mem | retrieve | real-lib-inmem |
| bm25s | `query_tokens + sparse index -> top_k ids + scores` | eagerly-scored sparse matrix (scipy) | in-mem | retrieve | real-lib-inmem |
| Pyserini (Anserini/Lucene) | `query_text + Lucene index -> ranked docids + scores` | prebuilt Lucene index; JVM (no server) | in-mem (needs Java) | retrieve | real-lib-inmem |
| Tantivy / tantivy-py | `query + index -> top docs + BM25 scores` | on-disk inverted index (embedded) | in-mem (embedded) | retrieve | real-lib-inmem |
| Whoosh | `query + index -> ranked hits + scores` | on-disk inverted index | in-mem | retrieve | real-lib-inmem |
| ColBERT | `query_tok_emb + token index -> MaxSim passages` | ColBERT checkpoint + PLAID index | in-mem (GPU pref.) | retrieve | real-lib-inmem |
| RAGatouille | `query_text + indexed corpus -> ranked passages` | ColBERT model + built index | in-mem | retrieve | real-lib-inmem |
| PLAID | `query_emb + compressed centroid index -> top_k` | quantized ColBERT index | in-mem | retrieve | research-prototype |
| SPLADE | `query_text -> sparse term weights -> ranked passages` | SPLADE model + sparse/inverted index | in-mem (index may be ES) | retrieve | research-prototype |
| DPR | `question_emb + FAISS passage index -> top_k passages` | Q/ctx encoders + FAISS index | in-mem | retrieve | research-prototype |
| sentence-transformers (bi-encoder) | `query_emb + corpus_emb -> top_k (semantic_search)` | downloaded model; corpus embeddings | in-mem (model dl) | retrieve | real-lib-inmem |
| Instructor embeddings | `(instruction, text) -> embedding` (feed to ANN) | downloaded INSTRUCTOR model | in-mem (model dl) | retrieve | real-lib-inmem |
| BGE / FlagEmbedding (retrieval) | `text -> embedding` (dense+sparse+colbert) | downloaded BGE model | in-mem (model dl) | retrieve | real-lib-inmem |
| ELSER (ES learned sparse) | `query_text -> expanded sparse tokens -> ranked hits` | ES cluster + deployed ELSER model | server | retrieve | real-lib-server |

### Rerankers (-> `rerank`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| FlashRank | `query + candidates -> reordered passages + scores` | ONNX cross-encoder download | in-mem (model dl) | rerank | real-lib-inmem |
| sentence-transformers CrossEncoder | `query + candidates -> reordered + relevance scores` | cross-encoder download | in-mem (model dl) | rerank | real-lib-inmem |
| BGE-reranker (FlagEmbedding) | `query + candidates -> reordered + scores` | BGE reranker download | in-mem (model dl) | rerank | real-lib-inmem |
| MixedBread mxbai-rerank | `query + candidates -> reordered + scores` | mxbai-rerank download | in-mem (model dl) | rerank | real-lib-inmem |
| monoT5 / castorini | `query + passage -> relevance score -> reordered` | T5 reranker download (pygaggle) | in-mem (model dl) | rerank | real-lib-inmem |
| ColBERT-as-reranker | `query + candidates -> MaxSim scores -> reordered` | ColBERT checkpoint download | in-mem (model dl) | rerank | real-lib-inmem |
| rerankers (AnswerDotAI) | `query + candidates -> reordered` (unified API) | backend model / API key | in-mem (model dl) | rerank | real-lib-inmem |
| RankGPT | `query + candidates -> LLM permutation -> reordered` | LLM API key | server (LLM API) | rerank | research-prototype |
| Cohere-rerank | `query + candidates -> reordered + scores` | Cohere API key (not OSS) | server (hosted) | rerank | real-lib-server |
| Lexical token-overlap reranker | `query + candidates -> reordered by overlap` | none | in-mem | rerank | fixture-stub |

### Answer generators (-> `generate`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| Template / extractive responder | `query + passages -> templated/extractive answer` | none | in-mem | generate | fixture-stub |
| HuggingFace QA pipeline (extractive) | `question + context -> answer span + score` | QA model download | in-mem (model dl) | generate | real-lib-inmem |
| Haystack readers | `query + docs -> answer span / generated + citations` | reader/LLM download or API key | in-mem (model dl) | generate | real-lib-inmem |
| LangChain generation | `query + passages -> LLM answer + citations` | LLM API key or local model | in-mem (model dl) | generate | real-lib-inmem |
| llama.cpp / Ollama | `query + passages prompt -> generated answer` | GGUF download / Ollama daemon | in-mem (model dl) | generate | real-lib-inmem |
| FiD (fusion-in-decoder) | `query + N passages -> fused generated answer` | trained FiD checkpoint | in-mem (model dl) | generate | research-prototype |

### Graph-RAG (-> `graph_rag`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| GraphRAG via networkx | `query -> in-mem graph traversal -> passages` | graph build (in-process) | in-mem | graph_rag | fixture-stub |
| Microsoft GraphRAG | `query -> community graph traversal -> answer` | graph build (parquet artifacts); LLM | in-mem (post-index) | graph_rag | real-lib-inmem |
| nano-graphrag | `query -> graph traversal -> context -> answer` | graph build; LLM API key | in-mem (file artifacts) | graph_rag | real-lib-inmem |
| LlamaIndex KnowledgeGraph / PropertyGraphIndex | `query -> KG traversal -> passages -> answer` | graph build; LLM API key | in-mem | graph_rag | real-lib-inmem |
| LightRAG | `query -> dual-level graph + vector -> answer` | graph build; embed/LLM API key | in-mem (file artifacts) | graph_rag | real-lib-inmem |
| HippoRAG | `query -> personalized PageRank over KG -> passages` | graph build; model download; LLM | in-mem (model dl) | graph_rag | research-prototype |
| Neo4j GraphRAG | `query -> Cypher/vector graph retrieval -> passages` | running Neo4j DB; LLM API key | server | graph_rag | real-lib-server |

### Text-to-SQL / structured retrieval (-> `structured`)

| System | Query contract (in -> out) | State needed | No-Docker? | Family | Pedigree |
|---|---|---|---|---|---|
| Rule-based text-to-SQL (in-mem SQLite) | `question + table -> SQL -> typed rows` | in-mem SQLite copy of the table | in-mem | structured | fixture-stub |
| LlamaIndex NLSQLTableQueryEngine | `NL question + schema -> SQL -> rows -> answer` | SQL DB (SQLite ok); LLM API key | in-mem (SQLite) | structured | real-lib-inmem |
| LangChain SQLDatabaseChain | `NL question + schema -> SQL -> rows -> answer` | SQL DB (SQLite ok); LLM API key | in-mem (SQLite) | structured | real-lib-inmem |
| Vanna.AI | `NL question + trained schema -> SQL -> rows` | vector store of schema; DB; LLM API key | in-mem (embeddable) | structured | real-lib-inmem |
| sqlcoder (defog) | `NL question + schema prompt -> SQL` | sqlcoder model download | in-mem (model dl) | structured | real-lib-inmem |
| DAIL-SQL / DIN-SQL | `NL question + schema (few-shot) -> SQL` | LLM API key (GPT-4); benchmark data | server (LLM API) | structured | research-prototype |
| PICARD | `NL question + schema -> constrained decode -> SQL` | T5 model + PICARD parsing server | server | structured | research-prototype |

### Evaluation harnesses (cross-cutting, not a connector family)

These do not fit a retrieval family: they consume `(query, answer, contexts,
ground_truth)` and emit metric scores. They belong on top of the existing
`evaluation/` module, not as a connector family. Recorded for completeness.

| System | Query contract (in -> out) | No-Docker? | Disposition |
|---|---|---|---|
| RAGAS | `(query, answer, contexts, ground_truth) -> faithfulness/relevance scores` | in-mem (LLM API) | out-of-scope (eval) |
| TruLens | `(query, answer, contexts) + feedback fns -> scores (logged)` | in-mem (sqlite) | out-of-scope (eval) |
| DeepEval | `(query, answer, contexts, ground_truth) -> scores (pytest-style)` | in-mem (LLM API) | out-of-scope (eval) |
| ARES | `(query, answer, contexts) -> trained-judge scores` | in-mem (model dl) | out-of-scope (eval) |
| Phoenix (Arize) | `(query, answer, contexts, gt) -> scores + traces` | in-mem (local app) | out-of-scope (eval) |
| Giskard RAG (RAGET) | `(query, answer, contexts, gt) -> component scores + tests` | in-mem (LLM API) | out-of-scope (eval) |
| continuous-eval (relari) | `(query, answer, contexts, gt) -> modular metric scores` | in-mem (LLM API) | out-of-scope (eval) |

## The family map

Six families. Each has at least one no-Docker concrete. Contracts below are the
contracts declared on the family base classes.

| Family | Reader contract (in -> out) | No-Docker concrete | Other backends | Pedigree of the anchor |
|---|---|---|---|---|
| `retrieve` | `query_text + corpus + top_k -> ranked passages w/ scores` (`retrieved_passages: [{doc_id, text, score, rank}]`) | `Bm25sRetriever` (`bm25s`, zero-download lexical) | `TfidfRetriever` (vector-space lexical); no dense/FAISS backend yet | real-lib-inmem |
| `rerank` | `query_text + candidates + top_k -> reordered passages w/ scores` (`reranked_passages`) | `LexicalReranker` (pure-Python token overlap, zero-download) | `FlashRankReranker` (ONNX cross-encoder, `rerank` extra, CI-skip on model download) | fixture-stub anchor + real-lib |
| `generate` | `query_text + passages -> answer + citations` (`generated_answer: {answer, citations}`), grounded by construction | `ExtractiveResponder` (stdlib sentence extraction) | `TemplateResponder` (multi-citation template) | fixture-stub anchor |
| `graph_rag` | `query_text + nodes + edges + top_k -> ranked passages` (`graph_passages`); query-overlap + one-hop neighbour bonus | `AdjacencyGraphRag` (stdlib adjacency map, zero-download) | `NetworkxGraphRag` (`networkx`, `graph` extra); parity test pins identical ranking | fixture-stub anchor + real-lib |
| `structured` | `question + table -> SQL -> typed rows` (`structured_rows: {sql, rows}`); in-mem SQLite, single-SELECT sqlglot guard | `RuleBasedSql` (deterministic NL->SQL over in-mem SQLite) | `AggregateSql` (aggregation queries) | fixture-stub anchor |
| `orchestrator` | `query_text + corpus + top_k -> answer + documents` (internals opaque) (`orchestrated_answer: {answer, documents}`) | `HaystackOrchestrator` (Haystack 2.x BM25 pipeline, offline, telemetry off) | `R2RFixtureOrchestrator` (file-fixture REST stub with `SUPPORTED_VALUES` + stripped params) | real-lib-inmem + fixture-stub |

What each family is for:

- **`retrieve`** holds the vector-store / lexical / late-interaction backends
  (FAISS, Chroma, bm25s, ColBERT, ...): all share `query + top_k -> ranked
  passages`. Paradigm and vendor are backend and pedigree distinctions.
- **`rerank`** takes *candidates* in, not a corpus (FlashRank, cross-encoders,
  RankGPT).
- **`generate`** returns prose plus citations, a different out-shape from a
  ranked list (extractive QA, Haystack readers, local LLMs).
- **`graph_rag`** traverses a node/edge graph; the value is connected context
  (GraphRAG, LightRAG, HippoRAG, networkx prototypes).
- **`structured`** returns typed rows via generated SQL (Vanna,
  NLSQLTableQueryEngine).
- **`orchestrator`** is the opaque end-to-end surface for whole frameworks and
  apps (LlamaIndex, Haystack, txtai, and the server-shaped R2R/RAGFlow/Verba/...
  reached through a fixture stub).

## Cross-cutting properties

These recur across families and are currently implemented inline in each
family's `base.py`. The shared axis:

- **TopK / score-threshold** (retrieve, rerank, graph_rag, orchestrator)
- **Metadata filter** (corpus subset selection)
- **Corpus / index handle** (the locator: which prebuilt index or fixture)
- **Embedding-model selection** (retrieve dense backend, graph_rag)
- **Citation / provenance** (generate, orchestrator)

## Relationship to the stage pipeline

The stage pipeline (`feature_groups/rag_pipeline/`) has a FAISS-backed
`retrieval` stage and an `llm_response` stage, which cover the same ground as the
`retrieve` and `generate` connectors:

- A connector and the corresponding stage emit the same passage / answer row
  shape, so a downstream feature is agnostic to which produced it.
- Stages assemble a pipeline step by step; a connector drops in one external tool
  that subsumes embed + index + retrieve.
- Switching between them is a change of connector id / options, same
  `Feature -> run_all` shape, no pipeline rewrite.

The `retrieve` family currently has lexical backends (`bm25s`, `tfidf`) only;
there is no dense / FAISS backend yet, and the FAISS retrieval stage is not yet
wired in as one.

## Package layout

```
rag_integration/feature_groups/connectors/
  <family>/
    base.py            Base<Family>Connector (contract, option keys, validation)
    <backend>.py       concrete backend (declares its <family>_backend selector)
tests/connectors/
  <family>/
    <family>_contract.py   inheritable contract-test suite
    test_<backend>.py      concrete adapter test
```
