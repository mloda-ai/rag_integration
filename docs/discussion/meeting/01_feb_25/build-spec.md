# mloda-RAG Build Specification

This document outlines the components to build for the mloda-RAG composable pipeline system, derived from the project vision.

---

## P0: Core Foundation

### 1. Pipeline Orchestration Layer

- [ ] Declarative feature chain parser (`docs__ocr__pii_redacted__chunked__deduped__embedded`)
- [ ] Pipeline compiler that resolves feature dependencies
- [ ] Stateless transformation interface/base class
- [ ] Plugin registration and discovery system

### 2. Basic Chunking Strategies

- [ ] Fixed-size chunking (with overlap)
- [ ] Semantic chunking (sentence/paragraph boundaries)
- [ ] Document-aware chunking (respects headers, sections)

### 3. Embedding Integrations

- [ ] OpenAI embeddings adapter
- [ ] Sentence Transformers (local) adapter
- [ ] Common embedding interface for swappability

### 4. Storage Adapters

- [ ] pgvector integration
- [ ] DuckDB integration (local development)
- [ ] Common vector store interface

---

## P1: Extended Capabilities

### 5. Additional Embedding Providers

- [ ] Cohere embeddings adapter
- [ ] Custom model support (HuggingFace)

### 6. Additional Vector Stores

- [ ] Pinecone adapter
- [ ] Qdrant adapter
- [ ] Milvus adapter

### 7. Data Quality Transformations

- [ ] Deduplication (exact and near-duplicate detection)
- [ ] Language detection

### 8. Multi-Framework Support

- [ ] Polars backend (local/single-machine)
- [ ] DuckDB backend (local/analytical)
- [ ] Framework abstraction layer for backend swapping

---

## P2: Advanced Features

### 9. Data Governance

- [ ] PII detection and redaction
- [ ] PHI anonymization (HIPAA compliance)
- [ ] Access control metadata tagging
- [ ] Audit trail logging

### 10. Multimodal Pre-processing

- [ ] OCR integration (Tesseract, cloud OCR APIs)
- [ ] Table extraction from PDFs/images
- [ ] Image captioning integration

### 11. Advanced Chunking

- [ ] Hierarchical chunking
- [ ] Long-context chunking strategies

---

## P3: Domain-Specific & Scale

### 12. Domain-Specific Transformations

- [ ] **Legal:** Citation extraction, precedent linking
- [ ] **Medical:** Medical NER, terminology normalization
- [ ] **Financial:** Entity extraction, compliance tagging
- [ ] **Code:** AST parsing, dependency graphs, API extraction

### 13. Distributed Processing

- [ ] Spark backend integration
- [ ] Pipeline scaling from local to distributed without code changes

### 14. Advanced Retrieval

- [ ] GraphRAG support
- [ ] Hybrid search (vector + keyword)
- [ ] Re-ranking integrations

---

## Reference Pipelines to Validate

Once core components are built, validate with these end-to-end pipelines:

| Pipeline | Flow |
|----------|------|
| **Basic** | Documents → Chunking → Embedding → pgvector |
| **Healthcare** | PDFs → OCR → PHI anonymization → Medical NER → HIPAA-compliant vectorDB |
| **Legal** | Court docs → Citation extraction → Precedent linking → GraphRAG |
| **Enterprise** | Internal docs → PII redaction → Access control → Multi-language → Audit trail |
| **Code/GitHub** | Repos → Syntax parsing → Dependency graphs → API extraction → Code-aware chunking → Semantic search |

---

## Architecture Principles

1. **Every step = swappable mloda plugin** - No hardcoded dependencies between stages
2. **Declarative over imperative** - Pipelines defined as feature chains, not code
3. **Provider-agnostic** - Abstract interfaces for all external services
4. **Scale-transparent** - Same pipeline definition works local (Polars) or distributed (Spark)
5. **Composable** - Insert/remove/swap stages without rewriting pipelines
