# RAG Pipeline CLI

A command-line interface for running the RAG (Retrieval-Augmented Generation) pipeline.

## Usage

```bash
python3 -m cli.rag_demo --help
```

## Commands

### List available components

```bash
python3 -m cli.rag_demo list
```

### Run the pipeline

```bash
python3 -m cli.rag_demo run [OPTIONS]
```

## Examples

### Full pipeline with example docs folder

The `cli/docs/` folder contains example documents with PII data. Run the full pipeline:

```bash
python3 -m cli.rag_demo run --input cli/docs/ --pii regex --chunking sentence --embedding tfidf --dedup normalized -v
```

### Basic run with inline JSON document

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"Hello world, this is a test document."}]'
```

### Run with PII redaction

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"Contact john@example.com or call 555-123-4567"}]' --pii regex -v
```

### Run with sentence chunking and tfidf embeddings

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"First sentence. Second sentence. Third sentence."}]' --chunking sentence --embedding tfidf -v
```

### Run with custom chunk size and overlap

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"A longer document that will be split into smaller chunks for processing."}]' --chunk-size 100 --chunk-overlap 20 -v
```

### Run with hash embeddings and normalized deduplication

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"Test one"},{"doc_id":"2","text":"Test two"}]' --embedding hash --dedup normalized -v
```

### Run with multiple documents and save output to file

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"Document one"},{"doc_id":"2","text":"Document two"},{"doc_id":"3","text":"Document one"}]' --output results.json -v
```

### Full pipeline with all options

```bash
python3 -m cli.rag_demo run --docs '[{"doc_id":"1","text":"Email me at test@example.com for details."}]' --pii regex --chunking paragraph --embedding hash --dedup ngram --chunk-size 256 --embedding-dim 128 -v
```

### Load from a file

```bash
python3 -m cli.rag_demo run --input ./path/to/document.txt -v
```

### Load from a directory of files

```bash
python3 -m cli.rag_demo run --input ./path/to/docs/ --chunking sentence -v
```

## Options

### Input Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input file or directory |
| `--docs` | Inline JSON documents |

### Method Options

| Option | Choices | Default |
|--------|---------|---------|
| `--chunking` | `fixed_size`, `sentence`, `paragraph`, `semantic` | `fixed_size` |
| `--embedding` | `mock`, `hash`, `tfidf`, `sentence_transformer` | `mock` |
| `--dedup` | `exact_hash`, `normalized`, `ngram` | `exact_hash` |
| `--pii` | `regex`, `simple`, `pattern`, `presidio` | None (disabled) |

### Parameter Options

| Option | Default | Description |
|--------|---------|-------------|
| `--chunk-size` | 512 | Chunk size in characters |
| `--chunk-overlap` | 50 | Overlap between chunks |
| `--embedding-dim` | 384 | Embedding dimension |

### Output Options

| Option | Description |
|--------|-------------|
| `--output`, `-o` | Output file (JSON) |
| `--verbose`, `-v` | Verbose output |
