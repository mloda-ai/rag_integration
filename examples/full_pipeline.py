#!/usr/bin/env python3
"""
Example: Full RAG Pipeline

Demonstrates the complete docs__pii_redacted__chunked__deduped__embedded pipeline.
"""

from rag_integration.feature_groups.rag_pipeline import (
    DictDocumentSource,
    RegexPIIRedactor,
    FixedSizeChunker,
    ExactHashDeduplicator,
    MockEmbedder,
)

# Sample documents with various PII patterns for demonstration
SAMPLE_DOCUMENTS = [
    {
        "doc_id": "doc_001",
        "text": (
            "Hello, my name is John Smith and I work at Acme Corp. "
            "You can reach me at john.smith@acme.com or call me at 555-123-4567. "
            "My social security number is 123-45-6789."
        ),
        "metadata": {"source": "example", "category": "contact"},
    },
    {
        "doc_id": "doc_002",
        "text": (
            "Meeting notes from March 15th, 2024. "
            "Attendees: Jane Doe (jane.doe@example.org), Bob Wilson. "
            "Action items: Review Q1 report, schedule follow-up call at 800-555-0199."
        ),
        "metadata": {"source": "example", "category": "meeting"},
    },
    {
        "doc_id": "doc_003",
        "text": (
            "Customer support ticket #4521. "
            "Customer: Alice Johnson, Phone: (555) 987-6543. "
            "Issue: Unable to access account. Email: alice.j@customer.net. "
            "Resolution: Password reset sent to registered email."
        ),
        "metadata": {"source": "example", "category": "support"},
    },
]


def run_pipeline_manually() -> list[list[float]]:
    """Run the pipeline step by step to see intermediate results."""
    print("=" * 60)
    print("RAG Pipeline Example (Manual Execution)")
    print("=" * 60)

    # Step 1: Load documents
    print("\n--- Step 1: Load Documents ---")
    from mloda.user import Options

    docs = DictDocumentSource._load_documents(Options(context={"documents": SAMPLE_DOCUMENTS}))
    print(f"Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  [{doc['doc_id']}]: {doc['text'][:50]}...")

    # Step 2: PII Redaction
    print("\n--- Step 2: PII Redaction ---")
    texts = [doc["text"] for doc in docs]
    redacted = RegexPIIRedactor._redact_pii(texts, ["ALL"], "type_label")
    for i, (orig, red) in enumerate(zip(texts, redacted)):
        if orig != red:
            print(f"  Doc {i}: PII found and redacted")
            print(f"    Before: {orig[:60]}...")
            print(f"    After:  {red[:60]}...")

    # Step 3: Chunking
    print("\n--- Step 3: Chunking ---")
    all_chunks = []
    for i, text in enumerate(redacted):
        chunks = FixedSizeChunker._chunk_text(text, chunk_size=200, chunk_overlap=20)
        print(f"  Doc {i}: Split into {len(chunks)} chunks")
        for j, chunk in enumerate(chunks):
            all_chunks.append({"doc_idx": i, "chunk_idx": j, "text": chunk})

    print(f"  Total chunks: {len(all_chunks)}")

    # Step 4: Deduplication
    print("\n--- Step 4: Deduplication ---")
    chunk_texts: list[str] = [str(c["text"]) for c in all_chunks]
    duplicates = ExactHashDeduplicator._find_duplicates(chunk_texts, threshold=1.0)
    unique_chunks = [c for c, dup in zip(all_chunks, duplicates) if dup is None]
    dup_count = sum(1 for d in duplicates if d is not None)
    print(f"  Found {dup_count} duplicate chunks")
    print(f"  Unique chunks: {len(unique_chunks)}")

    # Step 5: Embedding
    print("\n--- Step 5: Embedding ---")
    unique_texts: list[str] = [str(c["text"]) for c in unique_chunks]
    embeddings = MockEmbedder._embed_texts(unique_texts, embedding_dim=384, model_name="mock")
    print(f"  Generated {len(embeddings)} embeddings")
    print(f"  Embedding dimension: {len(embeddings[0])}")

    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Summary")
    print("=" * 60)
    print(f"  Input documents:  {len(docs)}")
    print(f"  After chunking:   {len(all_chunks)} chunks")
    print(f"  After dedup:      {len(unique_chunks)} unique chunks")
    print(f"  Embeddings:       {len(embeddings)} vectors of dim {len(embeddings[0])}")

    return embeddings


def show_provider_alternatives() -> None:
    """Show the different provider alternatives available."""
    print("\n" + "=" * 60)
    print("Available Provider Alternatives")
    print("=" * 60)

    print("\n1. Document Sources:")
    print("   - FileDocumentSource: Load from JSON files")
    print("   - DictDocumentSource: Pass documents via Options")

    print("\n2. PII Redaction:")
    print("   - RegexPIIRedactor: Regex patterns (email, phone, SSN)")
    print("   - SimplePIIRedactor: Word-list based (common names)")
    print("   - PatternPIIRedactor: User-configurable patterns")

    print("\n3. Chunking:")
    print("   - FixedSizeChunker: Fixed character count")
    print("   - SentenceChunker: Split on sentence boundaries")
    print("   - ParagraphChunker: Split on paragraph boundaries")

    print("\n4. Deduplication:")
    print("   - ExactHashDeduplicator: MD5 hash (exact match)")
    print("   - NormalizedDeduplicator: Whitespace-normalized")
    print("   - NGramDeduplicator: N-gram Jaccard similarity")

    print("\n5. Embedding:")
    print("   - MockEmbedder: Deterministic random vectors")
    print("   - HashEmbedder: Feature hashing")
    print("   - TfidfEmbedder: TF-IDF vectors")


if __name__ == "__main__":
    # Run the manual pipeline example
    embeddings = run_pipeline_manually()

    # Show available alternatives
    show_provider_alternatives()

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
