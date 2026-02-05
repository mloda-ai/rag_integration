"""RAG Pipeline feature groups with provider inheritance pattern."""

from rag_integration.feature_groups.rag_pipeline.document_source import (
    BaseDocumentSource,
    FileDocumentSource,
    DictDocumentSource,
)
from rag_integration.feature_groups.rag_pipeline.pii_redaction import (
    BasePIIRedactor,
    RegexPIIRedactor,
    SimplePIIRedactor,
    PatternPIIRedactor,
    PresidioPIIRedactor,
)
from rag_integration.feature_groups.rag_pipeline.chunking import (
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    SemanticChunker,
)
from rag_integration.feature_groups.rag_pipeline.deduplication import (
    BaseDeduplicator,
    ExactHashDeduplicator,
    NormalizedDeduplicator,
    NGramDeduplicator,
)
from rag_integration.feature_groups.rag_pipeline.embedding import (
    BaseEmbedder,
    MockEmbedder,
    HashEmbedder,
    TfidfEmbedder,
    SentenceTransformerEmbedder,
)

__all__ = [
    # Document Source
    "BaseDocumentSource",
    "FileDocumentSource",
    "DictDocumentSource",
    # PII Redaction
    "BasePIIRedactor",
    "RegexPIIRedactor",
    "SimplePIIRedactor",
    "PatternPIIRedactor",
    "PresidioPIIRedactor",
    # Chunking
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "SemanticChunker",
    # Deduplication
    "BaseDeduplicator",
    "ExactHashDeduplicator",
    "NormalizedDeduplicator",
    "NGramDeduplicator",
    # Embedding
    "BaseEmbedder",
    "MockEmbedder",
    "HashEmbedder",
    "TfidfEmbedder",
    "SentenceTransformerEmbedder",
]
