"""RAG Pipeline feature groups with provider inheritance pattern."""

from rag_integration.feature_groups.rag_pipeline.chunking import (
    BaseChunker,
    FixedSizeChunker,
    ParagraphChunker,
    SemanticChunker,
    SentenceChunker,
)
from rag_integration.feature_groups.rag_pipeline.deduplication import (
    BaseDeduplicator,
    ExactHashDeduplicator,
    NGramDeduplicator,
    NormalizedDeduplicator,
)
from rag_integration.feature_groups.rag_pipeline.document_source import (
    BaseDocumentSource,
    DictDocumentSource,
    FileDocumentSource,
)
from rag_integration.feature_groups.rag_pipeline.embedding import (
    BaseEmbedder,
    HashEmbedder,
    MockEmbedder,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
)
from rag_integration.feature_groups.rag_pipeline.llm_response import (
    BaseLLMResponse,
    ClaudeCliResponse,
)
from rag_integration.feature_groups.rag_pipeline.pii_redaction import (
    BasePIIRedactor,
    PatternPIIRedactor,
    PresidioPIIRedactor,
    RegexPIIRedactor,
    SimplePIIRedactor,
)
from rag_integration.feature_groups.rag_pipeline.retrieval import (
    BaseRetriever,
    FaissRetriever,
)
from rag_integration.feature_groups.rag_pipeline.vector_store import (
    BaseVectorStore,
    FaissFlatIndexer,
    FaissHNSWIndexer,
    FaissIVFIndexer,
    VectorStoreArtifact,
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
    # Vector Store
    "BaseVectorStore",
    "FaissFlatIndexer",
    "FaissIVFIndexer",
    "FaissHNSWIndexer",
    "VectorStoreArtifact",
    # Retrieval
    "BaseRetriever",
    "FaissRetriever",
    # LLM Response
    "BaseLLMResponse",
    "ClaudeCliResponse",
]
