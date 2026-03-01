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
from rag_integration.feature_groups.rag_pipeline.vector_store import (
    BaseVectorStore,
    FaissFlatIndexer,
    FaissIVFIndexer,
    FaissHNSWIndexer,
    VectorStoreArtifact,
)
from rag_integration.feature_groups.rag_pipeline.retrieval import (
    BaseRetriever,
    FaissRetriever,
)
from rag_integration.feature_groups.rag_pipeline.llm_response import (
    BaseLLMResponse,
    ClaudeCliResponse,
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
