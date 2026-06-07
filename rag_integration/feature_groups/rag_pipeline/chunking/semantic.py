"""Semantic chunking using sentence embeddings to find natural boundaries."""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, List, Optional, Tuple

from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker

if TYPE_CHECKING:
    from mloda.user import Feature


class SemanticChunker(BaseChunker):
    """
    Semantic chunker using sentence embeddings to find natural boundaries.

    Splits text into sentences, then groups semantically similar sentences
    together. Detects topic shifts by measuring embedding similarity between
    consecutive sentences.

    Requires: pip install sentence-transformers

    Configuration:
        chunk_size: Maximum characters per chunk (soft limit, respects sentence boundaries)
        chunk_overlap: Not used for semantic chunking (boundaries are semantic)
        similarity_threshold: Cosine similarity threshold for grouping (default: 0.5)
        model_name: Sentence transformer model (default: "all-MiniLM-L6-v2")

    Config-based matching:
        chunking_method="semantic"

    Note: Caches the model at class level for performance. Initialization is
    guarded by a lock so concurrent callers do not build the model twice.
    """

    # Additional configuration keys
    SIMILARITY_THRESHOLD = "similarity_threshold"
    MODEL_NAME = "model_name"

    # Defaults for the additional options (kept in sync with PROPERTY_MAPPING).
    DEFAULT_SIMILARITY_THRESHOLD = 0.5
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    PROPERTY_MAPPING = {
        BaseChunker.CHUNKING_METHOD: {
            "semantic": "Semantic boundary aware chunks using embeddings",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseChunker.CHUNK_SIZE: {
            "explanation": "Maximum size of each chunk (in characters, soft limit)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 512,
        },
        SIMILARITY_THRESHOLD: {
            "explanation": "Cosine similarity threshold for grouping sentences (0.0-1.0)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: DEFAULT_SIMILARITY_THRESHOLD,
        },
        MODEL_NAME: {
            "explanation": "Sentence transformer model name",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: DEFAULT_MODEL,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to chunk",
            DefaultOptionKeys.context: True,
        },
    }

    # Cached as a single (model_name, model) tuple so the lock-free fast path
    # reads it atomically (one attribute load) instead of two fields that could
    # be observed mid-update.
    _model_cache: Optional[Tuple[str, object]] = None
    _model_lock = threading.Lock()

    # Pattern to split on sentence boundaries
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    @classmethod
    def _get_similarity_threshold(cls, feature: "Feature") -> float:
        """Get the similarity threshold from feature options."""
        value = feature.options.get(cls.SIMILARITY_THRESHOLD)
        return float(value) if value is not None else cls.DEFAULT_SIMILARITY_THRESHOLD

    @classmethod
    def _get_model_name(cls, feature: "Feature") -> str:
        """Get the model name from feature options."""
        value = feature.options.get(cls.MODEL_NAME)
        return str(value) if value is not None else cls.DEFAULT_MODEL

    @classmethod
    def _chunk_text_for_feature(cls, text: str, feature: "Feature") -> List[str]:
        """Chunk text using the per-feature similarity threshold and model."""
        return cls._chunk_text_semantic(
            text,
            cls._get_chunk_size(feature),
            cls._get_similarity_threshold(feature),
            cls._get_model_name(feature),
        )

    @classmethod
    def _get_model(cls, model_name: str) -> object:
        """Get or create the sentence transformer model (thread-safe)."""
        # Fast path: single atomic read of the (name, model) cache.
        cache = cls._model_cache
        if cache is not None and cache[0] == model_name:
            return cache[1]

        with cls._model_lock:
            # Re-check inside the lock: another thread may have built it.
            cache = cls._model_cache
            if cache is not None and cache[0] == model_name:
                return cache[1]

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SemanticChunker. "
                    "Install with: pip install sentence-transformers"
                ) from e

            model = SentenceTransformer(model_name)
            cls._model_cache = (model_name, model)
            return model

    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = cls.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    @classmethod
    def _cosine_similarity(cls, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    @classmethod
    def _chunk_text(
        cls,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Uses default similarity threshold and model. For custom settings,
        use _chunk_text_semantic directly.

        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk (soft limit)
            chunk_overlap: Not used for semantic chunking

        Returns:
            List of text chunks
        """
        return cls._chunk_text_semantic(
            text,
            chunk_size,
            similarity_threshold=cls.DEFAULT_SIMILARITY_THRESHOLD,
            model_name=cls.DEFAULT_MODEL,
        )

    @classmethod
    def _chunk_text_semantic(
        cls,
        text: str,
        chunk_size: int,
        similarity_threshold: float,
        model_name: str,
    ) -> List[str]:
        """
        Split text into semantically coherent chunks.

        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk (soft limit)
            similarity_threshold: Threshold for grouping sentences
            model_name: Sentence transformer model name

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return [text] if text else [""]

        sentences = cls._split_sentences(text)
        if not sentences:
            return [text]

        if len(sentences) == 1:
            return sentences

        # Get sentence embeddings
        model = cls._get_model(model_name)
        embeddings = model.encode(sentences)  # type: ignore[attr-defined]

        # Group sentences by semantic similarity
        chunks: List[str] = []
        current_chunk: List[str] = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            prev_embedding = embeddings[i - 1].tolist()
            curr_embedding = embeddings[i].tolist()

            similarity = cls._cosine_similarity(prev_embedding, curr_embedding)

            # Check if we should start a new chunk
            would_exceed_size = current_length + len(sentence) + 1 > chunk_size
            topic_shift = similarity < similarity_threshold

            if (topic_shift and current_chunk) or (would_exceed_size and current_chunk):
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [""]
