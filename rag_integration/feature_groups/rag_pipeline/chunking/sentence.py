"""Sentence-based text chunking."""

from __future__ import annotations

import re
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker


class SentenceChunker(BaseChunker):
    """
    Sentence-based text chunker.

    Splits text at sentence boundaries, respecting natural language structure.
    Combines sentences until chunk_size is reached.

    Better readability than fixed-size chunking for narrative content.

    Config-based matching:
        chunking_method="sentence"
    """

    PROPERTY_MAPPING = {
        BaseChunker.CHUNKING_METHOD: {
            "sentence": "Sentence-boundary aware chunks",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseChunker.CHUNK_SIZE: {
            "explanation": "Maximum size of each chunk (in characters)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 512,
        },
        BaseChunker.CHUNK_OVERLAP: {
            "explanation": "Overlap between consecutive chunks",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 50,
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to chunk",
            DefaultOptionKeys.context: True,
        },
    }

    # Pattern to split on sentence boundaries
    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = cls.SENTENCE_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    @classmethod
    def _chunk_text(
        cls,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of sentences to overlap (reinterpreted)

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return [text] if text else [""]

        sentences = cls._split_sentences(text)
        if not sentences:
            return [text]

        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If single sentence exceeds chunk_size, include it anyway
            if not current_chunk and sentence_length > chunk_size:
                chunks.append(sentence)
                continue

            # Check if adding this sentence exceeds chunk_size
            new_length = current_length + sentence_length + (1 if current_chunk else 0)

            if new_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                # Overlap: keep last N sentences where N ~ chunk_overlap / avg_sentence_length
                overlap_sentences = max(1, chunk_overlap // 50)  # Rough estimate
                current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences < len(current_chunk) else []
                current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1 if current_chunk else 0

            current_chunk.append(sentence)
            current_length = sum(len(s) for s in current_chunk) + len(current_chunk) - 1

        # Add remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [""]
