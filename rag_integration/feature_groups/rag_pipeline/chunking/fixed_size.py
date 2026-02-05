"""Fixed-size text chunking."""

from __future__ import annotations

from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """
    Fixed-size text chunker.

    Splits text into chunks of fixed character count with configurable overlap.
    Simple and fast, works well for uniform content.

    Configuration:
        chunk_size: Maximum characters per chunk (default: 512)
        chunk_overlap: Characters to overlap between chunks (default: 50)

    Config-based matching:
        chunking_method="fixed_size"
    """

    PROPERTY_MAPPING = {
        BaseChunker.CHUNKING_METHOD: {
            "fixed_size": "Fixed character count chunks",
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

    @classmethod
    def _chunk_text(
        cls,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Split text into fixed-size chunks.

        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return [text] if text else [""]

        # Ensure valid parameters
        chunk_size = max(1, chunk_size)
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
        step = chunk_size - chunk_overlap

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]

            # Try to break at word boundary if not at end
            if end < text_len and chunk_size > 10:
                # Look for last space in last 20% of chunk
                search_start = max(0, len(chunk) - chunk_size // 5)
                last_space = chunk.rfind(" ", search_start)
                if last_space > search_start:
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())

            # Move to next chunk
            if end >= text_len:
                break
            start = start + step
            if start >= text_len:
                break

        # Filter empty chunks
        chunks = [c for c in chunks if c]

        return chunks if chunks else [""]
