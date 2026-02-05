"""Paragraph-based text chunking."""

from __future__ import annotations

import re
from typing import List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.chunking.base import BaseChunker


class ParagraphChunker(BaseChunker):
    """
    Paragraph-based text chunker.

    Splits text at paragraph boundaries (double newlines).
    Respects document structure for better semantic coherence.

    Ideal for structured documents with clear paragraph separation.

    Config-based matching:
        chunking_method="paragraph"
    """

    PROPERTY_MAPPING = {
        BaseChunker.CHUNKING_METHOD: {
            "paragraph": "Paragraph-boundary aware chunks",
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

    # Pattern to split on paragraph boundaries
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    @classmethod
    def _split_paragraphs(cls, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = cls.PARAGRAPH_PATTERN.split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    @classmethod
    def _chunk_text(
        cls,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Split text into chunks at paragraph boundaries.

        Args:
            text: Text to split
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap (approximated by paragraphs)

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return [text] if text else [""]

        paragraphs = cls._split_paragraphs(text)
        if not paragraphs:
            return [text]

        # If only one paragraph, use fixed-size fallback
        if len(paragraphs) == 1:
            if len(paragraphs[0]) <= chunk_size:
                return paragraphs
            # Fall back to simple splitting for long single paragraphs
            return cls._split_long_paragraph(paragraphs[0], chunk_size, chunk_overlap)

        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            # If single paragraph exceeds chunk_size, split it
            if not current_chunk and para_length > chunk_size:
                sub_chunks = cls._split_long_paragraph(para, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                continue

            # Check if adding this paragraph exceeds chunk_size
            separator_length = 2 if current_chunk else 0  # "\n\n"
            new_length = current_length + para_length + separator_length

            if new_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap (keep last paragraph)
                if chunk_overlap > 0 and current_chunk:
                    last_para = current_chunk[-1]
                    if len(last_para) <= chunk_overlap:
                        current_chunk = [last_para]
                        current_length = len(last_para)
                    else:
                        current_chunk = []
                        current_length = 0
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length = sum(len(p) for p in current_chunk) + 2 * (len(current_chunk) - 1)

        # Add remaining paragraphs
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks if chunks else [""]

    @classmethod
    def _split_long_paragraph(cls, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split a long paragraph into smaller chunks."""
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > chunk_size // 2:
                    chunk = chunk[:last_space]

            chunks.append(chunk.strip())
            start += step

        return [c for c in chunks if c]
