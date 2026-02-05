"""File-based document source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.document_source.base import BaseDocumentSource


class FileDocumentSource(BaseDocumentSource):
    """
    File-based document source.

    Loads documents from JSON files. Each document should have
    'doc_id' and 'text' fields.

    Configuration:
        file_path: Path to JSON file containing documents
        text_field: Field name containing text (default: "text")
        id_field: Field name containing document ID (default: "doc_id")

    Expected JSON format:
        [
            {"doc_id": "1", "text": "Document content..."},
            {"doc_id": "2", "text": "Another document..."}
        ]
    """

    @classmethod
    def _load_documents(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load documents from a JSON file.

        Args:
            options: Options containing file_path and optional field mappings

        Returns:
            List of document dictionaries

        Raises:
            ValueError: If file_path not provided or file not found
        """
        file_path = options.get("file_path") if options else None
        text_field_opt = options.get("text_field") if options else None
        text_field = str(text_field_opt) if text_field_opt is not None else "text"
        id_field_opt = options.get("id_field") if options else None
        id_field = str(id_field_opt) if id_field_opt is not None else "doc_id"

        if not file_path:
            raise ValueError("file_path is required for FileDocumentSource")

        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)

        if not isinstance(raw_documents, list):
            raw_documents = [raw_documents]

        # Normalize field names
        documents = []
        for i, doc in enumerate(raw_documents):
            normalized = {
                "doc_id": doc.get(id_field, f"doc_{i}"),
                "text": doc.get(text_field, ""),
            }
            # Preserve other fields as metadata
            for key, value in doc.items():
                if key not in (id_field, text_field):
                    normalized[key] = value
            documents.append(normalized)

        return documents
