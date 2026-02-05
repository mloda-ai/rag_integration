"""In-memory dictionary document source."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.document_source.base import BaseDocumentSource


class DictDocumentSource(BaseDocumentSource):
    """
    In-memory dictionary document source.

    Accepts documents directly via Options. Useful for programmatic
    input and API integrations.

    Configuration:
        documents: List of document dictionaries
        text_field: Field name containing text (default: "text")
        id_field: Field name containing document ID (default: "doc_id")

    Usage:
        Feature("docs", Options(context={
            "documents": [
                {"doc_id": "1", "text": "Document content..."},
                {"doc_id": "2", "text": "Another document..."}
            ]
        }))
    """

    @classmethod
    def _load_documents(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load documents from Options.

        Args:
            options: Options containing documents list

        Returns:
            List of document dictionaries

        Raises:
            ValueError: If documents not provided
        """
        documents = options.get("documents") if options else None
        text_field_opt = options.get("text_field") if options else None
        text_field = str(text_field_opt) if text_field_opt is not None else "text"
        id_field_opt = options.get("id_field") if options else None
        id_field = str(id_field_opt) if id_field_opt is not None else "doc_id"

        if not documents:
            raise ValueError("documents list is required for DictDocumentSource")

        if not isinstance(documents, list):
            documents = [documents]

        # Normalize field names
        result = []
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                # Handle plain strings
                normalized = {
                    "doc_id": f"doc_{i}",
                    "text": doc,
                }
            else:
                normalized = {
                    "doc_id": doc.get(id_field, f"doc_{i}"),
                    "text": doc.get(text_field, ""),
                }
                # Preserve other fields
                for key, value in doc.items():
                    if key not in (id_field, text_field):
                        normalized[key] = value
            result.append(normalized)

        return result
