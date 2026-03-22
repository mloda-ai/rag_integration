"""Tests for DictDocumentSource."""

from typing import Type

import pytest
from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.document_source import DictDocumentSource
from rag_integration.feature_groups.rag_pipeline.document_source.base import BaseDocumentSource
from tests.feature_groups.document_source.document_source_test_base import DocumentSourceTestBase


class TestDictDocumentSource(DocumentSourceTestBase):
    """Tests for DictDocumentSource."""

    @property
    def source_class(self) -> Type[BaseDocumentSource]:
        return DictDocumentSource

    def test_load_from_dict_list(self) -> None:
        """Should load documents from list of dicts."""
        documents = [
            {"doc_id": "1", "text": "First document"},
            {"doc_id": "2", "text": "Second document"},
        ]
        docs = DictDocumentSource._load_documents(Options(context={"documents": documents}))
        assert len(docs) == 2
        assert docs[0]["doc_id"] == "1"
        assert docs[1]["text"] == "Second document"

    def test_load_plain_strings(self) -> None:
        """Should handle list of plain strings."""
        documents = ["Plain text one", "Plain text two"]
        docs = DictDocumentSource._load_documents(Options(context={"documents": documents}))
        assert len(docs) == 2
        assert docs[0]["text"] == "Plain text one"
        assert docs[0]["doc_id"] == "doc_0"
        assert docs[1]["text"] == "Plain text two"
        assert docs[1]["doc_id"] == "doc_1"

    def test_custom_field_mapping(self) -> None:
        """Should use custom field names for text and id."""
        documents = [{"id": "abc", "content": "Custom fields"}]
        docs = DictDocumentSource._load_documents(
            Options(context={"documents": documents, "text_field": "content", "id_field": "id"})
        )
        assert docs[0]["doc_id"] == "abc"
        assert docs[0]["text"] == "Custom fields"

    def test_missing_documents_error(self) -> None:
        """Should raise ValueError when documents not provided."""
        with pytest.raises(ValueError, match="documents list is required"):
            DictDocumentSource._load_documents(Options())

    def test_preserve_metadata(self) -> None:
        """Should preserve extra fields as metadata."""
        documents = [{"doc_id": "1", "text": "Hello", "author": "Bob", "category": "test"}]
        docs = DictDocumentSource._load_documents(Options(context={"documents": documents}))
        assert docs[0]["author"] == "Bob"
        assert docs[0]["category"] == "test"
