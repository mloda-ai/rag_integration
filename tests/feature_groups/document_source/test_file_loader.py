"""Tests for FileDocumentSource."""

from pathlib import Path
from typing import Type

import pytest
from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.document_source import FileDocumentSource
from rag_integration.feature_groups.rag_pipeline.document_source.base import BaseDocumentSource
from tests.feature_groups.document_source.document_source_test_base import DocumentSourceTestBase


class TestFileDocumentSource(DocumentSourceTestBase):
    """Tests for FileDocumentSource."""

    @property
    def source_class(self) -> Type[BaseDocumentSource]:
        return FileDocumentSource

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        """Should load documents from JSON file."""
        json_file = tmp_path / "docs.json"
        json_file.write_text('[{"doc_id": "1", "text": "Hello world"}]')

        docs = FileDocumentSource._load_documents(Options(context={"file_path": str(json_file)}))
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "1"
        assert docs[0]["text"] == "Hello world"

    def test_custom_field_mapping(self, tmp_path: Path) -> None:
        """Should use custom field names for text and id."""
        json_file = tmp_path / "docs.json"
        json_file.write_text('[{"id": "abc", "content": "Custom fields"}]')

        docs = FileDocumentSource._load_documents(
            Options(context={"file_path": str(json_file), "text_field": "content", "id_field": "id"})
        )
        assert docs[0]["doc_id"] == "abc"
        assert docs[0]["text"] == "Custom fields"

    def test_missing_file_path_error(self) -> None:
        """Should raise ValueError when file_path not provided."""
        with pytest.raises(ValueError, match="file_path is required"):
            FileDocumentSource._load_documents(Options())

    def test_file_not_found_error(self) -> None:
        """Should raise ValueError when file doesn't exist."""
        with pytest.raises(ValueError, match="File not found"):
            FileDocumentSource._load_documents(Options(context={"file_path": "/nonexistent/file.json"}))

    def test_single_document_wrapping(self, tmp_path: Path) -> None:
        """Should wrap single document dict in list."""
        json_file = tmp_path / "docs.json"
        json_file.write_text('{"doc_id": "single", "text": "Single document"}')

        docs = FileDocumentSource._load_documents(Options(context={"file_path": str(json_file)}))
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "single"

    def test_preserve_metadata(self, tmp_path: Path) -> None:
        """Should preserve extra fields as metadata."""
        json_file = tmp_path / "docs.json"
        json_file.write_text('[{"doc_id": "1", "text": "Hello", "author": "Alice", "category": "test"}]')

        docs = FileDocumentSource._load_documents(Options(context={"file_path": str(json_file)}))
        assert docs[0]["author"] == "Alice"
        assert docs[0]["category"] == "test"
