"""Document source feature groups."""

from rag_integration.feature_groups.rag_pipeline.document_source.base import BaseDocumentSource
from rag_integration.feature_groups.rag_pipeline.document_source.file_loader import FileDocumentSource
from rag_integration.feature_groups.rag_pipeline.document_source.dict_loader import DictDocumentSource

__all__ = [
    "BaseDocumentSource",
    "FileDocumentSource",
    "DictDocumentSource",
]
