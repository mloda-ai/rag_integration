"""Base test class for document source feature groups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

from mloda.user import Options

from rag_integration.feature_groups.rag_pipeline.document_source.base import BaseDocumentSource


class DocumentSourceTestBase(ABC):
    """Abstract base providing shared tests for all document source implementations."""

    @property
    @abstractmethod
    def source_class(self) -> Type[BaseDocumentSource]: ...

    def test_feature_matching_pattern(self) -> None:
        """Should match document source features."""
        assert self.source_class.match_feature_group_criteria("docs", Options())
