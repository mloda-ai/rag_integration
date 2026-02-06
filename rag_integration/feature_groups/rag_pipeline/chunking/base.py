"""Base class for text chunking feature groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Set, Type, Union

from mloda.provider import FeatureGroup, ComputeFramework, FeatureSet
from mloda.provider import FeatureChainParserMixin
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys


class BaseChunker(FeatureChainParserMixin, FeatureGroup):
    """
    Base class for text chunking feature groups.

    Splits text documents into smaller chunks for embedding and retrieval.

    Feature Naming Pattern:
        {in_feature}__chunked

    Examples:
        - docs__pii_redacted__chunked
        - text__chunked

    Note: Chunking transforms 1 document into N chunks. The output data
    structure will have more rows than the input, with chunk metadata added.

    ## Configuration-Based Creation

    Uses Options with proper group/context parameter separation:

    ```python
    feature = Feature(
        name="my_chunked",
        options=Options(
            context={
                "chunking_method": "fixed_size",
                DefaultOptionKeys.in_features: "docs",
            }
        )
    )
    ```
    """

    # Configuration keys
    CHUNK_SIZE = "chunk_size"
    CHUNK_OVERLAP = "chunk_overlap"

    # Discriminator key for config-based feature matching
    CHUNKING_METHOD = "chunking_method"

    # Supported chunking methods (implementations must define which they handle)
    CHUNKING_METHODS = {
        "fixed_size": "Fixed character count chunks",
        "sentence": "Sentence-boundary aware chunks",
        "paragraph": "Paragraph-boundary aware chunks",
    }

    PREFIX_PATTERN = r".*__chunked$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    PROPERTY_MAPPING = {
        CHUNKING_METHOD: {
            **CHUNKING_METHODS,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        CHUNK_SIZE: {
            "explanation": "Maximum size of each chunk (in characters)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 512,
        },
        CHUNK_OVERLAP: {
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
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PythonDictFramework}

    @classmethod
    def _get_source_feature_name(cls, feature: Feature) -> str:
        """Extract source feature name from the feature."""
        source_features = cls._extract_source_features(feature)
        return source_features[0]

    @classmethod
    def _get_chunk_size(cls, feature: Feature) -> int:
        """Get chunk size from feature options."""
        size = feature.options.get(cls.CHUNK_SIZE)
        return int(size) if size is not None else 512

    @classmethod
    def _get_chunk_overlap(cls, feature: Feature) -> int:
        """Get chunk overlap from feature options."""
        overlap = feature.options.get(cls.CHUNK_OVERLAP)
        return int(overlap) if overlap is not None else 50

    @classmethod
    @abstractmethod
    def _chunk_text(
        cls,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> List[str]:
        """
        Split a single text into chunks.

        Args:
            text: Text to split
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        ...

    @classmethod
    def calculate_feature(cls, data: List[Dict[str, Any]], features: FeatureSet) -> List[Dict[str, Any]]:
        """Perform chunking on the source feature."""
        result = []

        for feature in features.features:
            source_feature = cls._get_source_feature_name(feature)
            chunk_size = cls._get_chunk_size(feature)
            chunk_overlap = cls._get_chunk_overlap(feature)
            feature_name = feature.get_name()

            for row in data:
                # Get text from source feature or 'text' field
                if source_feature in row:
                    text = str(row[source_feature])
                elif "text" in row:
                    text = str(row["text"])
                else:
                    text = ""

                chunks = cls._chunk_text(text, chunk_size, chunk_overlap)

                # Create a new row for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    new_row = row.copy()
                    new_row[feature_name] = chunk
                    new_row["chunk_index"] = chunk_idx
                    new_row["chunk_count"] = len(chunks)
                    # Create chunk_id from doc_id if available
                    if "doc_id" in row:
                        new_row["chunk_id"] = f"{row['doc_id']}_chunk_{chunk_idx}"
                    result.append(new_row)

        return result
