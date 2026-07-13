"""Source rows may carry per-item extra fields, so the row set is heterogeneous.

``metadata`` is optional per document/image and the loaders preserve arbitrary extra fields. mloda
0.9.0 rejects heterogeneous ``list[dict]`` instead of union-of-keys normalizing it, so the sources
homogenize on the way out. Without that, a document list where one doc carries ``metadata`` and
another does not dies inside mloda with "Inconsistent row keys", which no uniform fixture catches.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from mloda.user import mlodaAPI, Feature, Options, PluginCollector
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.image_pipeline.image_source import DictImageSource
from rag_integration.feature_groups.rag_pipeline.chunking.fixed_size import FixedSizeChunker
from rag_integration.feature_groups.rag_pipeline.document_source import DictDocumentSource
from tests.integration.helpers import flatten_result

_DOCS: List[Dict[str, Any]] = [
    {"doc_id": "d1", "text": "first doc text", "metadata": {"src": "a"}},
    {"doc_id": "d2", "text": "second doc text"},
]

_IMAGES: List[Dict[str, Any]] = [
    {"image_id": "i1", "image_data": b"\x89PNG_1", "format": "png", "caption": "a cat"},
    {"image_id": "i2", "image_data": b"\xff\xd8_2", "format": "jpeg"},
]


def _feature_set(options: Options) -> Any:
    feature = MagicMock()
    feature.options = options
    features = MagicMock()
    features.features = [feature]
    return features


def test_documents_with_uneven_extra_fields_run_through_a_chain() -> None:
    """The README's entry point: DictDocumentSource feeding a chunker. Raised ValueError before."""
    source = Feature("docs", options=Options(context={"documents": _DOCS}))
    chunked = Feature(
        "result_chunked",
        options=Options(context={"chunking_method": "fixed_size", "in_features": source}),
    )
    result = mlodaAPI.run_all(
        [chunked],
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups({DictDocumentSource, FixedSizeChunker}),
    )

    rows = flatten_result(result[0])
    assert [row["result_chunked"] for row in rows] == ["first doc text", "second doc text"]


def test_document_source_fills_the_absent_key() -> None:
    rows = DictDocumentSource.calculate_feature(None, _feature_set(Options(context={"documents": _DOCS})))

    assert rows[0]["metadata"] == {"src": "a"}
    # Filled, not dropped: every row carries the union, so the frame has one schema.
    assert rows[1]["metadata"] is None
    assert {key for row in rows for key in row} == set(rows[0])


def test_image_source_fills_the_absent_key() -> None:
    rows = DictImageSource.calculate_feature(None, _feature_set(Options(context={"images": _IMAGES})))

    assert rows[0]["caption"] == "a cat"
    assert rows[1]["caption"] is None
    assert {key for row in rows for key in row} == set(rows[0])
