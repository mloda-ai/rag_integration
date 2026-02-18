"""
Manual test: Run a real image through the full 5-stage image pipeline.

Usage:
    conda activate mloda
    python test_manual_image.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Type

from mloda.user import mlodaAPI, PluginCollector
from mloda.provider import DataCreator, FeatureGroup
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from rag_integration.feature_groups.image_pipeline import (
    SolidFillPIIRedactor,
    ResizePreprocessor,
    ExactHashImageDeduplicator,
    CLIPImageEmbedder,
)

# Path to test image
IMAGE_PATH = Path(__file__).parent.parent / "test_image.png"


class RealImageDataCreator(FeatureGroup):
    """DataCreator that loads a real image file."""

    @classmethod
    def input_data(cls) -> DataCreator:
        return DataCreator({"image_docs"})

    @classmethod
    def match_feature_group_criteria(cls, feature_name: Any, options: Any, data_access_collection: Any = None) -> bool:
        return str(feature_name) == "image_docs"

    @classmethod
    def compute_framework_rule(cls) -> Set[Type[Any]]:
        return {PythonDictFramework}

    @classmethod
    def calculate_feature(cls, data: Any, features: Any) -> List[Dict[str, Any]]:
        image_data = IMAGE_PATH.read_bytes()
        fmt = IMAGE_PATH.suffix.lstrip(".").lower()
        if fmt == "jpg":
            fmt = "jpeg"

        # Load the same image twice to test deduplication
        return [
            {
                "image_docs": image_data,
                "image_id": "test_image_1",
                "image_data": image_data,
                "format": fmt,
            },
            {
                "image_docs": image_data,
                "image_id": "test_image_2_duplicate",
                "image_data": image_data,
                "format": fmt,
            },
        ]


def get_providers() -> Set[Type[FeatureGroup]]:
    return {
        RealImageDataCreator,
        SolidFillPIIRedactor,
        ResizePreprocessor,
        ExactHashImageDeduplicator,
        CLIPImageEmbedder,
    }


def flatten_result(result: List[Any]) -> List[Dict[str, Any]]:
    if result and isinstance(result[0], list):
        return result[0]
    return result


def main() -> None:
    if not IMAGE_PATH.exists():
        print(f"Image not found: {IMAGE_PATH}")
        sys.exit(1)

    print(f"Loading image: {IMAGE_PATH}")
    print(f"Image size: {IMAGE_PATH.stat().st_size:,} bytes")
    print()

    feature_names = [
        "image_docs",
        "image_docs__pii_redacted",
        "image_docs__pii_redacted__preprocessed",
        "image_docs__pii_redacted__preprocessed__deduped",
        "image_docs__pii_redacted__preprocessed__deduped__embedded",
    ]

    print("Running full pipeline...")
    raw_result = mlodaAPI.run_all(
        features=list(feature_names),
        compute_frameworks={PythonDictFramework},
        plugin_collector=PluginCollector.enabled_feature_groups(get_providers()),
    )

    print()
    for i, name in enumerate(feature_names):
        rows = flatten_result([raw_result[i]])
        print(f"Stage: {name}")
        print(f"  Rows: {len(rows)}")

        if name == "image_docs":
            for row in rows:
                print(f"  - {row.get('image_id')}: {len(row.get('image_data', b'')):,} bytes, format={row.get('format')}")

        elif "pii_redacted" in name and "preprocessed" not in name:
            for row in rows:
                data = row.get(name, b"")
                print(f"  - {row.get('image_id')}: {len(data):,} bytes (redacted)")

        elif "preprocessed" in name and "deduped" not in name:
            for row in rows:
                data = row.get(name, b"")
                size = row.get("preprocessed_size", "?")
                print(f"  - {row.get('image_id')}: {len(data):,} bytes, target_size={size}")

        elif "deduped" in name and "embedded" not in name:
            for row in rows:
                is_dup = row.get("is_duplicate", False)
                print(f"  - {row.get('image_id')}: is_duplicate={is_dup}")

        elif "embedded" in name:
            for row in rows:
                embedding = row.get(name, [])
                magnitude = math.sqrt(sum(x * x for x in embedding)) if embedding else 0
                print(f"  - {row.get('image_id')}: dim={len(embedding)}, magnitude={magnitude:.4f}")
                print(f"    embedding (first 10): {embedding[:10]}")
                print(f"    embedding (last 10):  {embedding[-10:]}")

        print()

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
