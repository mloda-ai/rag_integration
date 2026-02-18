"""In-memory dictionary image source."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.image_source.base import BaseImageSource


class DictImageSource(BaseImageSource):
    """
    In-memory dictionary image source.

    Accepts images directly via Options. Useful for programmatic
    input and API integrations.

    Configuration:
        images: List of image dictionaries
        image_data_field: Field name containing image bytes (default: "image_data")
        id_field: Field name containing image ID (default: "image_id")

    Usage:
        Feature("image_docs", Options(context={
            "images": [
                {"image_id": "img_001", "image_data": b"...", "format": "png"},
                {"image_id": "img_002", "image_data": b"...", "format": "jpeg"},
            ]
        }))
    """

    @classmethod
    def _load_images(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load images from Options.

        Args:
            options: Options containing images list

        Returns:
            List of image dictionaries

        Raises:
            ValueError: If images not provided
        """
        images = options.get("images") if options else None
        image_data_field_opt = options.get("image_data_field") if options else None
        image_data_field = str(image_data_field_opt) if image_data_field_opt is not None else "image_data"
        id_field_opt = options.get("id_field") if options else None
        id_field = str(id_field_opt) if id_field_opt is not None else "image_id"

        if not images:
            raise ValueError("images list is required for DictImageSource")

        if not isinstance(images, list):
            images = [images]

        result = []
        for i, img in enumerate(images):
            if isinstance(img, bytes):
                # Handle raw bytes
                normalized: Dict[str, Any] = {
                    "image_id": f"img_{i}",
                    "image_data": img,
                    "format": "unknown",
                }
            else:
                normalized = {
                    "image_id": img.get(id_field, f"img_{i}"),
                    "image_data": img.get(image_data_field, b""),
                    "format": img.get("format", "unknown"),
                }
                # Preserve other fields
                for key, value in img.items():
                    if key not in (id_field, image_data_field, "format"):
                        normalized[key] = value
            result.append(normalized)

        return result
