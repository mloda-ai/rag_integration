"""Solid fill PII redaction for images."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor


class SolidFillPIIRedactor(BaseImagePIIRedactor):
    """
    Solid color fill PII redactor for images.

    Fills specified bounding box regions with a solid color to completely
    obscure PII content. Requires Pillow.

    Configuration:
        fill_color: RGB tuple for fill color (default: (0, 0, 0) = black)
        pii_regions: List of region dicts with 'bbox' and 'type'

    Config-based matching:
        image_redaction_method="solid"
    """

    FILL_COLOR = "fill_color"

    PROPERTY_MAPPING = {
        BaseImagePIIRedactor.IMAGE_REDACTION_METHOD: {
            "solid": "Solid color fill over PII regions",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        FILL_COLOR: {
            "explanation": "RGB color tuple for solid fill (e.g., [0, 0, 0] for black)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: [0, 0, 0],
        },
        BaseImagePIIRedactor.PII_REGIONS: {
            "explanation": "List of PII region dicts with 'bbox' [x1,y1,x2,y2] and 'type'",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: [],
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to redact",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _redact_region(
        cls,
        image_data: bytes,
        image_format: str,
        regions: List[Dict[str, Any]],
    ) -> bytes:
        """
        Apply solid fill to PII regions.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            regions: List of region dicts with 'bbox' [x1, y1, x2, y2]

        Returns:
            Image bytes with solid-filled regions
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError("Pillow is required for SolidFillPIIRedactor. Install with: pip install Pillow")

        import io

        img = Image.open(io.BytesIO(image_data))
        draw = ImageDraw.Draw(img)

        fill_color = (0, 0, 0)  # Default black

        for region in regions:
            bbox = region.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x1 = max(0, min(x1, img.width))
            y1 = max(0, min(y1, img.height))
            x2 = max(0, min(x2, img.width))
            y2 = max(0, min(y2, img.height))

            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        output = io.BytesIO()
        save_format = "PNG" if image_format.lower() == "png" else "JPEG"
        img.save(output, format=save_format)
        return output.getvalue()
