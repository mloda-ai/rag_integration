"""Solid fill PII redaction for images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor

if TYPE_CHECKING:
    from mloda.user import Feature


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

    # Default solid fill color (black), kept in sync with PROPERTY_MAPPING.
    DEFAULT_FILL_COLOR = (0, 0, 0)

    PROPERTY_MAPPING = {
        BaseImagePIIRedactor.IMAGE_REDACTION_METHOD: {
            "solid": "Solid color fill over PII regions",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        FILL_COLOR: {
            "explanation": "RGB color tuple for solid fill (e.g., [0, 0, 0] for black)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: list(DEFAULT_FILL_COLOR),
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
    def _get_fill_color(cls, feature: "Feature") -> Tuple[int, int, int]:
        """Get the RGB fill color from feature options."""
        value = feature.options.get(cls.FILL_COLOR)
        if value is None:
            return cls.DEFAULT_FILL_COLOR
        channels = [int(c) for c in value]
        if len(channels) != 3:
            raise ValueError(f"{cls.FILL_COLOR} must be an RGB triple, got {len(channels)} value(s): {value!r}")
        r, g, b = channels
        return (r, g, b)

    @classmethod
    def _redact_region_for_feature(
        cls,
        image_data: bytes,
        image_format: str,
        regions: List[Dict[str, Any]],
        feature: "Feature",
    ) -> bytes:
        """Apply solid fill using the per-feature fill color."""
        return cls._redact_region(image_data, image_format, regions, fill_color=cls._get_fill_color(feature))

    @classmethod
    def _redact_region(
        cls,
        image_data: bytes,
        image_format: str,
        regions: List[Dict[str, Any]],
        fill_color: Tuple[int, int, int] = DEFAULT_FILL_COLOR,
    ) -> bytes:
        """
        Apply solid fill to PII regions.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            regions: List of region dicts with 'bbox' [x1, y1, x2, y2]
            fill_color: RGB color tuple for the solid fill (default: black)

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
