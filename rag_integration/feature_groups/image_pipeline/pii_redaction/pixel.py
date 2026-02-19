"""Pixelation PII redaction for images."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor


class PixelPIIRedactor(BaseImagePIIRedactor):
    """
    Pixelation PII redactor for images.

    Pixelates specified bounding box regions by downscaling and upscaling
    to create a mosaic effect. Requires Pillow.

    Configuration:
        pixel_size: Size of each pixel block (default: 10)
        pii_regions: List of region dicts with 'bbox' and 'type'

    Config-based matching:
        image_redaction_method="pixel"
    """

    PIXEL_SIZE = "pixel_size"

    PROPERTY_MAPPING = {
        BaseImagePIIRedactor.IMAGE_REDACTION_METHOD: {
            "pixel": "Pixelate PII regions",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        PIXEL_SIZE: {
            "explanation": "Size of each pixel block in the mosaic effect",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 10,
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
        Apply pixelation to PII regions.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            regions: List of region dicts with 'bbox' [x1, y1, x2, y2]

        Returns:
            Image bytes with pixelated regions
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for PixelPIIRedactor. Install with: pip install Pillow")

        import io

        img: Image.Image = Image.open(io.BytesIO(image_data))
        pixel_size = 10

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

            # Crop region
            region_img = img.crop((x1, y1, x2, y2))
            region_w, region_h = region_img.size

            # Downscale then upscale to pixelate
            small_w = max(1, region_w // pixel_size)
            small_h = max(1, region_h // pixel_size)
            small = region_img.resize((small_w, small_h), Image.Resampling.NEAREST)
            pixelated = small.resize((region_w, region_h), Image.Resampling.NEAREST)

            img.paste(pixelated, (x1, y1))

        output = io.BytesIO()
        save_format = "PNG" if image_format.lower() == "png" else "JPEG"
        img.save(output, format=save_format)
        return output.getvalue()
