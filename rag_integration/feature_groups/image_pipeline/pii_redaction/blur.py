"""Gaussian blur PII redaction for images."""

from __future__ import annotations

from typing import Any, Dict, List

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor


class BlurPIIRedactor(BaseImagePIIRedactor):
    """
    Gaussian blur PII redactor for images.

    Applies Gaussian blur to specified bounding box regions to obscure
    PII content. Requires Pillow.

    Configuration:
        blur_radius: Radius of the Gaussian blur (default: 20)
        pii_regions: List of region dicts with 'bbox' and 'type'

    Config-based matching:
        image_redaction_method="blur"
    """

    BLUR_RADIUS = "blur_radius"

    PROPERTY_MAPPING = {
        BaseImagePIIRedactor.IMAGE_REDACTION_METHOD: {
            "blur": "Gaussian blur over PII regions",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BLUR_RADIUS: {
            "explanation": "Radius of the Gaussian blur effect",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 20,
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
        Apply Gaussian blur to PII regions.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            regions: List of region dicts with 'bbox' [x1, y1, x2, y2]

        Returns:
            Image bytes with blurred regions
        """
        try:
            from PIL import Image, ImageFilter
        except ImportError:
            raise ImportError("Pillow is required for BlurPIIRedactor. Install with: pip install Pillow")

        import io

        img = Image.open(io.BytesIO(image_data))

        for region in regions:
            bbox = region.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # Clamp to image bounds
            x1 = max(0, min(x1, img.width))
            y1 = max(0, min(y1, img.height))
            x2 = max(0, min(x2, img.width))
            y2 = max(0, min(y2, img.height))

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop, blur, paste back
            region_img = img.crop((x1, y1, x2, y2))
            blurred = region_img.filter(ImageFilter.GaussianBlur(radius=20))
            img.paste(blurred, (x1, y1))

        # Save back to bytes
        output = io.BytesIO()
        save_format = "PNG" if image_format.lower() == "png" else "JPEG"
        img.save(output, format=save_format)
        return output.getvalue()
