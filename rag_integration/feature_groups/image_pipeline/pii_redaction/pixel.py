"""Pixelation PII redaction for images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from mloda.provider import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.pii_redaction.base import BaseImagePIIRedactor

if TYPE_CHECKING:
    from mloda.user import Feature


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

    # Default pixel block size (kept in sync with PROPERTY_MAPPING).
    DEFAULT_PIXEL_SIZE = 10

    PROPERTY_MAPPING = {
        BaseImagePIIRedactor.IMAGE_REDACTION_METHOD: {
            "pixel": "Pixelate PII regions",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        PIXEL_SIZE: {
            "explanation": "Size of each pixel block in the mosaic effect",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: DEFAULT_PIXEL_SIZE,
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
    def _get_pixel_size(cls, feature: "Feature") -> int:
        """Get the pixel block size from feature options."""
        value = feature.options.get(cls.PIXEL_SIZE)
        return int(value) if value is not None else cls.DEFAULT_PIXEL_SIZE

    @classmethod
    def _redact_region_for_feature(
        cls,
        image_data: bytes,
        image_format: str,
        regions: List[Dict[str, Any]],
        feature: "Feature",
    ) -> bytes:
        """Apply pixelation using the per-feature pixel size."""
        return cls._redact_region(image_data, image_format, regions, pixel_size=cls._get_pixel_size(feature))

    @classmethod
    def _redact_region(
        cls,
        image_data: bytes,
        image_format: str,
        regions: List[Dict[str, Any]],
        pixel_size: int = DEFAULT_PIXEL_SIZE,
    ) -> bytes:
        """
        Apply pixelation to PII regions.

        Args:
            image_data: Raw image bytes
            image_format: Image format (png, jpeg, etc.)
            regions: List of region dicts with 'bbox' [x1, y1, x2, y2]
            pixel_size: Size of each pixel block in the mosaic effect (default: 10)

        Returns:
            Image bytes with pixelated regions
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for PixelPIIRedactor. Install with: pip install Pillow")

        import io

        img: Image.Image = Image.open(io.BytesIO(image_data))
        pixel_size = max(1, pixel_size)  # guard against division by zero

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
