"""Difference hash-based image deduplication."""

from __future__ import annotations

from typing import List, Optional

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator


class DifferenceHashImageDeduplicator(BaseImageDeduplicator):
    """
    Difference hash-based image deduplicator.

    Uses difference hashing (dHash) to detect visually similar images.
    dHash compares adjacent pixels in a resized grayscale image.

    The dHash algorithm:
    1. Resize image to (hash_size+1) x hash_size grayscale
    2. Compare each pixel with its right neighbor
    3. Hash bit is 1 if left pixel > right pixel

    Faster than pHash but slightly less robust to transformations.
    Requires Pillow.

    Config-based matching:
        image_deduplication_method="dhash"
    """

    PROPERTY_MAPPING = {
        BaseImageDeduplicator.IMAGE_DEDUPLICATION_METHOD: {
            "dhash": "Difference hash-based near-duplicate detection",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseImageDeduplicator.SIMILARITY_THRESHOLD: {
            "explanation": "Threshold for considering images as duplicates (0.0-1.0)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 0.9,
        },
        BaseImageDeduplicator.KEEP_STRATEGY: {
            **BaseImageDeduplicator.KEEP_STRATEGIES,
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "first",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing images to deduplicate",
            DefaultOptionKeys.context: True,
        },
    }

    @classmethod
    def _compute_dhash(cls, image_data: bytes, hash_size: int = 8) -> int:
        """
        Compute difference hash for an image.

        Args:
            image_data: Raw image bytes
            hash_size: Size of the hash (default 8 = 64-bit hash)

        Returns:
            Integer hash value
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for DifferenceHashImageDeduplicator. Install with: pip install Pillow"
            )

        import io

        img: Image.Image = Image.open(io.BytesIO(image_data))
        # Resize to (hash_size + 1) x hash_size grayscale
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

        pixels: list[int] = list(img.get_flattened_data())  # type: ignore[arg-type]
        width = hash_size + 1

        hash_val = 0
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * width + col]
                right = pixels[row * width + col + 1]
                hash_val = (hash_val << 1) | (1 if left > right else 0)

        return hash_val

    @classmethod
    def _hamming_distance(cls, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two integer hashes."""
        return bin(hash1 ^ hash2).count("1")

    @classmethod
    def _find_duplicates(
        cls,
        image_data_list: List[bytes],
        threshold: float,
    ) -> List[Optional[int]]:
        """
        Find near-duplicates using difference hashing.

        Args:
            image_data_list: List of image bytes
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            List where each element is either None (not a duplicate)
            or the index of the image it duplicates.
        """
        hash_size = 8
        total_bits = hash_size * hash_size
        max_distance = int((1.0 - threshold) * total_bits)

        hashes: List[int] = []
        result: List[Optional[int]] = []

        for i, image_data in enumerate(image_data_list):
            if not image_data:
                hashes.append(0)
                result.append(None)
                continue

            current_hash = cls._compute_dhash(image_data, hash_size)
            hashes.append(current_hash)

            found_dup = False
            for j in range(i):
                distance = cls._hamming_distance(current_hash, hashes[j])
                if distance <= max_distance:
                    result.append(j)
                    found_dup = True
                    break

            if not found_dup:
                result.append(None)

        return result
