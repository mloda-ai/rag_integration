"""Perceptual hash-based image deduplication."""

from __future__ import annotations

from typing import Dict, List, Optional

from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.image_pipeline.deduplication.base import BaseImageDeduplicator


class PerceptualHashImageDeduplicator(BaseImageDeduplicator):
    """
    Perceptual hash-based image deduplicator.

    Uses perceptual hashing (pHash) to detect visually similar images
    even if they differ in format, resolution, or minor edits.

    The pHash algorithm:
    1. Resize image to 32x32
    2. Convert to grayscale
    3. Apply DCT (via simple mean-based approximation)
    4. Compute hash from top-left 8x8 DCT coefficients

    Requires Pillow.

    Config-based matching:
        image_deduplication_method="phash"
    """

    PROPERTY_MAPPING = {
        BaseImageDeduplicator.IMAGE_DEDUPLICATION_METHOD: {
            "phash": "Perceptual hash-based near-duplicate detection",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseImageDeduplicator.SIMILARITY_THRESHOLD: {
            "explanation": "Threshold for considering images as duplicates (0.0-1.0, lower = more strict)",
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
    def _compute_phash(cls, image_data: bytes, hash_size: int = 8) -> int:
        """
        Compute perceptual hash for an image.

        Uses a simplified pHash: resize to small grayscale, compare
        each pixel to the mean to produce a binary hash.

        Args:
            image_data: Raw image bytes
            hash_size: Size of the hash grid (default 8 = 64-bit hash)

        Returns:
            Integer hash value
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for PerceptualHashImageDeduplicator. Install with: pip install Pillow"
            )

        import io

        img = Image.open(io.BytesIO(image_data))
        # Resize to hash_size x hash_size grayscale
        img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)

        pixels = list(img.get_flattened_data())
        mean_val = sum(pixels) / len(pixels)

        # Build hash: 1 if pixel > mean, 0 otherwise
        hash_val = 0
        for pixel in pixels:
            hash_val = (hash_val << 1) | (1 if pixel > mean_val else 0)

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
        Find near-duplicates using perceptual hashing.

        Args:
            image_data_list: List of image bytes
            threshold: Similarity threshold (0.0-1.0). Higher = more permissive.

        Returns:
            List where each element is either None (not a duplicate)
            or the index of the image it duplicates.
        """
        hash_size = 8
        total_bits = hash_size * hash_size  # 64 bits
        max_distance = int((1.0 - threshold) * total_bits)

        hashes: List[int] = []
        result: List[Optional[int]] = []

        for i, image_data in enumerate(image_data_list):
            if not image_data:
                hashes.append(0)
                result.append(None)
                continue

            current_hash = cls._compute_phash(image_data, hash_size)
            hashes.append(current_hash)

            # Compare against all previous hashes
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
