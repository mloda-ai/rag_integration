"""File-based image source."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from mloda.user import Options

from rag_integration.feature_groups.image_pipeline.image_source.base import BaseImageSource


# Supported image extensions
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"}


class FileImageSource(BaseImageSource):
    """
    File-based image source.

    Loads images from a directory or list of file paths.
    Supports common image formats: PNG, JPEG, GIF, BMP, TIFF, WebP.

    Configuration:
        image_dir: Path to directory containing images
        file_paths: List of specific file paths (alternative to image_dir)
        recursive: Whether to search subdirectories (default: False)

    Usage:
        Feature("image_docs", Options(context={
            "image_dir": "/path/to/images",
            "recursive": True,
        }))
    """

    @classmethod
    def _load_images(cls, options: Options) -> List[Dict[str, Any]]:
        """
        Load images from files.

        Args:
            options: Options containing image_dir or file_paths

        Returns:
            List of image dictionaries

        Raises:
            ValueError: If neither image_dir nor file_paths provided
        """
        image_dir = options.get("image_dir") if options else None
        file_paths = options.get("file_paths") if options else None
        recursive = options.get("recursive") if options else None
        recursive = bool(recursive) if recursive is not None else False

        paths: List[Path] = []

        if file_paths:
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            paths = [Path(p) for p in file_paths]
        elif image_dir:
            dir_path = Path(image_dir)
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {image_dir}")

            if recursive:
                for ext in SUPPORTED_FORMATS:
                    paths.extend(dir_path.rglob(f"*{ext}"))
            else:
                for ext in SUPPORTED_FORMATS:
                    paths.extend(dir_path.glob(f"*{ext}"))
            paths.sort()
        else:
            raise ValueError("Either image_dir or file_paths is required for FileImageSource")

        images = []
        for i, path in enumerate(paths):
            if not path.exists():
                raise ValueError(f"File not found: {path}")

            image_data = path.read_bytes()
            fmt = path.suffix.lstrip(".").lower()
            if fmt == "jpg":
                fmt = "jpeg"

            images.append(
                {
                    "image_id": path.stem,
                    "image_data": image_data,
                    "format": fmt,
                    "file_path": str(path),
                }
            )

        return images
