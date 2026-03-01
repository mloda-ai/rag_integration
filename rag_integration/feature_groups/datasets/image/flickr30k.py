"""Flickr30K dataset source for image retrieval evaluation."""

from __future__ import annotations

import ast
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

from mloda.user import Options

from rag_integration.feature_groups.datasets.image.base import BaseImageDatasetSource


class Flickr30kDatasetSource(BaseImageDatasetSource):
    """
    Loads the Flickr30K dataset from a local directory for image retrieval evaluation.

    Flickr30K (Young et al., 2014) is one of the most widely cited image-text
    retrieval benchmarks with 5000+ citations. The test split contains 1,000 images
    each with 5 human-written captions, giving 5,000 text-to-image query pairs.

    Download the dataset first using the provided Jupyter notebook:
        /Volumes/ExtraStorage/mlodadatasetevaluation/download_datasets.ipynb

    Expected directory structure (from snapshot_download + unzip)::

        <data_dir>/
            flickr_annotations_30k.csv   ← captions + split labels
            flickr30k-images/            ← unzipped image files

    Configuration:
        data_dir (str, required): Path to the local flickr30k folder.
            E.g. "/Volumes/ExtraStorage/mlodadatasetevaluation/datasets/flickr30k_raw"
        max_samples (int, optional): Limit number of test images loaded (default: all).

    Output rows:
        Image rows:   {"image_id": str, "image_data": bytes, "format": "jpeg",
                       "row_type": "corpus"}
        Caption rows: {"image_id": str, "image_data": None, "caption": str,
                       "row_type": "query", "relevant_image_ids": [str]}

    Example::

        from mloda.user import Feature, Options
        feature = Feature("eval_images", options=Options(context={
            "data_dir": "/Volumes/ExtraStorage/mlodadatasetevaluation/datasets/flickr30k_raw",
            "max_samples": 100,
        }))
    """

    DATA_DIR = "data_dir"
    MAX_SAMPLES = "max_samples"

    @classmethod
    def _load_dataset(cls, options: Options) -> List[Dict[str, Any]]:
        """Load Flickr30K test split from local CSV + image files."""
        data_dir = options.get(cls.DATA_DIR)
        if not data_dir:
            raise ValueError(
                f"'{cls.DATA_DIR}' option is required. Set it to the local path of the flickr30k dataset folder."
            )

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("The 'pandas' package is required. Install with: pip install pandas") from e

        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError("The 'Pillow' package is required. Install with: pip install Pillow") from e

        max_samples: Optional[int] = options.get(cls.MAX_SAMPLES)

        data_path = Path(str(data_dir))
        csv_path = data_path / "flickr_annotations_30k.csv"
        images_dir = data_path / "flickr30k-images"

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")
        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}. Unzip flickr30k-images.zip into the data_dir first."
            )

        df = pd.read_csv(csv_path)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        if max_samples is not None:
            test_df = test_df.head(int(max_samples))

        rows: List[Dict[str, Any]] = []

        for _, row in test_df.iterrows():
            filename = str(row["filename"])
            img_id = str(row["img_id"])
            captions: List[str] = ast.literal_eval(str(row["raw"]))

            # Load image bytes
            img_path = images_dir / filename
            image_data: Optional[bytes] = None
            img_format = "jpeg"
            if img_path.exists():
                with Image.open(img_path) as img:
                    buf = io.BytesIO()
                    fmt = img.format or "JPEG"
                    img.save(buf, format=fmt)
                    image_data = buf.getvalue()
                    img_format = fmt.lower()

            # Corpus row — the image itself
            rows.append(
                {
                    "image_id": img_id,
                    "image_data": image_data,
                    "format": img_format,
                    "row_type": "corpus",
                    "metadata": {"filename": filename},
                }
            )

            # Query rows — one per caption
            for cap_idx, caption in enumerate(captions):
                rows.append(
                    {
                        "image_id": f"{img_id}_cap{cap_idx}",
                        "image_data": None,
                        "caption": caption,
                        "row_type": "query",
                        "relevant_image_ids": [img_id],
                    }
                )

        return rows
