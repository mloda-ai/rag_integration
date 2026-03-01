"""Artifact for storing FAISS vector indices with metadata."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from mloda.provider import BaseArtifact, FeatureSet


class VectorStoreArtifact(BaseArtifact):
    """
    Artifact for storing FAISS vector indices with text metadata.

    Uses native FAISS serialization (faiss.write_index/read_index) for the index
    and a JSON sidecar file for the text/doc_id mapping.

    Files produced:
        - vector_store_{hash}.faiss: The FAISS index binary
        - vector_store_{hash}_metadata.json: Text and doc_id mapping
    """

    @classmethod
    def _get_storage_path(cls, features: FeatureSet) -> Path:
        """Get the storage directory from options or use temp directory."""
        storage_path = None

        options = cls.get_singular_option_from_options(features)
        if options:
            storage_path = options.get("artifact_storage_path")

        if storage_path is None:
            storage_path = tempfile.gettempdir()

        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        return storage_dir

    @classmethod
    def _get_artifact_file_paths(cls, features: FeatureSet, artifact_key: str) -> tuple[Path, Path]:
        """
        Generate file paths for the FAISS index and metadata sidecar.

        Returns:
            Tuple of (index_path, metadata_path)
        """
        storage_dir = cls._get_storage_path(features)
        key_hash = hashlib.md5(artifact_key.encode(), usedforsecurity=False).hexdigest()[:12]
        index_path = storage_dir / f"vector_store_{key_hash}.faiss"
        metadata_path = storage_dir / f"vector_store_{key_hash}_metadata.json"
        return index_path, metadata_path

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
        """
        Save FAISS index and metadata to files.

        Args:
            features: The feature set
            artifact: Dictionary of {artifact_key: artifact_data} where artifact_data contains
                     "index" (faiss.Index) and "metadata" (dict with texts/doc_ids)

        Returns:
            Dictionary of {artifact_key: {"index_path": str, "metadata_path": str}}
        """
        import faiss

        if not isinstance(artifact, dict):
            raise ValueError(f"Expected artifact to be a dictionary, got {type(artifact)}")

        saved_paths: Dict[str, Dict[str, str]] = {}

        for artifact_key, artifact_data in artifact.items():
            index_path, metadata_path = cls._get_artifact_file_paths(features, artifact_key)

            # Save FAISS index using native serialization
            faiss.write_index(artifact_data["index"], str(index_path))

            # Save metadata sidecar as JSON
            metadata = artifact_data.get("metadata", {})
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            saved_paths[artifact_key] = {
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
            }

        return saved_paths

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        Load all FAISS index artifacts from the storage directory.

        Returns:
            Dictionary of {artifact_key: {"index": faiss.Index, "metadata": dict}}
        """
        import faiss

        storage_dir = cls._get_storage_path(features)
        if not storage_dir.exists():
            return None

        loaded_artifacts: Dict[str, Any] = {}

        for index_path in storage_dir.glob("vector_store_*.faiss"):
            # Find the corresponding metadata file
            stem = index_path.stem  # e.g. vector_store_abc123
            metadata_path = index_path.parent / f"{stem}_metadata.json"

            index = faiss.read_index(str(index_path))

            metadata: Dict[str, Any] = {}
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)

            artifact_key = metadata.get("artifact_key", stem)
            loaded_artifacts[artifact_key] = {
                "index": index,
                "metadata": metadata,
            }

        if loaded_artifacts:
            return loaded_artifacts
        return None

    @classmethod
    def save_vector_store_artifact(
        cls,
        features: FeatureSet,
        artifact_key: str,
        index: Any,
        texts: List[str],
        doc_ids: List[str],
    ) -> None:
        """
        Helper to queue a vector store artifact for saving.

        Args:
            features: The feature set
            artifact_key: Unique key for this artifact
            index: The FAISS index object
            texts: List of text strings corresponding to each vector
            doc_ids: List of document IDs corresponding to each vector
        """
        if features.artifact_to_save:
            metadata = {
                "artifact_key": artifact_key,
                "texts": texts,
                "doc_ids": doc_ids,
                "num_vectors": index.ntotal,
            }

            if not isinstance(features.save_artifact, dict):
                features.save_artifact = {}
            features.save_artifact[artifact_key] = {
                "index": index,
                "metadata": metadata,
            }

    @classmethod
    def load_vector_store_artifact(cls, features: FeatureSet, artifact_key: str) -> Optional[Dict[str, Any]]:
        """
        Helper to load a specific vector store artifact by key.

        Returns:
            Dict with "index" and "metadata" if found, None otherwise.
        """
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            if artifacts and artifact_key in artifacts:
                return artifacts[artifact_key]  # type: ignore
            available_keys = list(artifacts.keys()) if artifacts else []
            raise ValueError(
                f"Vector store artifact not found for key '{artifact_key}'. Available artifacts: {available_keys}"
            )
        return None

    @classmethod
    def check_artifact_exists(cls, features: FeatureSet, artifact_key: str) -> bool:
        """Check if a specific vector store artifact exists."""
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            return artifacts is not None and artifact_key in artifacts
        return False
