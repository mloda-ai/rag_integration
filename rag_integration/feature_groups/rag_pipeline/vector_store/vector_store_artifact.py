"""Artifact for storing FAISS vector indices with metadata."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, ClassVar

from mloda.provider import BaseArtifact, FeatureSet

from rag_integration.feature_groups.rag_pipeline.artifact_storage import atomic_write, resolve_storage_dir


class VectorStoreArtifact(BaseArtifact):
    """
    Artifact for storing FAISS vector indices with text metadata.

    Uses native FAISS serialization (faiss.write_index/read_index) for the index
    and a JSON sidecar file for the text/doc_id mapping.

    Files produced:
        - vector_store_{hash}.faiss: The FAISS index binary
        - vector_store_{hash}_metadata.json: Text and doc_id mapping
    """

    # Keeps an index and its sidecar from two concurrent runs from mixing.
    _save_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _get_storage_path(cls, features: FeatureSet, create: bool = True) -> Path:
        """Get the storage directory from options or the private default."""
        options = cls.get_singular_option_from_options(features)
        return resolve_storage_dir(options.get("artifact_storage_path") if options else None, create)

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
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
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

        saved_paths: dict[str, dict[str, str]] = {}

        for artifact_key, artifact_data in artifact.items():
            index_path, metadata_path = cls._get_artifact_file_paths(features, artifact_key)

            metadata = artifact_data.get("metadata", {})

            def write_metadata(tmp: str, metadata: dict[str, Any] = metadata) -> None:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False)

            # Sidecar first: the index is what the loader globs for.
            with cls._save_lock:
                atomic_write(metadata_path, write_metadata)
                atomic_write(index_path, lambda tmp, index=artifact_data["index"]: faiss.write_index(index, tmp))

            saved_paths[artifact_key] = {
                "index_path": str(index_path),
                "metadata_path": str(metadata_path),
            }

        return saved_paths

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Any | None:
        """
        Load all FAISS index artifacts from the storage directory.

        Returns:
            Dictionary of {artifact_key: {"index": faiss.Index, "metadata": dict}}
        """
        loaded_artifacts: dict[str, Any] = {}
        storage_dir = cls._get_storage_path(features, create=False)

        for index_path in storage_dir.glob("vector_store_*.faiss") if storage_dir.exists() else ():
            loaded = cls._load_files(index_path)
            loaded_artifacts[loaded["metadata"].get("artifact_key", index_path.stem)] = loaded

        # The saved paths this feature was given, via Options or run(artifacts=...), win for its key.
        saved = features.artifact_value
        if isinstance(saved, dict) and "index_path" in saved and isinstance(features.artifact_to_load, str):
            metadata_path = saved.get("metadata_path")
            loaded_artifacts[features.artifact_to_load] = cls._load_files(
                Path(saved["index_path"]), Path(metadata_path) if metadata_path else None
            )

        return loaded_artifacts or None

    @staticmethod
    def _load_files(index_path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
        """Read one FAISS index and its optional metadata sidecar (default: next to the index)."""
        import faiss

        if metadata_path is None:
            metadata_path = index_path.with_name(f"{index_path.stem}_metadata.json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        index = faiss.read_index(str(index_path))
        num_vectors = metadata.get("num_vectors")
        if num_vectors is not None and num_vectors != index.ntotal:
            raise ValueError(
                f"Vector store sidecar {metadata_path} lists {num_vectors} vectors, index {index_path} has {index.ntotal}"
            )
        return {"index": index, "metadata": metadata}

    @classmethod
    def save_vector_store_artifact(
        cls,
        features: FeatureSet,
        artifact_key: str,
        index: Any,
        texts: list[str],
        doc_ids: list[str],
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
    def load_vector_store_artifact(cls, features: FeatureSet, artifact_key: str) -> dict[str, Any] | None:
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
