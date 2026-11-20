"""Artifact for storing pre-computed embeddings."""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
from typing import Any

from mloda.provider import BaseArtifact, FeatureSet

from rag_integration.feature_groups.rag_pipeline.artifact_storage import atomic_write, resolve_storage_dir


class EmbeddingArtifact(BaseArtifact):
    """
    Artifact for storing pre-computed embeddings.

    This artifact stores computed embeddings using joblib serialization,
    allowing for efficient persistence and reuse of embeddings between runs.

    The artifact contains:
    - embeddings: The computed embedding vectors
    - model_name: Name of the model used to generate embeddings
    - embedding_dim: Dimension of the embedding vectors
    - text_hashes: Hashes of input texts (for validation)
    - metadata: Additional metadata about the embeddings

    This class provides helper methods for managing embedding artifacts
    similar to the SklearnArtifact pattern.
    """

    @classmethod
    def _get_artifact_file_path_for_key(cls, features: FeatureSet, artifact_key: str) -> Path:
        """
        Generate a file path for storing a specific embedding artifact by key.

        Args:
            features: The feature set
            artifact_key: The specific artifact key (e.g., "text__embedded_all-MiniLM-L6-v2")

        Returns:
            Path object for the artifact file
        """
        # Hash the artifact key to ensure valid filename
        key_hash = hashlib.md5(artifact_key.encode(), usedforsecurity=False).hexdigest()[:12]
        return cls._get_storage_dir(features, create=True) / f"embedding_artifact_{key_hash}.joblib"

    @classmethod
    def _get_storage_dir(cls, features: FeatureSet, create: bool) -> Path:
        options = cls.get_singular_option_from_options(features)
        return resolve_storage_dir(options.get("artifact_storage_path") if options else None, create)

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Any | None:
        """
        Save embedding artifacts to file(s).

        Args:
            features: The feature set
            artifact: Dictionary of {artifact_key: artifact_data} where each artifact_data
                     contains embeddings, model_name, embedding_dim, etc.

        Returns:
            Dictionary of {artifact_key: file_path} where artifacts were saved
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for EmbeddingArtifact. Install with: pip install joblib")

        if not isinstance(artifact, dict):
            raise ValueError(f"Expected artifact to be a dictionary, got {type(artifact)}")

        saved_paths = {}

        for artifact_key, artifact_data in artifact.items():
            # Generate unique file path for this artifact
            file_path = cls._get_artifact_file_path_for_key(features, artifact_key)

            atomic_write(file_path, functools.partial(joblib.dump, artifact_data))
            saved_paths[artifact_key] = str(file_path)

        return saved_paths

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Any | None:
        """
        Load embedding artifacts from file(s).

        Files are unpickled by joblib, so storage paths and artifact values must be trusted.

        Args:
            features: The feature set

        Returns:
            Dictionary of {artifact_key: artifact_data} containing all available artifacts
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for EmbeddingArtifact. Install with: pip install joblib")

        storage_dir = cls._get_storage_dir(features, create=False)

        # Find all embedding artifact files
        pattern = "embedding_artifact_*.joblib"
        loaded_artifacts: dict[str, Any] = {}

        for file_path in storage_dir.glob(pattern) if storage_dir.exists() else ():
            try:
                # Load the artifact
                artifact_data = joblib.load(file_path)

                # Extract the artifact key from the stored metadata
                if isinstance(artifact_data, dict) and "artifact_key" in artifact_data:
                    artifact_key = artifact_data["artifact_key"]
                    loaded_artifacts[artifact_key] = artifact_data
                else:
                    # Fallback: use filename hash as key
                    filename = file_path.stem
                    if filename.startswith("embedding_artifact_"):
                        key_hash = filename[len("embedding_artifact_") :]
                        loaded_artifacts[key_hash] = artifact_data

            except (OSError, ValueError) as e:
                raise OSError(f"Failed to load artifact from {file_path}: {e}") from e

        # The saved file this feature was given, via Options or run(artifacts=...), wins for its key.
        if isinstance(features.artifact_value, (str, os.PathLike)) and isinstance(features.artifact_to_load, str):
            loaded_artifacts[features.artifact_to_load] = joblib.load(features.artifact_value)

        return loaded_artifacts or None

    @classmethod
    def load_embedding_artifact(cls, features: FeatureSet, artifact_key: str) -> dict[str, Any] | None:
        """
        Helper method to load a specific embedding artifact by key.

        Args:
            features: The feature set
            artifact_key: The specific artifact key to load

        Returns:
            The artifact data if found, None otherwise
        """
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            if artifacts and artifact_key in artifacts:
                return artifacts[artifact_key]  # type: ignore
            # If artifact_to_load is true but we can't find the specific key, that's an error
            available_keys = list(artifacts.keys()) if artifacts else []
            raise ValueError(
                f"Embedding artifact not found for key '{artifact_key}'. Available artifacts: {available_keys}"
            )
        return None

    @classmethod
    def save_embedding_artifact(cls, features: FeatureSet, artifact_key: str, artifact_data: dict[str, Any]) -> None:
        """
        Helper method to save an embedding artifact with the proper multiple artifact format.

        Args:
            features: The feature set
            artifact_key: The unique key for this artifact
            artifact_data: The artifact data to save (should include embeddings, model_name, etc.)
        """
        if features.artifact_to_save:
            # Add the artifact key to the data for later retrieval
            artifact_data["artifact_key"] = artifact_key

            # Support multiple artifacts by using a dictionary
            if not isinstance(features.save_artifact, dict):
                features.save_artifact = {}
            features.save_artifact[artifact_key] = artifact_data

    @classmethod
    def check_artifact_exists(cls, features: FeatureSet, artifact_key: str) -> bool:
        """
        Helper method to check if a specific embedding artifact exists.

        Args:
            features: The feature set
            artifact_key: The artifact key to check

        Returns:
            True if the artifact exists, False otherwise
        """
        if features.artifact_to_load:
            artifacts = cls.custom_loader(features)
            return artifacts is not None and artifact_key in artifacts
        return False
