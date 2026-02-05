"""Artifact for storing pre-computed embeddings."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from mloda.provider import BaseArtifact, FeatureSet


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
        # Get storage path from options or use default temp directory
        storage_path = None

        options = cls.get_singular_option_from_options(features)
        if options:
            storage_path = options.get("artifact_storage_path")

        if storage_path is None:
            storage_path = tempfile.gettempdir()

        # Simple filename based on artifact key
        # Hash the artifact key to ensure valid filename
        key_hash = hashlib.md5(artifact_key.encode(), usedforsecurity=False).hexdigest()[:12]
        filename = f"embedding_artifact_{key_hash}.joblib"

        # Ensure the directory exists
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        return storage_dir / filename

    @classmethod
    def custom_saver(cls, features: FeatureSet, artifact: Any) -> Optional[Any]:
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

            # Save this specific artifact
            joblib.dump(artifact_data, file_path)
            saved_paths[artifact_key] = str(file_path)

        return saved_paths

    @classmethod
    def custom_loader(cls, features: FeatureSet) -> Optional[Any]:
        """
        Load embedding artifacts from file(s).

        Args:
            features: The feature set

        Returns:
            Dictionary of {artifact_key: artifact_data} containing all available artifacts
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required for EmbeddingArtifact. Install with: pip install joblib")

        # Get storage path
        storage_path = None

        options = cls.get_singular_option_from_options(features)
        if options:
            storage_path = options.get("artifact_storage_path")
        if storage_path is None:
            storage_path = tempfile.gettempdir()

        storage_dir = Path(storage_path)
        if not storage_dir.exists():
            return None

        # Find all embedding artifact files
        pattern = "embedding_artifact_*.joblib"
        loaded_artifacts = {}

        for file_path in storage_dir.glob(pattern):
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

            except Exception as e:
                print(f"Warning: Failed to load artifact from {file_path}: {e}")
                continue

        if loaded_artifacts:
            return loaded_artifacts
        else:
            return None

    @classmethod
    def load_embedding_artifact(cls, features: FeatureSet, artifact_key: str) -> Optional[Dict[str, Any]]:
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
            raise ValueError(f"Embedding artifact not found for key '{artifact_key}'. Available artifacts: {available_keys}")
        return None

    @classmethod
    def save_embedding_artifact(cls, features: FeatureSet, artifact_key: str, artifact_data: Dict[str, Any]) -> None:
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
