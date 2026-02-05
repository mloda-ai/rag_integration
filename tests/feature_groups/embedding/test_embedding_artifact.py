"""Tests for EmbeddingArtifact."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_integration.feature_groups.rag_pipeline.embedding.embedding_artifact import EmbeddingArtifact


class TestEmbeddingArtifact:
    """Tests for the EmbeddingArtifact class."""

    def test_save_and_load_artifact(self, tmp_path: Path) -> None:
        """Test that embeddings are saved on first run and loaded on second run."""
        # Create mock features
        mock_feature = MagicMock()
        mock_feature.get_name.return_value = "text__embedded"
        mock_feature.options = MagicMock()
        mock_feature.options.get.side_effect = lambda key: str(tmp_path) if key == "artifact_storage_path" else None

        mock_features = MagicMock()
        mock_features.features = [mock_feature]
        mock_features.artifact_to_save = True
        mock_features.artifact_to_load = False
        mock_features.save_artifact = {}
        mock_features.name_of_one_feature = mock_feature

        # Create test embeddings
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        artifact_key = "text__embedded_test-model"
        artifact_data = {
            "embeddings": embeddings,
            "model_name": "test-model",
            "embedding_dim": 3,
            "num_texts": 2,
        }

        # Save the artifact
        EmbeddingArtifact.save_embedding_artifact(mock_features, artifact_key, artifact_data)

        # Verify the data was queued for saving
        assert artifact_key in mock_features.save_artifact
        assert mock_features.save_artifact[artifact_key]["embeddings"] == embeddings

        # Now simulate the actual save
        saved_paths = EmbeddingArtifact.custom_saver(mock_features, mock_features.save_artifact)

        # Verify file was saved
        assert saved_paths is not None
        assert artifact_key in saved_paths
        saved_path = Path(saved_paths[artifact_key])
        assert saved_path.exists()

        # Now test loading
        mock_features_load = MagicMock()
        mock_features_load.artifact_to_load = True
        mock_features_load.features = [mock_feature]

        loaded = EmbeddingArtifact.load_embedding_artifact(mock_features_load, artifact_key)

        assert loaded is not None
        assert loaded["embeddings"] == embeddings
        assert loaded["model_name"] == "test-model"
        assert loaded["embedding_dim"] == 3

    def test_artifact_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that loading a non-existent artifact raises ValueError."""
        mock_feature = MagicMock()
        mock_feature.get_name.return_value = "text__embedded"
        mock_feature.options = MagicMock()
        mock_feature.options.get.side_effect = lambda key: str(tmp_path) if key == "artifact_storage_path" else None

        mock_features = MagicMock()
        mock_features.features = [mock_feature]
        mock_features.artifact_to_load = True
        mock_features.name_of_one_feature = mock_feature

        with pytest.raises(ValueError, match="Embedding artifact not found"):
            EmbeddingArtifact.load_embedding_artifact(mock_features, "nonexistent_key")

    def test_check_artifact_exists(self, tmp_path: Path) -> None:
        """Test checking if artifact exists."""
        mock_feature = MagicMock()
        mock_feature.options = MagicMock()
        mock_feature.options.get.side_effect = lambda key: str(tmp_path) if key == "artifact_storage_path" else None

        mock_features = MagicMock()
        mock_features.features = [mock_feature]
        mock_features.artifact_to_load = True
        mock_features.artifact_to_save = True
        mock_features.save_artifact = {}
        mock_features.name_of_one_feature = mock_feature

        artifact_key = "test_artifact_key"

        # Initially should not exist
        assert not EmbeddingArtifact.check_artifact_exists(mock_features, artifact_key)

        # Save an artifact
        artifact_data = {
            "embeddings": [[0.1, 0.2]],
            "model_name": "test",
            "embedding_dim": 2,
        }
        EmbeddingArtifact.save_embedding_artifact(mock_features, artifact_key, artifact_data)
        EmbeddingArtifact.custom_saver(mock_features, mock_features.save_artifact)

        # Now should exist
        assert EmbeddingArtifact.check_artifact_exists(mock_features, artifact_key)

    def test_artifact_contains_metadata(self, tmp_path: Path) -> None:
        """Test that artifact contains correct metadata."""
        mock_feature = MagicMock()
        mock_feature.options = MagicMock()
        mock_feature.options.get.side_effect = lambda key: str(tmp_path) if key == "artifact_storage_path" else None

        mock_features = MagicMock()
        mock_features.features = [mock_feature]
        mock_features.artifact_to_save = True
        mock_features.artifact_to_load = True
        mock_features.save_artifact = {}
        mock_features.name_of_one_feature = mock_feature

        artifact_key = "metadata_test_key"
        artifact_data = {
            "embeddings": [[0.1, 0.2, 0.3, 0.4]],
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dim": 4,
            "num_texts": 1,
            "custom_metadata": "test_value",
        }

        EmbeddingArtifact.save_embedding_artifact(mock_features, artifact_key, artifact_data)
        EmbeddingArtifact.custom_saver(mock_features, mock_features.save_artifact)

        loaded = EmbeddingArtifact.load_embedding_artifact(mock_features, artifact_key)

        assert loaded is not None
        assert loaded["model_name"] == "all-MiniLM-L6-v2"
        assert loaded["embedding_dim"] == 4
        assert loaded["num_texts"] == 1
        assert loaded["custom_metadata"] == "test_value"
        assert loaded["artifact_key"] == artifact_key

    def test_no_load_when_artifact_to_load_false(self, tmp_path: Path) -> None:
        """Test that no loading occurs when artifact_to_load is False."""
        mock_features = MagicMock()
        mock_features.artifact_to_load = False

        result = EmbeddingArtifact.load_embedding_artifact(mock_features, "any_key")
        assert result is None

    def test_no_save_when_artifact_to_save_false(self, tmp_path: Path) -> None:
        """Test that no saving occurs when artifact_to_save is False."""
        mock_features = MagicMock()
        mock_features.artifact_to_save = False
        mock_features.save_artifact = None

        # This should not raise and should not modify save_artifact
        EmbeddingArtifact.save_embedding_artifact(
            mock_features,
            "test_key",
            {"embeddings": [[0.1]]}
        )

        # save_artifact should remain None since artifact_to_save is False
        assert mock_features.save_artifact is None

    def test_custom_saver_requires_dict(self, tmp_path: Path) -> None:
        """Test that custom_saver requires a dictionary artifact."""
        mock_features = MagicMock()

        with pytest.raises(ValueError, match="Expected artifact to be a dictionary"):
            EmbeddingArtifact.custom_saver(mock_features, "not a dict")

    def test_multiple_artifacts_in_same_directory(self, tmp_path: Path) -> None:
        """Test saving and loading multiple artifacts."""
        mock_feature = MagicMock()
        mock_feature.options = MagicMock()
        mock_feature.options.get.side_effect = lambda key: str(tmp_path) if key == "artifact_storage_path" else None

        mock_features = MagicMock()
        mock_features.features = [mock_feature]
        mock_features.artifact_to_save = True
        mock_features.artifact_to_load = True
        mock_features.save_artifact = {}
        mock_features.name_of_one_feature = mock_feature

        # Save first artifact
        key1 = "feature1_model1"
        data1 = {"embeddings": [[1.0, 2.0]], "model_name": "model1", "embedding_dim": 2}
        EmbeddingArtifact.save_embedding_artifact(mock_features, key1, data1)

        # Save second artifact
        key2 = "feature2_model2"
        data2 = {"embeddings": [[3.0, 4.0, 5.0]], "model_name": "model2", "embedding_dim": 3}
        EmbeddingArtifact.save_embedding_artifact(mock_features, key2, data2)

        # Save all artifacts
        EmbeddingArtifact.custom_saver(mock_features, mock_features.save_artifact)

        # Load and verify both
        loaded1 = EmbeddingArtifact.load_embedding_artifact(mock_features, key1)
        loaded2 = EmbeddingArtifact.load_embedding_artifact(mock_features, key2)

        assert loaded1["embeddings"] == [[1.0, 2.0]]
        assert loaded1["model_name"] == "model1"
        assert loaded2["embeddings"] == [[3.0, 4.0, 5.0]]
        assert loaded2["model_name"] == "model2"
