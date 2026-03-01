"""Tests for VectorStoreArtifact."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import faiss
import numpy as np
import pytest

from rag_integration.feature_groups.rag_pipeline.vector_store.vector_store_artifact import VectorStoreArtifact


class TestVectorStoreArtifact:
    """Tests for the VectorStoreArtifact class."""

    def _make_mock_features(self, tmp_path: Path) -> MagicMock:
        """Create mock features with artifact_storage_path pointing to tmp_path."""
        mock_feature = MagicMock()
        mock_feature.get_name.return_value = "test__indexed"
        mock_feature.options = MagicMock()
        mock_feature.options.get.side_effect = lambda key: str(tmp_path) if key == "artifact_storage_path" else None

        mock_features = MagicMock()
        mock_features.features = [mock_feature]
        mock_features.artifact_to_save = True
        mock_features.artifact_to_load = False
        mock_features.save_artifact = {}
        mock_features.name_of_one_feature = mock_feature
        return mock_features

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test that a FAISS index can be saved and loaded correctly."""
        mock_features = self._make_mock_features(tmp_path)

        # Build a small FAISS index
        dimension = 4
        index = faiss.IndexFlatL2(dimension)
        vectors = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        index.add(vectors)

        artifact_key = "test__indexed"
        texts = ["first document", "second document"]
        doc_ids = ["doc_1", "doc_2"]

        # Save
        VectorStoreArtifact.save_vector_store_artifact(mock_features, artifact_key, index, texts, doc_ids)
        VectorStoreArtifact.custom_saver(mock_features, mock_features.save_artifact)

        # Verify files exist
        faiss_files = list(tmp_path.glob("vector_store_*.faiss"))
        json_files = list(tmp_path.glob("vector_store_*_metadata.json"))
        assert len(faiss_files) == 1
        assert len(json_files) == 1

        # Load
        mock_features.artifact_to_load = True
        loaded = VectorStoreArtifact.load_vector_store_artifact(mock_features, artifact_key)

        assert loaded is not None
        assert loaded["index"].ntotal == 2
        assert loaded["metadata"]["texts"] == texts
        assert loaded["metadata"]["doc_ids"] == doc_ids

    def test_metadata_sidecar_content(self, tmp_path: Path) -> None:
        """Test that the JSON metadata sidecar contains correct data."""
        mock_features = self._make_mock_features(tmp_path)

        dimension = 3
        index = faiss.IndexFlatL2(dimension)
        vectors = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        index.add(vectors)

        artifact_key = "meta_test"
        VectorStoreArtifact.save_vector_store_artifact(mock_features, artifact_key, index, ["hello world"], ["doc_42"])
        VectorStoreArtifact.custom_saver(mock_features, mock_features.save_artifact)

        # Read the JSON sidecar directly
        json_files = list(tmp_path.glob("vector_store_*_metadata.json"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            metadata = json.load(f)

        assert metadata["artifact_key"] == artifact_key
        assert metadata["texts"] == ["hello world"]
        assert metadata["doc_ids"] == ["doc_42"]
        assert metadata["num_vectors"] == 1

    def test_missing_artifact_raises_error(self, tmp_path: Path) -> None:
        """Test that loading a non-existent artifact raises ValueError."""
        mock_features = self._make_mock_features(tmp_path)
        mock_features.artifact_to_load = True

        with pytest.raises(ValueError, match="Vector store artifact not found"):
            VectorStoreArtifact.load_vector_store_artifact(mock_features, "nonexistent_key")

    def test_custom_saver_requires_dict(self) -> None:
        """Test that custom_saver requires a dictionary artifact."""
        mock_features = MagicMock()

        with pytest.raises(ValueError, match="Expected artifact to be a dictionary"):
            VectorStoreArtifact.custom_saver(mock_features, "not a dict")

    def test_check_artifact_exists(self, tmp_path: Path) -> None:
        """Test checking if artifact exists."""
        mock_features = self._make_mock_features(tmp_path)
        mock_features.artifact_to_load = True

        artifact_key = "exist_check"
        assert not VectorStoreArtifact.check_artifact_exists(mock_features, artifact_key)

        # Save an artifact
        index = faiss.IndexFlatL2(2)
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        index.add(vectors)
        VectorStoreArtifact.save_vector_store_artifact(mock_features, artifact_key, index, ["text"], ["doc_1"])
        VectorStoreArtifact.custom_saver(mock_features, mock_features.save_artifact)

        assert VectorStoreArtifact.check_artifact_exists(mock_features, artifact_key)
