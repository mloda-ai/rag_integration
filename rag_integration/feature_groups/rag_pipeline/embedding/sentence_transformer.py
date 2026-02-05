"""Sentence Transformer embeddings using the sentence-transformers library."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Type

if TYPE_CHECKING:
    from mloda.user import Feature

from mloda.provider import BaseArtifact
from mloda_plugins.feature_group.experimental.default_options_key import DefaultOptionKeys

from rag_integration.feature_groups.rag_pipeline.embedding.base import BaseEmbedder
from rag_integration.feature_groups.rag_pipeline.embedding.embedding_artifact import EmbeddingArtifact


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence Transformer embedder using the sentence-transformers library.

    Generates high-quality semantic embeddings using pre-trained transformer models.
    Supports various models from Hugging Face's sentence-transformers.

    Requires: pip install sentence-transformers

    Popular Models:
        - all-MiniLM-L6-v2: Fast, good quality (384 dimensions)
        - all-mpnet-base-v2: Best quality (768 dimensions)
        - paraphrase-multilingual-MiniLM-L12-v2: Multilingual support

    Configuration:
        embedding_dim: Ignored (determined by model)
        model_name: Sentence transformer model (default: "all-MiniLM-L6-v2")

    Config-based matching:
        embedding_method="sentence_transformer"
    """

    PROPERTY_MAPPING = {
        BaseEmbedder.EMBEDDING_METHOD: {
            "sentence_transformer": "Sentence Transformer semantic embeddings",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.strict_validation: True,
        },
        BaseEmbedder.EMBEDDING_DIM: {
            "explanation": "Dimension of embeddings (determined by model, this is informational)",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: 384,
        },
        BaseEmbedder.MODEL_NAME: {
            "explanation": "Sentence transformer model name from Hugging Face",
            DefaultOptionKeys.context: True,
            DefaultOptionKeys.default: "all-MiniLM-L6-v2",
        },
        DefaultOptionKeys.in_features: {
            "explanation": "Source feature containing text to embed",
            DefaultOptionKeys.context: True,
        },
    }

    # Default model for sentence transformers
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    _model: Optional[object] = None
    _model_name: Optional[str] = None

    @staticmethod
    def artifact() -> Optional[Type[BaseArtifact]]:
        """Return EmbeddingArtifact for persisting sentence transformer embeddings."""
        return EmbeddingArtifact

    @classmethod
    def _get_model_name(cls, feature: "Feature") -> str:
        """Get model name from feature options, defaulting to all-MiniLM-L6-v2."""
        name = feature.options.get(cls.MODEL_NAME)
        return str(name) if name is not None else cls.DEFAULT_MODEL

    @classmethod
    def _get_model(cls, model_name: str) -> object:
        """Get or create the sentence transformer model."""
        if cls._model is None or cls._model_name != model_name:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerEmbedder. "
                    "Install with: pip install sentence-transformers"
                ) from e

            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name
        return cls._model

    @classmethod
    def _embed_texts(
        cls,
        texts: List[str],
        embedding_dim: int,
        model_name: str,
    ) -> List[List[float]]:
        """
        Generate embeddings using Sentence Transformers.

        Args:
            texts: List of text strings to embed
            embedding_dim: Ignored (determined by model)
            model_name: Sentence transformer model name

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = cls._get_model(model_name)

        # Handle empty strings
        processed_texts = [t if t.strip() else " " for t in texts]

        # Generate embeddings
        embeddings = model.encode(  # type: ignore[attr-defined]
            processed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Return unit-normalized vectors
        )

        # Convert to list of lists
        return [embedding.tolist() for embedding in embeddings]
