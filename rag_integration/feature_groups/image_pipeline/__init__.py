"""Image Pipeline feature groups with provider inheritance pattern."""

from rag_integration.feature_groups.image_pipeline.deduplication import (
    BaseImageDeduplicator,
    DifferenceHashImageDeduplicator,
    ExactHashImageDeduplicator,
    PerceptualHashImageDeduplicator,
)
from rag_integration.feature_groups.image_pipeline.embedding import (
    BaseImageEmbedder,
    CLIPImageEmbedder,
    HashImageEmbedder,
    MockImageEmbedder,
)
from rag_integration.feature_groups.image_pipeline.image_source import (
    BaseImageSource,
    DictImageSource,
    FileImageSource,
)
from rag_integration.feature_groups.image_pipeline.pii_redaction import (
    BaseImagePIIRedactor,
    BlurPIIRedactor,
    PixelPIIRedactor,
    SolidFillPIIRedactor,
)
from rag_integration.feature_groups.image_pipeline.preprocessing import (
    BaseImagePreprocessor,
    NormalizePreprocessor,
    ResizePreprocessor,
    ThumbnailPreprocessor,
)

__all__ = [
    # Image Source
    "BaseImageSource",
    "DictImageSource",
    "FileImageSource",
    # PII Redaction
    "BaseImagePIIRedactor",
    "BlurPIIRedactor",
    "PixelPIIRedactor",
    "SolidFillPIIRedactor",
    # Preprocessing
    "BaseImagePreprocessor",
    "ResizePreprocessor",
    "NormalizePreprocessor",
    "ThumbnailPreprocessor",
    # Deduplication
    "BaseImageDeduplicator",
    "ExactHashImageDeduplicator",
    "PerceptualHashImageDeduplicator",
    "DifferenceHashImageDeduplicator",
    # Embedding
    "BaseImageEmbedder",
    "MockImageEmbedder",
    "HashImageEmbedder",
    "CLIPImageEmbedder",
]
