"""Image Pipeline feature groups with provider inheritance pattern."""

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
    ResizePreprocessor,
    NormalizePreprocessor,
    ThumbnailPreprocessor,
)
from rag_integration.feature_groups.image_pipeline.deduplication import (
    BaseImageDeduplicator,
    ExactHashImageDeduplicator,
    PerceptualHashImageDeduplicator,
    DifferenceHashImageDeduplicator,
)
from rag_integration.feature_groups.image_pipeline.embedding import (
    BaseImageEmbedder,
    MockImageEmbedder,
    HashImageEmbedder,
    CLIPImageEmbedder,
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
