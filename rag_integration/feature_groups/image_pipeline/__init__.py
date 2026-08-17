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
    "BaseImageDeduplicator",
    "BaseImageEmbedder",
    "BaseImagePIIRedactor",
    "BaseImagePreprocessor",
    "BaseImageSource",
    "BlurPIIRedactor",
    "CLIPImageEmbedder",
    "DictImageSource",
    "DifferenceHashImageDeduplicator",
    "ExactHashImageDeduplicator",
    "FileImageSource",
    "HashImageEmbedder",
    "MockImageEmbedder",
    "NormalizePreprocessor",
    "PerceptualHashImageDeduplicator",
    "PixelPIIRedactor",
    "ResizePreprocessor",
    "SolidFillPIIRedactor",
    "ThumbnailPreprocessor",
]
