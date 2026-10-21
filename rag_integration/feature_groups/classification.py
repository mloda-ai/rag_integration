"""Data classification declarations, read by the mloda-registry OTel, OpenLineage and audit extenders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from mloda.provider import FeatureSet

CLASSIFICATION_KEY = "mloda.data.classification"


class PassThroughClassification:
    """Declares ``public``: the step adds no sensitivity, so its outputs keep their inputs' most restrictive level.

    An undeclared step would get the extender's ``undeclared`` level instead, undoing an upstream redactor.
    """

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {CLASSIFICATION_KEY: "public"}


class RedactedClassification:
    """Masking redactor: outputs get exactly ``REDACTED_CLASSIFICATION``, whatever level their inputs carried.

    ``confidential`` rather than lower, since redaction is best effort and removes PII only. Subclass to change
    the level, or set ``masking = False`` to stop lowering it.
    """

    masking: ClassVar[bool] = True
    REDACTED_CLASSIFICATION: ClassVar[str] = "confidential"

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> Mapping[str, str | int | float | bool]:
        return {CLASSIFICATION_KEY: cls.REDACTED_CLASSIFICATION}
