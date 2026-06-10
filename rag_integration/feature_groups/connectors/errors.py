"""Typed errors shared by the connector families.

All subclass ``ValueError`` (via :class:`ConnectorError`), so callers and
contract tests that catch ``ValueError`` keep working. Messages stay at the
raise site, where the per-family wording differs.
"""

from __future__ import annotations


class ConnectorError(ValueError):
    """Base for every connector-family validation / rejection error."""


class MissingOptionError(ConnectorError):
    """A required option is absent."""


class InvalidOptionError(ConnectorError):
    """An option has the wrong type or an unusable value."""


class DuplicateDocIdError(ConnectorError):
    """Two entries share an effective ``doc_id``."""


class RankingContractError(ConnectorError):
    """A backend ``_rank`` result violates the ranking contract."""


class GroundingError(ConnectorError):
    """An answer cites or surfaces something not in the supplied input."""


class SqlSafetyError(ConnectorError):
    """Backend SQL is unsafe or not a single bare ``SELECT``."""
