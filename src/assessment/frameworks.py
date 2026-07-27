"""Regulation-neutral identifiers for supported assessment frameworks."""

from __future__ import annotations

from enum import Enum


class RegulatoryFramework(str, Enum):
    """Stable regulatory framework identifiers used across domain models."""

    EU_AI_ACT = "EU_AI_ACT"
    GDPR = "GDPR"
    EU_DATA_ACT = "EU_DATA_ACT"
    UNKNOWN = "UNKNOWN"
