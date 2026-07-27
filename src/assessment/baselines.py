"""Version and validity baselines for future v0.6 assessment runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.assessment.models import SerializableModel


@dataclass(frozen=True, slots=True, order=True)
class RuleBaselineEntry(SerializableModel):
    rule_id: str
    rule_version: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip() or not self.rule_version.strip():
            raise ValueError("rule baseline identity and version are required")


@dataclass(frozen=True, slots=True, order=True)
class EvidencePackBaseline(SerializableModel):
    instrument_id: str
    pack_version: str | None = None
    manifest_hash: str | None = None
    source_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, str) or not self.instrument_id.strip():
            raise ValueError("instrument_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class AssessmentBaseline(SerializableModel):
    """Complete version surface; unavailable values remain explicitly ``None``."""

    baseline_version: str = "1.0.0"
    facts_schema_version: str = "3.0.0"
    engine_version: str = "3.0.0"
    questionnaire_version: str = "3.0.0"
    report_schema_version: str = "2.0.0"
    assessment_context_version: str = "1.0.0"
    execution_record_version: str = "1.0.0"
    ordered_rules: tuple[RuleBaselineEntry, ...] = ()
    rule_dependency_graph_hash: str | None = None
    ruleset_baseline_id: str | None = None
    evidence_packs: tuple[EvidencePackBaseline, ...] = ()
    legal_source_baseline_id: str | None = None

    def __post_init__(self) -> None:
        if self.baseline_version != "1.0.0":
            raise ValueError("unsupported AssessmentBaseline baseline_version")
        for field_name in (
            "facts_schema_version",
            "engine_version",
            "questionnaire_version",
            "report_schema_version",
            "assessment_context_version",
            "execution_record_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        ordered_rules = tuple(
            item
            if isinstance(item, RuleBaselineEntry)
            else RuleBaselineEntry.from_dict(item)
            for item in self.ordered_rules
        )
        object.__setattr__(self, "ordered_rules", ordered_rules)
        if len({entry.rule_id for entry in self.ordered_rules}) != len(
            self.ordered_rules
        ):
            raise ValueError("ordered_rules must not contain duplicate rule IDs")
        evidence_packs = tuple(
            item
            if isinstance(item, EvidencePackBaseline)
            else EvidencePackBaseline.from_dict(item)
            for item in self.evidence_packs
        )
        if len({item.instrument_id for item in evidence_packs}) != len(
            evidence_packs
        ):
            raise ValueError(
                "evidence_packs must not contain duplicate instrument IDs"
            )
        object.__setattr__(
            self,
            "evidence_packs",
            tuple(sorted(evidence_packs, key=lambda item: item.instrument_id)),
        )
