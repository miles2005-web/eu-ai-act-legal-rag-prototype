"""Deterministic UI-layer normalization for controlled legal fact phrases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re

from src.assessment.facts import AssessmentFacts
from src.assessment.models import SerializableModel, TriState


class NormalizationStatus(str, Enum):
    """Confidence state for deterministic input normalization."""

    EMPTY = "empty"
    PASSTHROUGH = "passthrough"
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class PhraseMapping:
    mapping_id: str
    phrases: tuple[str, ...]
    canonical_task: str | None = None
    fact_updates: tuple[tuple[str, TriState], ...] = ()


@dataclass(slots=True)
class NormalizationResult(SerializableModel):
    """Auditable normalization result retaining the original text."""

    original_text: str
    status: NormalizationStatus
    canonical_task: str | None = None
    fact_updates: dict[str, TriState] = field(default_factory=dict)
    mapping_ids: list[str] = field(default_factory=list)


_MAPPINGS = (
    PhraseMapping(
        "employment.recruitment_screening.v1",
        (
            "招聘筛选",
            "简历筛选",
            "候选人筛选",
            "recruitment screening",
            "cv screening",
            "candidate screening",
        ),
        canonical_task="recruitment screening of candidates",
    ),
    PhraseMapping(
        "employment.candidate_ranking.v1",
        (
            "候选人排序",
            "求职者排名",
            "candidate ranking",
            "applicant ranking",
        ),
        canonical_task="candidate ranking",
    ),
    PhraseMapping(
        "employment.material_influence.v1",
        (
            "对录用或面试决定产生实质性影响",
            "实质影响招聘决定",
            "materially influences employment decisions",
            "materially influences interview decisions",
        ),
        fact_updates=(("use_context.materially_influences_decision", TriState.YES),),
    ),
    PhraseMapping(
        "data_act.connected_product.v1",
        (
            "互联产品",
            "联网设备",
            "联网机械",
            "connected product",
            "connected equipment",
            "connected machinery",
        ),
        fact_updates=(("data_act.connected_product", TriState.YES),),
    ),
    PhraseMapping(
        "data_act.related_service.v1",
        (
            "相关服务",
            "配套数字服务",
            "related service",
            "related digital service",
        ),
        fact_updates=(("data_act.related_service", TriState.YES),),
    ),
    PhraseMapping(
        "data_act.data_generated.v1",
        (
            "设备生成数据",
            "产品运行数据",
            "服务生成数据",
            "equipment-generated data",
            "product operational data",
            "service-generated data",
        ),
        fact_updates=(("data_act.data_generated", TriState.YES),),
    ),
    PhraseMapping(
        "gdpr.personal_data.v1",
        (
            "处理个人数据",
            "涉及个人数据",
            "processes personal data",
            "personal data involved",
        ),
        fact_updates=(("data_protection.personal_data_processed", TriState.YES),),
    ),
    PhraseMapping(
        "gdpr.solely_automated_decision.v1",
        (
            "完全自动化决定",
            "仅由自动化处理作出决定",
            "solely automated decision",
            "decision based solely on automated processing",
        ),
        fact_updates=(
            ("data_protection.automated_individual_decision", TriState.YES),
        ),
    ),
    PhraseMapping(
        "gdpr.significant_effect.v1",
        (
            "产生法律效果",
            "产生类似重大影响",
            "produces legal effects",
            "similarly significant effect",
        ),
        fact_updates=(("use_context.materially_influences_decision", TriState.YES),),
    ),
)

_CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]")
_NEGATION_PATTERN = re.compile(
    r"(?:不|未|无|否认|不涉及|不是|not|does\s+not|no\s+)",
    re.IGNORECASE,
)


def normalize_legal_input(text: str | None) -> NormalizationResult:
    """Map controlled phrases without probabilistic inference."""

    original = "" if text is None else str(text)
    normalized = " ".join(original.casefold().split())
    if not normalized:
        return NormalizationResult(
            original_text=original,
            status=NormalizationStatus.EMPTY,
        )

    matched = [
        mapping
        for mapping in _MAPPINGS
        if any(phrase.casefold() in normalized for phrase in mapping.phrases)
    ]
    if matched and _NEGATION_PATTERN.search(normalized):
        return NormalizationResult(
            original_text=original,
            status=NormalizationStatus.AMBIGUOUS,
        )
    if not matched:
        status = (
            NormalizationStatus.AMBIGUOUS
            if _CHINESE_PATTERN.search(original)
            else NormalizationStatus.PASSTHROUGH
        )
        return NormalizationResult(
            original_text=original,
            status=status,
            canonical_task=(
                original.strip()
                if status is NormalizationStatus.PASSTHROUGH
                else None
            ),
        )

    tasks: list[str] = []
    updates: dict[str, TriState] = {}
    mapping_ids: list[str] = []
    for mapping in matched:
        mapping_ids.append(mapping.mapping_id)
        if mapping.canonical_task and mapping.canonical_task not in tasks:
            tasks.append(mapping.canonical_task)
        for fact_path, value in mapping.fact_updates:
            updates[fact_path] = value
    return NormalizationResult(
        original_text=original,
        status=NormalizationStatus.MATCHED,
        canonical_task="; ".join(tasks) or None,
        fact_updates=updates,
        mapping_ids=mapping_ids,
    )


def apply_normalized_input(
    facts: AssessmentFacts,
    result: NormalizationResult,
    *,
    ambiguous_text_confirmed: bool = False,
    protected_fact_paths: frozenset[str] = frozenset(),
) -> None:
    """Apply only deterministic or explicitly confirmed UI input to facts."""

    if not isinstance(facts, AssessmentFacts):
        raise TypeError("facts must be an AssessmentFacts instance")
    if not isinstance(result, NormalizationResult):
        raise TypeError("result must be a NormalizationResult")

    if result.status in (
        NormalizationStatus.MATCHED,
        NormalizationStatus.PASSTHROUGH,
    ):
        facts.use_context.task = result.canonical_task
    elif result.status is NormalizationStatus.AMBIGUOUS:
        facts.use_context.task = (
            result.original_text.strip() if ambiguous_text_confirmed else None
        )
    else:
        facts.use_context.task = None

    if result.status is not NormalizationStatus.MATCHED:
        return
    for fact_path, value in result.fact_updates.items():
        if fact_path in protected_fact_paths:
            continue
        target: object = facts
        segments = fact_path.split(".")
        for segment in segments[:-1]:
            target = getattr(target, segment)
        setattr(target, segments[-1], value)
