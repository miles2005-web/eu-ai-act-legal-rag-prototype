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
        "ai_act.product_safety_component.v1",
        (
            "ai safety component",
            "product safety component",
            "safety component",
            "安全部件人工智能",
            "ai 安全部件",
            "安全部件",
        ),
    ),
    PhraseMapping(
        "ai_act.regulated_ai_product.v1",
        (
            "ai itself is a regulated product",
            "ai system is a regulated product",
            "ai 本身属于受监管产品",
            "ai 系统本身是受监管产品",
        ),
    ),
    PhraseMapping(
        "ai_act.product_safety_context.v1",
        (
            "industrial product safety",
            "machinery product safety",
            "industrial robot",
            "safety function",
            "emergency stop",
            "automatic stop",
            "trigger stop",
            "protective control",
            "工业产品安全",
            "机械产品安全",
            "工业机器人",
            "安全功能",
            "人员受伤",
            "紧急制动",
            "自动停止",
            "触发停止",
            "安全控制",
            "产品安全",
        ),
    ),
    PhraseMapping(
        "ai_act.medical_device_context.v1",
        (
            "medical device ai",
            "ai medical device",
            "医疗器械 ai",
            "ai 医疗器械",
        ),
    ),
    PhraseMapping(
        "ai_act.regulated_equipment_context.v1",
        (
            "regulated equipment",
            "regulated industrial equipment",
            "受监管设备",
            "受监管工业设备",
        ),
    ),
    PhraseMapping(
        "ai_act.conformity_assessment.v1",
        (
            "third-party conformity assessment",
            "independent conformity assessment",
            "conformity assessment",
            "third party assessment",
            "independent third party",
            "第三方合格评定",
            "独立合格评定",
            "合格评定",
            "独立第三方",
        ),
    ),
    PhraseMapping(
        "gdpr.personal_data.v1",
        (
            "处理个人数据",
            "涉及个人数据",
            "个人财务与信贷分析",
            "processes personal data",
            "personal data involved",
            "personal financial and credit analysis",
        ),
        fact_updates=(("data_protection.personal_data_processed", TriState.YES),),
    ),
    PhraseMapping(
        "decision.credit.v1",
        (
            "自动贷款批准或拒绝",
            "自动批准或拒绝贷款",
            "automated loan approval/rejection",
            "automated loan approval or rejection",
        ),
        canonical_task="automated consumer credit decision",
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
            "法律效果或重大经济影响",
            "produces legal effects",
            "similarly significant effect",
            "legal or significant economic effect",
        ),
        fact_updates=(("use_context.materially_influences_decision", TriState.YES),),
    ),
)

_CHINESE_PATTERN = re.compile(r"[\u3400-\u9fff]")
_NEGATION_PATTERN = re.compile(
    r"(?:不|未|无|否认|不涉及|不是|not|does\s+not|no\s+)",
    re.IGNORECASE,
)


def _normalize_matching_text(value: str) -> str:
    """Normalize punctuation and spacing without fuzzy or semantic inference."""

    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", value.casefold())
    return " ".join(normalized.split())


def _contains_controlled_phrase(normalized_text: str, phrase: str) -> bool:
    """Match one authored phrase, tolerating punctuation and Chinese spacing."""

    normalized_phrase = _normalize_matching_text(phrase)
    if not normalized_phrase:
        return False
    if normalized_phrase in normalized_text:
        return True
    if _CHINESE_PATTERN.search(normalized_phrase):
        return normalized_phrase.replace(" ", "") in normalized_text.replace(" ", "")
    return False


def normalize_legal_input(text: str | None) -> NormalizationResult:
    """Map controlled phrases without probabilistic inference."""

    original = "" if text is None else str(text)
    normalized = _normalize_matching_text(original)
    if not normalized:
        return NormalizationResult(
            original_text=original,
            status=NormalizationStatus.EMPTY,
        )

    matched = [
        mapping
        for mapping in _MAPPINGS
        if any(
            _contains_controlled_phrase(normalized, phrase)
            for phrase in mapping.phrases
        )
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
