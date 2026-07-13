"""Run the recruitment AI fixture through the v2 assessment workflow."""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assessment import (  # noqa: E402
    AssessmentFacts,
    AssessmentReport,
    TriState,
)
from src.assessment.demo import create_assessment_workflow  # noqa: E402
from src.assessment.workflow import AssessmentWorkflowService  # noqa: E402
from src.assessment.facts import (  # noqa: E402
    AffectedPerson,
    AnnexIIIArea,
    DegreeOfAutonomy,
    FactMetadata,
    FactSource,
    LifecycleStatus,
    SystemOutput,
    UseDomain,
)
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "recruitment_ai_case.json"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store.json"


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """Load the pure-data demonstration fixture."""

    with path.open("r", encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), dict):
        raise ValueError("fixture must contain a facts object")
    return payload


def build_assessment_facts(data: dict[str, Any]) -> AssessmentFacts:
    """Hydrate the fixture's AssessmentFacts-shaped JSON payload."""

    facts = AssessmentFacts(schema_version=data["schema_version"])

    case_data = data["case"]
    facts.case.title = case_data["title"]
    facts.case.reference = case_data["reference"]
    facts.case.assessment_date = date.fromisoformat(case_data["assessment_date"])
    facts.case.notes = case_data["notes"]

    system_data = data["system"]
    facts.system.name = system_data["name"]
    facts.system.description = system_data["description"]
    facts.system.lifecycle_status = LifecycleStatus(system_data["lifecycle_status"])
    facts.system.intended_purpose = system_data["intended_purpose"]
    facts.system.outputs = [SystemOutput(value) for value in system_data["outputs"]]
    facts.system.machine_based_inference = TriState(
        system_data["machine_based_inference"]
    )
    facts.system.degree_of_autonomy = DegreeOfAutonomy(
        system_data["degree_of_autonomy"]
    )
    facts.system.adaptiveness_after_deployment = TriState(
        system_data["adaptiveness_after_deployment"]
    )

    _assign_tri_state_values(facts.scope, data["scope"])
    facts.organisation.name = data["organisation"]["name"]
    facts.organisation.establishment_countries = list(
        data["organisation"]["establishment_countries"]
    )
    _assign_tri_state_values(facts.supply_chain, data["supply_chain"])

    use_context_data = data["use_context"]
    facts.use_context.domain = UseDomain(use_context_data["domain"])
    facts.use_context.task = use_context_data["task"]
    facts.use_context.affected_persons = [
        AffectedPerson(value) for value in use_context_data["affected_persons"]
    ]
    facts.use_context.materially_influences_decision = TriState(
        use_context_data["materially_influences_decision"]
    )
    facts.use_context.human_review_before_effect = TriState(
        use_context_data["human_review_before_effect"]
    )
    facts.use_context.profiles_natural_persons = TriState(
        use_context_data["profiles_natural_persons"]
    )

    _assign_tri_state_values(facts.practices, data["practices"])

    high_risk_data = data["high_risk"]
    for fact_path in (
        "is_safety_component_or_product",
        "product_covered_by_annex_i",
        "requires_third_party_conformity_assessment",
        "narrow_procedural_task",
        "improves_completed_human_activity",
        "detects_patterns_without_replacing_or_influencing_human_assessment",
        "preparatory_task",
    ):
        setattr(facts.high_risk, fact_path, TriState(high_risk_data[fact_path]))
    facts.high_risk.annex_iii_area = AnnexIIIArea(
        high_risk_data["annex_iii_area"]
    )
    facts.high_risk.annex_iii_use_case = high_risk_data["annex_iii_use_case"]

    for section_name in ("data_protection", "data_act"):
        section_data = data.get(section_name)
        if section_data is not None:
            _assign_tri_state_values(getattr(facts, section_name), section_data)

    facts.fact_metadata = {
        fact_path: FactMetadata(
            source=FactSource(metadata["source"]),
            question_id=metadata["question_id"],
            recorded_at=datetime.fromisoformat(metadata["recorded_at"]),
        )
        for fact_path, metadata in data["fact_metadata"].items()
    }
    return facts


def _assign_tri_state_values(target: object, values: dict[str, str]) -> None:
    for field_name, value in values.items():
        if not hasattr(target, field_name):
            raise ValueError(f"unknown tri-state fact field: {field_name}")
        setattr(target, field_name, TriState(value))


def build_workflow(
    payload: dict[str, Any],
) -> tuple[AssessmentWorkflowService, str, str]:
    """Assemble existing services for one fixture-backed demonstration case."""

    facts = build_assessment_facts(payload["facts"])
    bundle = create_assessment_workflow(vector_store_path=VECTOR_STORE_PATH)
    assessment_case = bundle.case_service.create_case(
        payload["scenario"]["name"],
        description=payload["scenario"]["description"],
        facts=facts,
        case_id=payload["scenario_id"],
    )
    return bundle.workflow, assessment_case.case_id, assessment_case.name


def validate_expected_result(
    report: AssessmentReport,
    expected: dict[str, Any],
) -> None:
    """Fail clearly if the fixture no longer produces its documented outcome."""

    if not report.findings:
        raise RuntimeError("demo fixture produced no finding")
    finding = report.findings[0]
    actual = {
        "rule_id": finding.rule_id,
        "finding_category": finding.category.value,
        "finding_status": finding.status.value,
        "requires_legal_review": finding.requires_legal_review,
        "legal_basis": [basis.citation for basis in finding.legal_basis],
    }
    if actual != expected:
        raise RuntimeError(
            "demo result does not match fixture expectation:\n"
            f"expected={expected!r}\nactual={actual!r}"
        )


def print_report_summary(case_name: str, report: AssessmentReport) -> None:
    """Print a concise human-readable assessment summary."""

    print(f"Case: {case_name}")
    print("Findings:")
    if not report.findings:
        print("  None")
    for finding in report.findings:
        print(f"  - [{finding.status.value}] {finding.title}")
        citations = ", ".join(basis.citation for basis in finding.legal_basis)
        print(f"    Legal basis: {citations or 'None'}")

    print(f"Evidence count: {len(report.evidence)}")
    print("Missing information:")
    if not report.missing_information:
        print("  None")
    for item in report.missing_information:
        print(f"  - {item.fact_path} ({item.reason.value})")


def main() -> None:
    payload = load_fixture()
    workflow, case_id, case_name = build_workflow(payload)
    report = workflow.run(case_id)
    validate_expected_result(report, payload["expected_assessment"])
    print_report_summary(case_name, report)


if __name__ == "__main__":
    main()
