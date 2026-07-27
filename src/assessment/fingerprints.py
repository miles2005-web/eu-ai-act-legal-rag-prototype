"""Canonical v0.6 fingerprint inputs kept inactive in the v0.5 workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
from hashlib import sha256
import json

from src.assessment.baselines import AssessmentBaseline
from src.assessment.invocations import RuleInvocation
from src.assessment.models import SerializableModel, freeze_value, to_primitive


_TOP_LEVEL_IDENTITIES = {
    "actors": "actor_id",
    "ai_systems": "system_id",
    "recruitment_workflows": "workflow_id",
    "processing_operations": "processing_operation_id",
    "recruitment_processes": "process_id",
    "compliance_artefacts": "artefact_id",
}
_SET_LIKE_REFERENCE_FIELDS = frozenset(
    {
        "operates_system_ids",
        "develops_or_commissions_system_ids",
        "markets_system_ids_under_own_name",
        "uses_system_ids_in_own_organisation",
        "uses_system_ids_on_behalf_of_actor_ids",
        "vendor_actor_ids",
        "commissioning_actor_ids",
        "branding_actor_ids",
        "selected_by_actor_ids",
        "configured_by_actor_ids",
        "employer_actor_ids",
        "recruiter_actor_ids",
        "system_ids",
        "output_recipient_actor_ids",
        "final_decision_actor_ids",
        "processing_operation_ids",
        "participating_actor_ids",
        "recipients",
        "selecting_actor_ids",
        "configuring_actor_ids",
        "retired_entity_ids",
        "supersedes_ids",
    }
)
_UNORDERED_VALUE_FIELDS = frozenset(
    {
        "affected_persons",
        "affected_person_locations",
        "actor_establishment_locations",
        "data_categories",
        "data_sources",
        "establishment_countries",
        "establishment_locations",
        "output_use_locations",
        "outputs",
        "processing_operation_context",
        "system_use_locations",
        "territorial_context",
    }
)
# The order of these fields is part of the asserted process narrative.
_ORDERED_VALUE_FIELDS = frozenset({"recruitment_stages"})


def canonicalize_v3_facts(facts: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize only semantically unordered v3 fact collections."""

    payload = deepcopy(dict(facts))
    if payload.get("schema_version") != "3.0.0":
        raise ValueError("V06FingerprintInput requires AssessmentFacts 3.0.0")

    def normalize(value: object, *, path: tuple[str, ...] = ()) -> object:
        if isinstance(value, Mapping):
            result = {
                str(key): normalize(item, path=path + (str(key),))
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
            criteria = result.get("screening_criteria")
            if isinstance(criteria, list):
                identities = [str(item["criterion_id"]) for item in criteria]
                if len(set(identities)) != len(identities):
                    raise ValueError(
                        "screening_criteria must not contain duplicate criterion IDs"
                    )
                result["screening_criteria"] = sorted(
                    criteria, key=lambda item: str(item["criterion_id"])
                )
            return result
        if isinstance(value, (list, tuple)):
            normalized = [
                normalize(item, path=path + ("[]",)) for item in value
            ]
            containing_fields = set(path)
            if containing_fields.intersection(_ORDERED_VALUE_FIELDS):
                return normalized
            unordered = bool(
                containing_fields.intersection(_SET_LIKE_REFERENCE_FIELDS)
                or containing_fields.intersection(_UNORDERED_VALUE_FIELDS)
            )
            if unordered:
                serialized = [
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    for item in normalized
                ]
                if len(set(serialized)) != len(serialized):
                    raise ValueError(
                        f"{'.'.join(path)} must not contain duplicate values"
                    )
                return sorted(
                    normalized,
                    key=lambda item: json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                )
            return normalized
        return deepcopy(value)

    canonical = normalize(payload)
    assert isinstance(canonical, dict)
    for field_name, identity_name in _TOP_LEVEL_IDENTITIES.items():
        collection = canonical.get(field_name)
        if isinstance(collection, list):
            identities = [str(item[identity_name]) for item in collection]
            if len(set(identities)) != len(identities):
                raise ValueError(
                    f"{field_name} must not contain duplicate stable identities"
                )
            canonical[field_name] = sorted(
                collection,
                key=lambda item: str(item[identity_name]),
            )
    return canonical


@dataclass(frozen=True, slots=True)
class V06FingerprintInput(SerializableModel):
    """Inactive v0.6 input; later phases supply graph and Evidence hashes."""

    facts: Mapping[str, object]
    invocations: tuple[RuleInvocation, ...]
    baseline: AssessmentBaseline
    contract_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.contract_version != "1.0.0":
            raise ValueError("unsupported V06FingerprintInput contract_version")
        canonical_facts = canonicalize_v3_facts(self.facts)
        if not isinstance(self.invocations, (list, tuple)):
            raise TypeError("invocations must be a list or tuple")
        normalized_invocations: list[RuleInvocation] = []
        by_id: dict[str, dict] = {}
        for item in self.invocations:
            if isinstance(item, dict):
                invocation = RuleInvocation.from_dict(item)
            elif isinstance(item, RuleInvocation):
                invocation = RuleInvocation.from_dict(item.to_dict())
            else:
                raise TypeError(
                    "invocations must contain RuleInvocation objects or dictionaries"
                )
            identifier = str(invocation.invocation_id)
            payload = invocation.to_dict()
            if identifier in by_id:
                if by_id[identifier] != payload:
                    raise ValueError(
                        "conflicting invocation definitions share one invocation_id"
                    )
                raise ValueError("duplicate invocation_id")
            by_id[identifier] = payload
            normalized_invocations.append(invocation)
        if isinstance(self.baseline, dict):
            baseline = AssessmentBaseline.from_dict(self.baseline)
        elif isinstance(self.baseline, AssessmentBaseline):
            baseline = AssessmentBaseline.from_dict(self.baseline.to_dict())
        else:
            raise TypeError(
                "baseline must be an AssessmentBaseline or dictionary"
            )
        if canonical_facts["schema_version"] != baseline.facts_schema_version:
            raise ValueError(
                "facts schema_version does not match baseline"
            )
        for invocation in normalized_invocations:
            matching_rules = [
                item
                for item in baseline.ordered_rules
                if item.rule_id == invocation.rule_id
            ]
            if len(matching_rules) != 1:
                raise ValueError(
                    "invocation rule must appear exactly once in "
                    "baseline.ordered_rules"
                )
            if matching_rules[0].rule_version != invocation.rule_version:
                raise ValueError(
                    "invocation rule_version does not match "
                    "baseline.ordered_rules"
                )
        object.__setattr__(self, "facts", freeze_value(canonical_facts))
        object.__setattr__(self, "invocations", tuple(normalized_invocations))
        object.__setattr__(self, "baseline", baseline)

    def canonical_payload(self) -> dict:
        invocations = sorted(
            self.invocations,
            key=lambda item: (
                item.rule_id,
                item.rule_version,
                tuple(value or "" for value in item.scope.canonical_tuple()),
                item.invocation_id,
            ),
        )
        return {
            "contract_version": self.contract_version,
            "facts": to_primitive(self.facts),
            "invocations": [item.to_dict() for item in invocations],
            "baseline": self.baseline.to_dict(),
        }

    def digest(self) -> str:
        canonical = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()
