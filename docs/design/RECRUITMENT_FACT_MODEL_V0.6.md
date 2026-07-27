# Recruitment Fact Model v0.6

Document status: `0.6-design-draft`

## 1. Principles

`AssessmentFacts 3.0.0` records observable conduct, reported decision rights,
contract wording, dates, locations, controls, and metadata-only artefacts. It
does not ask users to decide who determines “essential means” or to declare a
provider, deployer, controller, processor, or joint controller as a raw fact.

The model is:

- explicit-unknown preserving;
- multi-actor, multi-system, multi-workflow, and multi-operation;
- stable-ID based;
- deterministic and JSON serializable;
- provenance-bearing;
- independent of presentation language;
- backward compatible with v0.5 serialized facts.

Use `TriState` for legal-relevant yes/no questions. Use `None` for an unknown
scalar or collection; an empty list means the collection was assessed and no
items were reported.

## 2. Information layers

| Layer | Representation | Rule use |
| --- | --- | --- |
| observable facts | actors, systems, workflows, operations, relationships, decision rights, process, controls | canonical rule input |
| contractual statements | `contractual_allocations` | separate evidence of wording |
| user legal characterizations | `user_characterizations` | displayed opinion; never a substitute for facts |
| system role hypotheses | result-layer `RoleHypothesis` | informational |
| formal legal output | scoped `Finding` | authorized Evidence-ready rule only |
| organizational artefacts | `compliance_artefacts` | metadata about case material, not Legal Evidence |

Contradictions between layers are preserved and traced.

## 3. Proposed AssessmentFacts structure

```text
AssessmentFacts 3.0.0
├── existing v0.5 namespaces
├── temporal_context: TemporalContextFacts
├── territorial_context: TerritorialContextFacts
├── actors: list[ActorFacts] | None
├── ai_systems: list[AISystemFacts] | None
├── recruitment_workflows: list[RecruitmentWorkflowFacts] | None
├── processing_operations: list[ProcessingOperationFacts] | None
├── actor_relationships: list[ActorRelationshipFacts] | None
├── contractual_allocations: list[ContractualAllocationFacts] | None
├── decision_rights: list[DecisionRightFacts] | None
├── recruitment_processes: list[RecruitmentDecisionProcessFacts] | None
├── recruitment_controls: list[RecruitmentControlFacts] | None
├── compliance_artefacts: list[ComplianceArtefactFacts] | None
├── user_characterizations: list[UserLegalCharacterization] | None
└── fact_metadata: stable record/field provenance
```

The canonical scope tuple is:

```text
(actor_id, system_id, workflow_id, processing_operation_id)
```

Every role hypothesis, invocation, Finding, artefact requirement, and gap uses
the applicable tuple. One actor may have different GDPR roles for different
processing operations and different AI Act roles for different systems or
uses.

## 4. Identity lifecycle

### 4.1 Stable IDs

- IDs are opaque, immutable, case-local, and never translated.
- Editing a record preserves its ID.
- A deleted ID is retired and not silently reused.
- Merge and split operations create explicit supersession references.
- Imports preserve IDs only when namespace and collision checks pass.
- Fingerprints use stable IDs plus normalized content, not list indexes.

Recommended records include `created_at`, `updated_at`, `record_version`, and
optional `supersedes_ids`. These are record-history metadata, not legal facts.

### 4.2 Referential integrity

Serialization rejects duplicate IDs and unknown references. Deleting an actor,
system, workflow, or operation requires explicit dependent-record handling and
invalidates every transitive output for that scope.

## 5. Actor facts

`ActorFacts` contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `actor_id` | stable string | case-local identity |
| `display_name` | string or `None` | reported name |
| `actor_kind` | enum | employer, recruiter, headhunter, AI vendor, data supplier, applicant, other, unknown |
| `legal_form` | enum or `None` | descriptive form |
| `establishment_locations` | list or `None` | countries/regions reported |
| `acts_in_own_name` | `TriState` | observable presentation |
| `operates_system_ids` | list or `None` | actual operation |
| `develops_or_commissions_system_ids` | list or `None` | development conduct |
| `markets_system_ids_under_own_name` | list or `None` | branding conduct |
| `uses_system_ids_in_own_organisation` | list or `None` | own-organization use |
| `uses_system_ids_on_behalf_of_actor_ids` | mapping or `None` | reported operational arrangement |

`actor_kind` and operational fields do not encode a regulatory role.

## 6. AI system facts

`AISystemFacts` contains:

- stable `system_id`;
- name, description, lifecycle status, intended purpose, outputs, and autonomy;
- developer/vendor actor IDs;
- commissioning and branding actor IDs;
- selected-by actor IDs;
- configured-by and modified-by actor IDs;
- model/vendor/version references;
- put-into-service date where known;
- system-use locations;
- instructions-for-use metadata reference;
- provenance and record version.

Two AI systems in one recruitment case remain separate even when they support
one workflow.

## 7. Recruitment workflow facts

`RecruitmentWorkflowFacts` contains:

- stable `workflow_id`;
- title and recruitment objective;
- employer/client actor IDs;
- recruiter actor IDs;
- system IDs;
- candidate population description;
- recruitment stages;
- output recipients;
- final-decision actor IDs;
- intended-use date;
- system-use and output-use locations;
- affected-person locations;
- associated processing-operation IDs;
- provenance and record version.

A workflow describes operational sequence. It does not imply that every data
use within it is one GDPR processing operation.

## 8. Processing-operation facts

`ProcessingOperationFacts` contains:

- stable `processing_operation_id`;
- workflow ID and relevant system IDs;
- participating actor IDs;
- reported purpose;
- candidate population;
- data categories and sources;
- collection, use, disclosure, retention, and deletion activities;
- recipients;
- actor acting within documented instructions;
- actor acting outside documented instructions;
- independent reuse purpose;
- operation start/end dates where known;
- territorial context;
- affected-person locations;
- provenance and record version.

Client recruitment screening and independent talent-pool reuse are separate
operations. A reuse flag may help routing, but cannot substitute for creating
the second operation.

## 9. Observable relationship facts

`ActorRelationshipFacts` is a directed edge with:

- stable `relationship_id`;
- `from_actor_id` and `to_actor_id`;
- system, workflow, and processing-operation scope;
- descriptive relationship type such as instructs, contracts with, supplies
  data to, receives output from, reviews for, or decides for;
- reported presence;
- scope and frequency;
- supporting artefact IDs;
- conflicting relationship IDs;
- provenance and record version.

Relationship edges do not ask whether an actor determines essential means or
has legally confirmed practical control.

## 10. Concrete decision-right and conduct facts

`DecisionRightFacts` records one concrete subject, scoped to system, workflow,
and processing operation:

| Decision or conduct | Unknown-preserving fields |
| --- | --- |
| chooses recruitment objective | actor IDs; reported practice; contract wording |
| chooses candidate population | actor IDs; reported practice; contract wording |
| chooses data categories and sources | actor IDs; reported practice; contract wording |
| chooses screening criteria | actor IDs; reported practice; contract wording |
| sets weights or thresholds | actor IDs; reported practice; contract wording |
| selects AI system or vendor | actor IDs; approval/veto facts |
| configures or modifies system | actor IDs; scope; approval facts |
| chooses output recipients | actor IDs; reported practice |
| chooses retention period | actor IDs; reported practice; contract wording |
| designs human review | actor IDs; reported practice |
| holds approval, veto, or override authority | actor IDs; authority type; actual use |
| acts within documented instructions | actor IDs; `TriState`; scope |
| acts outside documented instructions | actor IDs; `TriState`; scope |

Each record keeps contract wording and reported practice in separate fields.
Rules project legal significance from combinations of these concrete facts.

## 11. Contractual statements and user characterizations

### 11.1 ContractualAllocationFacts

- stable allocation ID and version;
- actor parties;
- system/workflow/operation scope;
- agreement or instruction artefact reference;
- responsibility topic;
- stated responsible actor;
- instruction scope;
- permitted/prohibited uses;
- effective dates;
- provenance.

### 11.2 UserLegalCharacterization

- framework;
- scoped actor/system/workflow/operation references;
- asserted role or conclusion;
- author type;
- rationale;
- timestamp.

It remains a user-supplied opinion and cannot satisfy a rule's observable-fact
requirement.

## 12. Recruitment decision-process facts

One `RecruitmentDecisionProcessFacts` record is scoped by workflow, operation,
and system.

### 12.1 Functions and effects

- employment or worker-management purpose;
- sourcing, screening, ranking, filtering, recommendation, or exclusion;
- automatic removal from consideration;
- output materially influences decision;
- decision is solely automated;
- legal effect;
- similarly significant effect;
- profiling relevance;
- affected persons and locations.

### 12.2 Human review

- reviewer actor IDs;
- substantive basis reviewed;
- formal authority to change result;
- realistic time and information;
- ability to disregard output;
- ability to restore an excluded person;
- actual override behavior;
- AI output routinely followed;
- escalation available.

Human presence alone does not determine meaningful human involvement.

### 12.3 Screening criteria

Each `ScreeningCriterionFacts` contains:

- stable criterion ID;
- category: sex_or_gender, education_institution,
  first_degree_background, work_experience, age_indicator, other, unknown;
- description;
- source and selecting/configuring actor IDs;
- ranking/filtering/recommendation/exclusion use;
- weight or threshold facts;
- `gdpr_special_category_data`: `TriState`;
- `employment_equality_protected_characteristic`: `TriState`;
- `proxy_for_protected_characteristic`: `TriState`;
- `classification_review_state`: reviewed, not_reviewed, unknown;
- provenance.

Sex/gender, age, education, first-degree background, and proxies are not
automatically classified as GDPR Article 9 data, protected characteristics
under a particular applicable law, or unlawful criteria.

### 12.4 Data sources

Each source records supplier/recipient actors, direct/third-party origin, data
categories, operation IDs, verification, retention, deletion, and provenance.

## 13. Controls

`RecruitmentControlFacts` uses `TriState` and is scoped by actor/system/workflow/
operation. It covers:

- applicant notice;
- AI-assisted-processing information;
- contestation or appeal;
- human intervention;
- override and escalation;
- event and decision logging;
- log access and retention;
- monitoring responsibility;
- performance, drift, or bias monitoring;
- incident handling;
- instructions for use;
- vendor documentation;
- configuration/change history;
- validation and bias testing;
- data retention and deletion.

These facts do not establish substantive compliance.

## 14. Compliance artefact metadata

`ComplianceArtefactFacts` contains:

- stable artefact ID and type;
- title and custodian actor;
- scoped actors, systems, workflows, and operations;
- document version, date, and scope;
- supplied and review status;
- content assessment only if a future authorized human review records it;
- fact references;
- opaque `file_reference`;
- provenance.

For minimum v0.6, artefacts are metadata-only. There is no file upload,
document ingestion, extraction, hashing, storage, or persistence. The opaque
reference must not resolve or fetch content.

Legal Evidence is never stored in this namespace.

## 15. Temporal and territorial facts

### 15.1 TemporalContextFacts

- assessment date;
- intended-use date by workflow;
- put-into-service date by system;
- operation start/end dates;
- applicable legal-source snapshot or legal baseline ID.

### 15.2 TerritorialContextFacts

- actor establishment locations;
- system-use locations;
- AI-output-use locations;
- affected-person locations;
- processing-operation territorial context.

The minimum release collects these facts but does not implement temporal or
territorial applicability law.

## 16. Provenance

Canonical paths use stable IDs, for example:

```text
actors[actor:employer].operates_system_ids
ai_systems[system:ranker].put_into_service_date
recruitment_workflows[workflow:client-hiring].final_decision_actor_ids
processing_operations[operation:talent-pool].reported_purpose
decision_rights[right:criteria-selection].reported_practice_actor_ids
```

Mutable list indexes are never durable trace references.

## 17. Contradictions and validation

Validation detects without resolving:

- duplicate or dangling IDs;
- contradictory contract and practice;
- within-instructions and outside-instructions both reported for the same
  scoped conduct;
- decision makers or reviewers outside the scope;
- automatic exclusion paired with incompatible process descriptions;
- missing operation separation for reported independent reuse;
- artefact references to unknown records.

Contradictions produce structured consistency issues and normally cause the
affected invocation to report `missing_facts` or a rule-supported
`undetermined` Finding. No source layer silently overrides another.

## 18. RoleHypothesis

An informational `RoleHypothesis` contains:

- stable hypothesis ID;
- framework and candidate role;
- actor/system/workflow/processing-operation scope;
- status: supported, contradicted, unresolved, or not assessed;
- supporting, conflicting, and missing fact paths;
- projection version;
- explanation key and provisional disclaimer.

One actor may have multiple hypotheses across frameworks, systems, uses, and
operations. Only a dedicated Evidence-ready rule invocation may emit a formal
role Finding.

## 19. Serialization and migration

- `AssessmentFacts.schema_version` is `3.0.0`.
- New booleans default to `unknown`.
- New repeated collections default to `None` when unassessed.
- v0.5 facts load unchanged with unknown/empty v0.6 namespaces.
- A derived legacy view may create display-only default identities, but cannot
  write manufactured facts back to the case.
- The canonical fingerprint includes identity versions, normalized
  execution-relevant facts, and the facts schema version.

## 20. Public deployment restrictions

The public UI must warn against and prevent upload/storage paths for real
applicant data, CVs, names, contacts, special-category data, confidential or
privileged material, trade secrets, and identifiable production records.
Minimum v0.6 accepts fictional or safely anonymized structured facts only and
metadata-only artefact references.
