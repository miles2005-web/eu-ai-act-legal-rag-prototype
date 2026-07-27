# Recruitment Compliance Consequence Chain v0.6

Document status: `0.6-design-draft`

## 1. Purpose and boundary

This specification defines a design-only extension of the released
`v0.5.0-prototype`:

```text
structured facts
  -> scoped actors, systems, workflows, and processing operations
  -> provisional role hypotheses
  -> applicability screening
  -> obligation relevance
  -> deterministic artefact requirements
  -> informational unresolved gaps
  -> Legal Evidence bindings
  -> canonical report
```

It does not implement rules or claim complete AI Act, GDPR, employment, or
equality-law coverage. Formal assessment remains deterministic. No LLM
participates in fact normalization, authorization, rule execution, Finding
generation, Evidence selection, or canonical reporting.

## 2. Reference scenario

An employer instructs an external recruiter to use one or more AI systems to
screen applicants. The recruiter may follow client instructions for a
client-specific recruitment operation while separately reusing data for its
own talent pool. Screening may use sex or gender, education, first-degree
background, experience, age-related indicators, or other criteria.

Contract labels are not conclusive. The assessment records concrete
decision-right and conduct facts. Regulatory roles are projected separately.

## 3. Preserved legal distinctions

- Actor identity is not regulatory-role classification.
- One actor may hold different GDPR roles for different processing operations
  and multiple AI Act roles for different systems or uses.
- Applicability is not substantive compliance.
- Obligation relevance is not proof of performance.
- Human presence is not necessarily meaningful review.
- Contract wording does not override reported practice.
- Legal Evidence supports a legal proposition; a compliance artefact supports
  a factual claim about an organization.
- Artefact absence is not automatically non-compliance.
- An AI output does not become the legally accountable person or organization.
- A module-level negative Finding does not establish absence of all legal risk.

All Findings remain preliminary and subject to appropriate legal and factual
review.

## 4. Reuse and required extension

### 4.1 Reused responsibilities

| Current component | v0.6 responsibility |
| --- | --- |
| `AssessmentCase` | Holds current mutable facts. |
| `AssessmentRun` | Preserves an immutable facts and authorization snapshot. |
| `TriState` | Preserves `yes`, `no`, and `unknown`. |
| `AssessmentRule` | Remains the deterministic rule interface, extended with dependency metadata. |
| `RuleRegistry` | Registers rules and validates the dependency graph. |
| requirement validation | Prevents missing facts from producing substantive Findings. |
| questionnaire router | Preserves explicit module confirmation and deterministic routing. |
| Evidence service and retriever | Bind exact legal bases to reviewed Evidence without an LLM. |
| report builder | Produces deterministic, traceable output. |
| bilingual presentation | Translates display labels without mutating canonical objects. |

### 4.2 Required extensions

| Area | v0.6 extension |
| --- | --- |
| identity | stable `actor_id`, `system_id`, `workflow_id`, and `processing_operation_id` |
| facts | repeated actors, AI systems, recruitment workflows, processing operations, concrete decision rights, controls, and artefact metadata |
| execution | immutable `AssessmentContext`, scoped `RuleInvocation`, dependency DAG, and `RuleExecutionRecord` |
| authorization | scoped `AuthorizedRuleInvocation`, while preserving `authorized_rule_ids` for legacy readers |
| Finding | explicit subject scope; one invocation produces at most one substantive Finding |
| invalidation | transitive invalidation across facts, authorization, rule graph, Evidence baseline, and reports |
| report | actors, operations, role hypotheses, formal Findings, execution records, artefact requirements, unresolved gaps, Evidence, and scope limitations |
| Evidence | reviewed AI Act and GDPR recruitment metadata-v2 packs |

## 5. Scope identity model

The canonical subject scope is the applicable combination of:

- `actor_id`;
- `system_id`;
- `workflow_id`;
- `processing_operation_id`.

Every role hypothesis, `RuleInvocation`, Finding, artefact requirement, gap,
missing-fact record, and Evidence trace must preserve that scope. A missing
dimension is explicit, not inferred from display text.

The client-screening operation and the recruiter's independent talent-pool reuse
are separate `ProcessingOperationFacts` records. They may share actors, a
workflow, data sources, or a system, but never share a role conclusion merely
because they occur in one case.

Details are defined in
[RECRUITMENT_FACT_MODEL_V0.6.md](RECRUITMENT_FACT_MODEL_V0.6.md).

## 6. AssessmentContext contract

v0.6 adopts `AssessmentContext` version `1.0.0`. It is read-only, immutable,
serializable, deterministic, and created by the engine for one invocation.

It contains only:

- immutable `AssessmentFacts` snapshot;
- current `RuleInvocation` scope;
- explicitly declared prerequisite Finding summaries;
- authorized execution scope;
- ruleset baseline;
- Legal Evidence baseline.

It must not contain:

- mutable services;
- UI or session state;
- translated display text;
- arbitrary access to undeclared Findings;
- callbacks, hooks, or references capable of mutating facts or authorization.

Prerequisite summaries expose only stable Finding identity, subject scope,
framework, rule ID/version, status, reason codes, and declared trace references.
They do not expose presentation text as rule input.

The rejected alternative is to keep a facts-only engine and repeat every
upstream predicate in every downstream rule. That approach preserves the old
signature but duplicates legal logic, obscures dependency invalidation, and
weakens traceability. v0.6 therefore changes the rule input contract to
`evaluate(context)`.

## 7. Rule metadata and dependency DAG

Every v0.6 rule declares machine-readable:

- `phase`;
- `depends_on`;
- `accepted_upstream_statuses`;
- `subject_selector`;
- deterministic `ordering_key`.

`depends_on` identifies rule IDs and required subject-scope relationships.
`accepted_upstream_statuses` is declared per dependency. The subject selector
expands canonical facts into deterministic invocation scopes.

Registration must reject:

- an undeclared or missing dependency;
- a dependency on an incompatible framework or scope;
- a cycle, including an indirect cycle;
- duplicate ordering keys within an ambiguous phase/scope;
- a rule whose selector cannot produce stable scope identities.

After validation, the registry produces a deterministic topological order.
Tie-breaking uses phase, ordering key, rule ID, rule version, and canonical
subject scope. Registration order must not change execution.

### 7.1 Execution phases

1. `screening`;
2. `role_relevance`;
3. `obligation_relevance`;
4. `artefact_projection`.

The minimum formal chain is:

```text
AI_ACT_HIGH_RISK_EMPLOYMENT
  -> AI_ACT_DEPLOYER_ROLE_RELEVANCE
  -> AI_ACT_DEPLOYER_USE_INSTRUCTIONS_RELEVANCE
  -> AI_ACT_DEPLOYER_HUMAN_OVERSIGHT_RELEVANCE
  -> deterministic artefact requirements
  -> informational unresolved gaps
```

The existing `GDPR_ARTICLE22_RELEVANCE / 2026.1` remains independent and
unchanged for v0.5 regression compatibility.

## 8. RuleInvocation and cardinality

`RuleInvocation` contains:

- invocation ID derived deterministically from rule identity and scope;
- rule ID and version;
- phase and ordering key;
- actor IDs;
- system IDs;
- workflow IDs;
- processing-operation IDs;
- declared prerequisite invocation/Finding references;
- accepted upstream statuses;
- authorization reference.

The cardinality rule is:

> One `RuleInvocation`, for one subject and one scoped operation, produces at
> most one substantive Finding.

Subject expansion creates separate invocations for separate actors, systems,
workflows, or processing operations. A multi-actor role rule cannot produce one
composite Finding covering unrelated actors or operations.

## 9. Scoped execution authorization

`authorized_rule_ids` remains serialized for v0.5 compatibility, but v0.6
authorization is canonicalized as `AuthorizedRuleInvocation`:

- rule ID;
- rule version;
- subject actor IDs or a deterministic subject selector;
- system IDs;
- workflow IDs;
- processing-operation IDs;
- authorization source;
- authorization timestamp.

The `AssessmentRun` preserves the expanded invocation list exactly. A rule ID
in the legacy list does not authorize every actor or operation. The engine
executes only the intersection of validated subject expansion and explicit
authorization.

Changing authorization scope invalidates affected invocations and all
transitive downstream state.

## 10. Execution state versus Finding status

Each planned invocation produces one `RuleExecutionRecord` version `1.0.0`
with one of:

- `completed`;
- `not_authorized`;
- `blocked_by_dependency`;
- `blocked_by_evidence`;
- `missing_facts`;
- `failed`.

The record contains invocation scope, rule identity, dependency references,
missing facts or failure metadata, baseline IDs, start/end timestamps, and an
optional resulting Finding ID.

Unauthorized, unsupported, Evidence-blocked, missing-fact, and failed paths do
not produce authoritative-looking Findings. `not_assessed` remains readable
only for backward compatibility. New v0.6 rules do not emit it as a
substantive Finding.

## 11. Formal and informational scope

### 11.1 Minimum formal rules

Existing unchanged:

- `AI_ACT_HIGH_RISK_EMPLOYMENT / 2026.1`;
- `GDPR_ARTICLE22_RELEVANCE / 2026.1`.

New candidates, only after Evidence approval:

- `AI_ACT_DEPLOYER_ROLE_RELEVANCE`;
- `AI_ACT_DEPLOYER_USE_INSTRUCTIONS_RELEVANCE`;
- `AI_ACT_DEPLOYER_HUMAN_OVERSIGHT_RELEVANCE`.

### 11.2 Informational-only v0.6 outputs

- GDPR controller-like hypothesis;
- GDPR processor-like hypothesis;
- joint-control analysis required;
- independent-reuse operation detected;
- actor and relationship summaries;
- supplied artefact inventory;
- unresolved gaps that are not supported by a dedicated gap rule;
- translated explanations.

### 11.3 Deferred formal rules

- `GDPR_ACTOR_ROLE_RELEVANCE`;
- `GDPR_ARTICLE22_SAFEGUARD_RELEVANCE`;
- `GDPR_DPIA_RELEVANCE`;
- AI Act logging/monitoring;
- AI Act affected-person information;
- FRIA;
- document-content sufficiency;
- final role classification.

The detailed rule design is in
[RECRUITMENT_RULE_MATRIX_V0.6.md](RECRUITMENT_RULE_MATRIX_V0.6.md).

## 12. Article 22 compatibility

`GDPR_ARTICLE22_RELEVANCE / 2026.1` retains its existing facts, predicate, and
output. Material influence remains part of that legacy preliminary trigger
only.

A future refined version must separately evaluate:

- personal-data processing;
- an individual decision;
- solely automated processing;
- legal effect;
- similarly significant effect;
- meaningful human involvement indicators;
- profiling.

Material influence must not substitute for that complete predicate. The
refined rule is outside the minimum v0.6 release unless its Evidence and truth
table receive separate approval.

## 13. Temporal and territorial context

Facts include assessment date, intended-use date, put-into-service date, actor
establishments, system-use locations, output-use locations, affected-person
locations, operation territorial context, and a legal-source baseline ID.

v0.6 does not implement temporal or territorial law. If those modules are
outside the authorized rule scope or required facts are incomplete, the report
must display:

> Territorial and temporal applicability not assessed.

The message is a scope limitation, not a Finding.

## 14. Questionnaire and invalidation

### 14.1 Stages

1. scenario plus actor/system/workflow/operation discovery;
2. instructions, contracts, concrete decision rights, and conduct;
3. decision process, automation, effects, and review;
4. subject-scoped role questions;
5. obligation controls and artefact metadata;
6. scoped module authorization.

Repeated records use stable IDs. Localized labels never become canonical IDs.

### 14.2 Transitive invalidation

Changing an actor, system, workflow, processing operation, relationship,
decision right, automation fact, authorization scope, rule graph, rule version,
or Evidence baseline invalidates:

- affected role hypotheses;
- affected invocations and execution records;
- every transitive downstream Finding;
- artefact requirements and unresolved gaps;
- Evidence bindings;
- selected-Finding presentation state;
- report validity.

Unrelated scopes and selected language remain intact.

## 15. Compliance artefacts and public deployment

Artefact mapping is deterministic and triggered by formal Findings. It does not
declare non-compliance. Presence, absence, review status, and content
sufficiency remain distinct.

For the minimum v0.6 public Streamlit deployment:

- artefacts are metadata-only records;
- `file_reference` is an opaque user-supplied reference;
- there is no upload, document ingestion, content extraction, or persistence.

The public deployment must visibly warn users and must not accept or store:

- real applicant personal data;
- CVs;
- names or contact details;
- special-category data;
- client-confidential information;
- privileged legal material;
- trade secrets;
- identifiable production assessment records.

UI acceptance must verify that upload paths are absent or rejected and that the
warning appears before fact entry.

## 16. Canonical report

Report version `2.0.0` contains:

1. run, baseline, fingerprint, and scope metadata;
2. territorial/temporal limitation;
3. actors, systems, workflows, and processing operations;
4. concrete relationship and decision-right facts;
5. informational role hypotheses;
6. formal applicability and obligation Findings;
7. `RuleExecutionRecord` entries, visually separate from Findings;
8. missing facts;
9. deterministic artefact requirements;
10. supplied metadata-only artefacts;
11. informational unresolved gaps;
12. per-Finding Legal Evidence Trace;
13. excluded, unauthorized, and deferred modules;
14. scope limitations.

Formal outputs are authorized Findings, missing-fact requirements,
RuleExecutionRecords, rule-derived artefact requirements with reviewed legal
bases, and exact Evidence bindings. Hypotheses, narrative explanations,
artefact inventory, and unsupported-module explanations are informational.

## 17. Fingerprint and baselines

The report-validity fingerprint deterministically includes:

- facts schema version;
- normalized facts affecting execution;
- expanded `RuleInvocation` scope;
- engine version;
- questionnaire version;
- ordered rule IDs and individual rule versions;
- rule dependency graph hash;
- rule behavior/ruleset baseline ID;
- Evidence-pack versions;
- manifest hashes;
- legal-source baseline ID;
- report schema version.

The ruleset baseline stores the ordered rules, versions, phases, dependency
edges, accepted upstream statuses, selector versions, ordering keys, and
behavior baseline ID. The Legal Evidence baseline stores pack versions,
manifest hashes, source snapshot IDs, and legal-source baseline ID.

Any change to an identity, scoped fact, authorization, dependency graph, rule
version/behavior, Evidence pack, manifest, or legal baseline invalidates the
affected report.

## 18. Minimum v0.6 implementation boundary

The minimum release includes:

1. schema `3.0.0` identities and facts;
2. scoped authorization and invocation expansion;
3. immutable `AssessmentContext 1.0.0`;
4. dependency-DAG validation and engine `3.0.0`;
5. `RuleExecutionRecord 1.0.0`;
6. existing rules unchanged;
7. the three new AI Act candidates only after Evidence approval;
8. deterministic artefact requirements and informational gaps;
9. reviewed AI Act and GDPR recruitment Evidence packs;
10. report `2.0.0`, questionnaire `3.0.0`, bilingual invariance, public-data
    restrictions, and acceptance regressions.

If a new candidate is not Evidence-ready, it remains registered only as
unavailable metadata or is omitted from the registry; execution records show
`blocked_by_evidence`, and no substantive Finding is emitted.

## 19. Migration and versioning

| Contract | Final v0.6 version |
| --- | --- |
| `AssessmentFacts` | `3.0.0` |
| questionnaire | `3.0.0` |
| engine | `3.0.0` |
| `AssessmentContext` | `1.0.0` |
| report | `2.0.0` |
| `RuleExecutionRecord` | `1.0.0` |

Existing rule versions remain `2026.1` unless their logic changes. Existing
v0.5 facts load with unknown/empty v0.6 namespaces. Existing v0.5 reports remain
immutable snapshots and must not be presented as v0.6 consequence reports.
Readers may support both report versions; no historical snapshot is rewritten.

## 20. Evidence priority and implementation sequence

1. Approve legal propositions and build reviewed atomic Evidence packs.
2. Implement schema v3 and backward-compatible serialization.
3. Implement stable identities and repeated-record provenance.
4. Implement authorization, invocation expansion, DAG validation, and context.
5. Implement routing and transitive invalidation.
6. Add informational hypotheses.
7. Implement one Evidence-approved AI Act rule at a time.
8. Implement artefact projection and execution-state presentation.
9. Implement report v2 and bilingual UI.
10. Run clean-checkout, baseline, invalidation, privacy, and acceptance tests.

Evidence gaps are detailed in
[RECRUITMENT_EVIDENCE_GAP_ANALYSIS_V0.6.md](RECRUITMENT_EVIDENCE_GAP_ANALYSIS_V0.6.md).

## 21. Related documents

- [Fact model](RECRUITMENT_FACT_MODEL_V0.6.md)
- [Rule matrix](RECRUITMENT_RULE_MATRIX_V0.6.md)
- [Evidence gap analysis](RECRUITMENT_EVIDENCE_GAP_ANALYSIS_V0.6.md)
- [Acceptance matrix](RECRUITMENT_ACCEPTANCE_MATRIX_V0.6.md)
