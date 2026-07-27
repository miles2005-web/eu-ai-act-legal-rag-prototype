# Recruitment Rule Matrix v0.6

Document status: `0.6-design-draft`

## 1. Rule contract

v0.6 rules receive an immutable `AssessmentContext 1.0.0`. Every rule declares:

| Metadata | Purpose |
| --- | --- |
| `phase` | screening, role relevance, obligation relevance, or artefact projection |
| `depends_on` | prerequisite rule IDs and scope relationship |
| `accepted_upstream_statuses` | statuses that permit downstream execution |
| `subject_selector` | deterministic actor/system/workflow/operation expansion |
| `ordering_key` | deterministic order within the dependency DAG |

Registration performs missing-dependency detection, cycle detection, scope
compatibility validation, and deterministic topological ordering.

One `RuleInvocation`, for one subject and one scoped operation, produces at
most one substantive Finding. Separate actors, systems, workflows, or
processing operations receive separate invocations.

## 2. Execution record and Finding status

Every planned invocation receives a `RuleExecutionRecord 1.0.0`:

- `completed`;
- `not_authorized`;
- `blocked_by_dependency`;
- `blocked_by_evidence`;
- `missing_facts`;
- `failed`.

Only a completed invocation may reference a substantive Finding. Unsupported,
unauthorized, missing-fact, Evidence-blocked, and failed paths remain execution
records. New rules do not use `not_assessed` as a substantive Finding status;
legacy values remain readable.

Substantive Findings use narrow `potentially_applies`, `does_not_apply`, or
`undetermined` conclusions. `applies` requires an independently approved rule
design and is not proposed for the new minimum v0.6 rules.

## 3. Scoped authorization

An `AuthorizedRuleInvocation` identifies rule/version, actor IDs or subject
selector, system IDs, workflow IDs, processing-operation IDs, authorization
source, and timestamp. Legacy `authorized_rule_ids` remains serialized but
does not authorize all subjects.

The engine expands selectors deterministically, intersects the result with
authorization, constructs a scoped invocation, and exposes only explicitly
declared upstream summaries in `AssessmentContext`.

## 4. Existing unchanged rules

### 4.1 AI_ACT_HIGH_RISK_EMPLOYMENT / 2026.1

- **Phase:** `screening`.
- **Framework:** EU AI Act.
- **Purpose:** preserve the released preliminary employment high-risk screen.
- **Current required facts:** use domain, task, and material influence.
- **Subject expansion:** one invocation for each authorized
  system/workflow/operation employment use; a legacy case receives its existing
  compatibility scope.
- **Statuses:** current behavior unchanged.
- **Interaction:** accepted positive result may authorize the AI Act deployer
  chain for the same scope.
- **Negative limit:** no conclusion about other AI Act high-risk routes.
- **Evidence gate for logic change:** atomic Article 2, Article 6(2), relevant
  Article 6(3), and Annex III point 4(a).

No v0.6 schema addition silently changes this rule's `2026.1` predicate.

### 4.2 GDPR_ARTICLE22_RELEVANCE / 2026.1

- **Phase:** `screening`.
- **Framework:** GDPR.
- **Purpose:** preserve the released preliminary Article 22 relevance trigger.
- **Current facts and logic:** unchanged for v0.5 regression.
- **Subject expansion:** one authorized processing operation at a time, with a
  legacy compatibility scope.
- **Interaction:** does not formally authorize safeguards, DPIA, or role rules
  in minimum v0.6.
- **Negative limit:** no conclusion that GDPR is inapplicable or complied with.

Material influence remains a legacy preliminary input only. A future refined
version must separately evaluate:

- personal-data processing;
- a decision;
- solely automated processing;
- legal effect;
- similarly significant effect;
- meaningful human involvement indicators;
- profiling.

Material influence is not a substitute for the full predicate. The refined
rule is deferred unless its truth table and complete Evidence proposition set
are separately approved.

## 5. Minimum new AI Act candidates

All three candidates remain blocked until the exact Evidence listed in the
Evidence gap analysis is approved.

### 5.1 AI_ACT_DEPLOYER_ROLE_RELEVANCE

- **Phase:** `role_relevance`.
- **Framework:** EU AI Act.
- **Purpose:** screen deployer-role relevance for one actor and one
  system/workflow/use scope.
- **Legal question:** Do concrete use and authority facts make deployer-role
  analysis potentially relevant for this actor in this scope?
- **Depends on:** `AI_ACT_HIGH_RISK_EMPLOYMENT`.
- **Accepted upstream statuses:** `potentially_applies`.
- **Subject selector:** actors operating or using the scoped system in or for a
  scoped organization/workflow.
- **Required facts:** system operation/use; selected system; own-organization
  or on-behalf use; relevant actor and workflow identities.
- **Conditional facts:** concrete branding, development, commissioning,
  modification, and intended-purpose conduct when those facts conflict with a
  narrow deployer hypothesis.
- **Statuses:** `potentially_applies`, `does_not_apply`, `undetermined`.
- **Reason codes:** `SCOPED_SYSTEM_USE_REPORTED`,
  `USE_UNDER_ACTOR_AUTHORITY_REPORTED`, `ROLE_CONDUCT_CONFLICT`,
  `DEPLOYER_FACTS_INCOMPLETE`.
- **Legal basis required:** atomic relevant Article 3 definitions and any exact
  provision used by the predicate.
- **Negative limit:** no final deployer classification, no conclusion for other
  actors/systems/uses, and no exclusion of another AI Act role.
- **Evidence readiness:** blocked.

### 5.2 AI_ACT_DEPLOYER_USE_INSTRUCTIONS_RELEVANCE

- **Phase:** `obligation_relevance`.
- **Framework:** EU AI Act.
- **Purpose:** screen relevance of the deployer's instructions-for-use pathway.
- **Depends on:** `AI_ACT_DEPLOYER_ROLE_RELEVANCE`.
- **Accepted upstream statuses:** `potentially_applies`.
- **Subject selector:** same actor/system/workflow/operation scope as the
  accepted deployer invocation.
- **Required facts:** instructions-for-use availability; responsible actor;
  system/version correspondence; actual use/configuration scope.
- **Conditional facts:** deviations from documented instructions and
  modifications when reported.
- **Statuses:** `potentially_applies`, `does_not_apply`, `undetermined`.
- **Reason codes:** `DEPLOYER_PATH_ACTIVE`, `INSTRUCTIONS_AVAILABLE`,
  `INSTRUCTIONS_SCOPE_UNKNOWN`, `REPORTED_USE_DEVIATION`.
- **Legal basis required:** atomic Article 26(1) propositions and any definition
  needed by the predicate.
- **Negative limit:** no determination that instructions are adequate, followed,
  or sufficient for overall compliance.
- **Artefacts:** system instructions for use, version mapping, use procedure,
  and deviation record.
- **Evidence readiness:** blocked.

### 5.3 AI_ACT_DEPLOYER_HUMAN_OVERSIGHT_RELEVANCE

- **Phase:** `obligation_relevance`.
- **Framework:** EU AI Act.
- **Purpose:** screen relevance of scoped human-oversight use controls.
- **Depends on:** `AI_ACT_DEPLOYER_ROLE_RELEVANCE`.
- **Accepted upstream statuses:** `potentially_applies`.
- **Subject selector:** same actor/system/workflow/operation scope as the
  accepted deployer invocation.
- **Required facts:** assigned reviewer; substantive basis review; authority;
  realistic ability; disregard/override capacity; actual reliance pattern.
- **Conditional facts:** restoration and escalation where automatic exclusion
  occurs.
- **Statuses:** `potentially_applies`, `does_not_apply`, `undetermined`.
- **Reason codes:** `DEPLOYER_PATH_ACTIVE`, `OVERSIGHT_RELEVANT`,
  `NOMINAL_REVIEW_INDICATORS`, `OVERRIDE_AUTHORITY_UNKNOWN`,
  `REALISTIC_CAPACITY_UNKNOWN`.
- **Legal basis required:** atomic Article 26(2) and any relied-on Article 14
  propositions, separately identified.
- **Negative limit:** genuine-review indicators do not establish full AI Act
  compliance or resolve GDPR Article 22.
- **Artefacts:** human-review procedure, reviewer training, override and
  escalation records.
- **Evidence readiness:** blocked.

## 6. Deterministic artefact projection

Artefact projection is phase `artefact_projection` and consumes only completed
formal Findings in accepted statuses. It creates scoped
`ArtefactRequirement` records linked to the triggering Finding and legal basis.

It does not itself declare non-compliance. Missing, unreviewed, mismatched, or
contradictory artefacts become informational unresolved gaps unless a future
Evidence-ready rule expressly defines a formal gap predicate.

| Trigger | Candidate metadata-only artefacts |
| --- | --- |
| deployer-role relevance | responsibility mapping; scoped system/use record |
| use-instructions relevance | instructions for use; version mapping; deviation record |
| oversight relevance | human-review procedure; training; override/escalation record |

## 7. Informational role projection

Minimum v0.6 may deterministically display:

- GDPR controller-like hypothesis;
- GDPR processor-like hypothesis;
- joint-control analysis required;
- independent-reuse operation detected.

These outputs use concrete decision-right and operation facts. They are not
formal Findings, do not authorize downstream GDPR rules, and do not carry a
Finding status. Client recruitment and independent talent-pool reuse are
different processing-operation scopes.

## 8. Deferred formal rules

The following are not implemented as formal v0.6 rules:

- `GDPR_ACTOR_ROLE_RELEVANCE`;
- `GDPR_ARTICLE22_SAFEGUARD_RELEVANCE`;
- `GDPR_DPIA_RELEVANCE`;
- AI Act logging/monitoring;
- AI Act affected-person information;
- FRIA;
- document-content sufficiency;
- final provider, deployer, controller, processor, or joint-controller
  classification.

Their questions may be collected for future use or shown as informationally
unavailable, but they cannot produce authoritative-looking Findings.

## 9. Dependency validation and execution

The minimum graph is acyclic and subject-scoped:

```text
AI_ACT_HIGH_RISK_EMPLOYMENT
  -> AI_ACT_DEPLOYER_ROLE_RELEVANCE
       -> AI_ACT_DEPLOYER_USE_INSTRUCTIONS_RELEVANCE
       -> AI_ACT_DEPLOYER_HUMAN_OVERSIGHT_RELEVANCE
            -> artefact projection
```

The registry:

1. validates unique rule/version identity;
2. validates every dependency exists;
3. validates accepted statuses and scope compatibility;
4. detects cycles;
5. calculates deterministic topological order;
6. hashes the canonical dependency graph into the ruleset baseline.

A missing dependency or cycle is a registration error, not a runtime Finding.
An upstream status outside the accepted set produces
`blocked_by_dependency`. An upstream Finding change invalidates all transitive
downstream invocations, execution records, Findings, requirements, gaps,
bindings, and reports.

## 10. AssessmentContext access rules

For one invocation, context exposes:

- immutable facts snapshot;
- current invocation;
- only the Finding summaries named in `depends_on`;
- scoped authorization;
- ruleset baseline;
- Legal Evidence baseline.

Attempted access to another Finding is rejected. Rules cannot receive services,
UI state, translations, mutable collections, or mutation hooks.

## 11. Evidence and implementation gate

A new candidate may be enabled only when:

- rule metadata and scope selector validate;
- its predicate and conditional requirements are truth-tabled;
- every positive, negative, and unresolved proposition has atomic reviewed
  Evidence;
- reason codes and trace steps are fixed;
- unauthorized, dependency-blocked, Evidence-blocked, missing, contradictory,
  and failure states are tested;
- no invocation can access undeclared upstream state;
- bilingual output preserves canonical values.

## 12. Version decisions

- engine: `3.0.0`;
- `AssessmentContext`: `1.0.0`;
- `RuleExecutionRecord`: `1.0.0`;
- facts: `3.0.0`;
- questionnaire: `3.0.0`;
- report: `2.0.0`;
- existing rule versions: `2026.1` until logic changes;
- new rules: independently versioned only after Evidence approval.
