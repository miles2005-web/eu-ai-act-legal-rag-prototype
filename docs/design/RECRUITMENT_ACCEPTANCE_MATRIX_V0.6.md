# Recruitment Acceptance Matrix v0.6

Document status: `0.6-design-draft`

## 1. Conventions

This matrix defines intended behavior, not implemented tests.

- Only scoped `AuthorizedRuleInvocation` records permit execution.
- One invocation produces at most one substantive Finding.
- Execution state appears in `RuleExecutionRecord`, not as a legal conclusion.
- Missing, unauthorized, dependency-blocked, Evidence-blocked, and failed paths
  produce no substantive Finding.
- Role hypotheses are informational unless an Evidence-ready formal rule exists.
- English and Simplified Chinese produce identical canonical output.

## 2. Core recruitment scenarios

| ID | Scenario | Scoped authorization | Expected formal/informational output | Missing facts | Evidence requirement |
| --- | --- | --- | --- | --- | --- |
| A1 | Employer directly operates one ranking system | employment rule for employer/system/workflow/operation; deployer chain only if confirmed | employment Finding; scoped deployer chain after Evidence approval | only rule-specific facts | AI Act recruitment pack |
| A2 | Recruiter acts only within documented instructions | employment and Article 22 as confirmed; deployer chain only for authorized actor | processor-like and controller-like hypotheses are informational; contract alone does not establish roles | actual conduct and decision-right facts if unknown | GDPR definitions/guidance for explanation; AI Act pack for formal chain |
| A3 | Recruiter selects its own screening criteria or system | scoped employment/deployer authorization | concrete conduct shown; deployer-role relevance only after Evidence approval; GDPR role remains informational | configuration, system selection, scope | AI Act role Evidence |
| A4 | Employer and recruiter share concrete objective, criteria, or approval decisions | separate subject scopes | “joint-control analysis required” informational output; no formal GDPR role Finding | exact conduct by actor/operation | GDPR role Evidence remains deferred |
| A5 | Recruiter independently reuses candidate data for a talent pool | separate client-screening and talent-pool operations | independent-reuse operation detected; separate controller-like hypothesis for reuse | reuse purpose, data, recipients, retention | GDPR role Evidence deferred |
| A6 | Genuine substantive human review with authority and realistic capacity | Article 22 legacy screen as confirmed; AI oversight for scoped deployer chain | review facts shown; oversight relevance after Evidence approval; no compliance conclusion | actual reliance if unknown | AI Act Article 26(2); future Article 22 pack |
| A7 | Nominal review routinely follows AI output | same scoped modules | nominal-review indicators; future Article 22 refinement remains deferred; oversight result follows approved rule | authority/capacity where unknown | oversight Evidence and future guidance |
| A8 | Automatic exclusion before review | employment; legacy Article 22; scoped oversight if authorized | employment Finding; automatic-exclusion facts; no refined Article 22 claim | legal/similar effect and restoration facts | AI Act pack; future GDPR pack |
| A9 | Recommendation only with independent human decision | employment and legacy Article 22 as confirmed | employment may remain potentially applicable; legacy Article 22 unchanged; no overall clearance | material-influence and review facts | separate AI Act and GDPR Evidence |
| A10 | Explicit Unknown answers | module may be authorized | `missing_facts` execution record; no substantive Finding | deterministic scoped paths | none until execution completes |
| A11 | Contract says instructions-only but practice shows independent criteria/reuse | separate operation and actor scopes | contradiction retained; informational role hypotheses; formal rule may be undetermined only if approved | exact operation boundaries | formal GDPR role Evidence deferred |
| A12 | Actors known, role conduct incomplete | role module if explicitly scoped | unresolved hypotheses; `missing_facts`; no formal role Finding | concrete decision rights and conduct | no binding until complete |
| A13 | AI Act and GDPR modules both authorized | separate framework/scopes | independent Findings; no framework controls the other | per-invocation requirements | Evidence bound by framework |

## 3. Identity and multi-operation scenarios

| ID | Scenario | Expected behavior |
| --- | --- | --- |
| B1 | Two AI systems in one recruitment case | Stable system IDs create separate invocation scopes, Findings, execution records, and Evidence traces. A result for system A never appears for system B. |
| B2 | One recruiter participates in client screening and talent-pool reuse | Two processing-operation IDs are mandatory; purpose, data, role hypotheses, authorization, and gaps remain separate. |
| B3 | Employer is controller-like for client recruitment and recruiter controller-like for independent reuse | Two informational hypotheses use different operation scopes; no composite role Finding is emitted. |
| B4 | One actor holds different framework-specific roles | Role hypotheses and formal Findings retain framework plus actor/system/workflow/operation scope and may coexist. |
| B5 | Edit actor display name | Stable actor ID survives; localized labels and display name changes do not create a new legal subject. |
| B6 | Delete or split an actor | Dependent references require explicit handling; retired IDs are not reused; all affected downstream state is invalidated. |

## 4. Authorization and execution-state scenarios

| ID | Scenario | Expected authorization/result |
| --- | --- | --- |
| C1 | Authorize deployer chain for employer/system A only | Only matching invocations execute; system B and recruiter scopes remain `not_authorized`. |
| C2 | Same rule ID authorized for two operations | Two `AuthorizedRuleInvocation` records and two deterministic invocations are preserved. |
| C3 | Unauthorized actor has complete facts | Actor remains unassessed; a `not_authorized` execution record is separate from Findings. |
| C4 | Required Evidence pack unavailable | Invocation is `blocked_by_evidence`; no substantive Finding or empty Evidence trace appears. |
| C5 | Required fact Unknown | Invocation is `missing_facts`; questionnaire receives scoped paths; no substantive Finding appears. |
| C6 | Upstream status not accepted | Downstream invocation is `blocked_by_dependency`; it does not execute. |
| C7 | Rule throws an isolated failure | Invocation is `failed`; other independent scopes continue deterministically. |
| C8 | UI shows results | `RuleExecutionRecord` appears in a separate audit section and is not styled as a formal legal Finding. |

## 5. Dependency-DAG scenarios

| ID | Scenario | Expected registration/execution behavior |
| --- | --- |
| D1 | Direct cyclic dependency | Registration fails before assessment with a deterministic cycle error. |
| D2 | Indirect cycle across three rules | Registration fails and identifies the canonical cycle path. |
| D3 | Missing dependency | Registration fails; the rule cannot enter the registry or ruleset baseline. |
| D4 | Same graph registered in different input order | Canonical topological execution order and graph hash are identical. |
| D5 | Independent rules share phase/order prefix | Tie-break by phase, ordering key, rule ID, version, and canonical subject scope. |
| D6 | Upstream Finding status/reason changes | Every transitive downstream invocation, Finding, artefact requirement, gap, binding, and report is invalidated. |
| D7 | Upstream scope changes | Only affected transitive scopes invalidate; unrelated operations remain current. |

## 6. Baseline and fingerprint scenarios

| ID | Change | Expected report-validity result |
| --- | --- | --- |
| E1 | actor, system, workflow, or operation fact | affected report invalid |
| E2 | relationship or concrete decision-right fact | affected role/obligation chain invalid |
| E3 | scoped authorization | affected report invalid |
| E4 | ordered rule/version set | report invalid |
| E5 | dependency graph hash | report invalid |
| E6 | rule behavior/ruleset baseline ID | report invalid |
| E7 | Evidence-pack version | affected report invalid |
| E8 | manifest hash | affected report invalid |
| E9 | legal-source baseline ID | affected report invalid |
| E10 | facts schema, questionnaire, engine, or report version | report invalid |

Unchanged inputs reproduce the same fingerprint and deterministic ordering.

## 7. Temporal and territorial scenarios

| ID | Scenario | Expected output |
| --- | --- | --- |
| F1 | Intended-use or put-into-service date unknown | Missing temporal facts are preserved; report states “Territorial and temporal applicability not assessed.” |
| F2 | Establishment/use/output/affected-person locations unknown | Missing territorial facts are preserved; no scope Finding is invented. |
| F3 | Facts complete but module not implemented/authorized | Same scope-limitation statement; no temporal or territorial legal conclusion. |
| F4 | Legal-source baseline changes | Report validity is invalidated even when case facts are unchanged. |

## 8. Public deployment and artefact scenarios

| ID | Scenario | Expected behavior |
| --- | --- | --- |
| G1 | User opens public assessment | Visible warning prohibits real applicant data, CVs, contact details, special-category data, confidential/privileged material, trade secrets, and identifiable production records. |
| G2 | User attempts document upload | No upload control exists, or the path is rejected before transmission/storage. |
| G3 | User supplies artefact metadata | Only metadata and opaque reference are recorded; no content is fetched, extracted, hashed, or persisted. |
| G4 | Artefact is absent | Informational unresolved gap may appear; absence is not styled as non-compliance. |
| G5 | Artefact is supplied but unreviewed | `supplied_unreviewed` remains distinct from compliant or adequate. |
| G6 | Artefact wording conflicts with reported practice | Both remain visible; no automatic compliance conclusion. |

## 9. Article 22 compatibility scenarios

| ID | Scenario | Expected result |
| --- | --- | --- |
| H1 | Load v0.5 recruitment fixture | `GDPR_ARTICLE22_RELEVANCE / 2026.1` and existing Evidence behavior remain unchanged. |
| H2 | New meaningful-review facts are present | They do not silently alter the `2026.1` predicate. |
| H3 | Material influence is yes but complete refined predicate is unresolved | No future refined Article 22 Finding is emitted; material influence is not substituted for sole automation/effect analysis. |
| H4 | Refined rule lacks approved truth table or Evidence | It remains unavailable/deferred and cannot be authorized as a formal rule. |

## 10. Language and presentation scenarios

| ID | Scenario | Expected result |
| --- | --- | --- |
| I1 | Switch English to Simplified Chinese | Active case, IDs, authorization, context, execution records, Findings, Evidence, baselines, fingerprint, and report remain identical. |
| I2 | Translate role hypothesis | Only display text changes; scope and informational status remain canonical. |
| I3 | Translate Evidence page | Official excerpt, citation, and stable Evidence ID remain unchanged. |
| I4 | Display raw audit details | Rule IDs and scope IDs remain available in technical details, not primary recommendations. |

## 11. Canonical report acceptance

The report separately displays:

- baseline and fingerprint;
- temporal/territorial scope limitation;
- actors, systems, workflows, and operations;
- observable relationships and decision rights;
- informational role hypotheses;
- formal Findings;
- `RuleExecutionRecord` audit;
- scoped missing facts;
- deterministic artefact requirements;
- supplied metadata-only artefacts;
- informational unresolved gaps;
- per-Finding Legal Evidence Trace;
- unauthorized, blocked, deferred, and excluded modules.

Every formal Finding includes invocation ID, actor/system/workflow/operation
scope, framework, rule ID/version, status, reason codes, fact references, legal
bases, Evidence IDs, and limitation text.

## 12. Non-functional acceptance

- deterministic serialization and ordering;
- dependency validation before execution;
- no fact mutation through `AssessmentContext`;
- undeclared upstream Finding access rejected;
- schema v3 and report v2 round trips;
- v0.5 case and report compatibility;
- no LLM or runtime network participation in formal assessment;
- no localized value used as an ID;
- clean checkout includes approved runtime Evidence packs;
- public UI contains no document upload or persistence path.

## 13. Release gate

Minimum v0.6 is releasable only when:

1. the three new AI Act candidates are either Evidence-approved or clearly
   unavailable without Findings;
2. dependency graph validation and deterministic ordering pass;
3. scoped authorization and execution-state separation pass;
4. every fingerprint component is regression-tested;
5. all applicable scenarios above are automated;
6. privacy restrictions are visible and enforced;
7. English and Simplified Chinese canonical results are identical;
8. v0.5 rule and report regressions pass.
