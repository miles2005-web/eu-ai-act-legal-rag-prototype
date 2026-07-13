# Dynamic Rule-Driven Questionnaire Router Plan

## 1. Purpose and boundaries

The custom-case questionnaire should be driven by implemented assessment modules and their declared fact requirements, rather than by a fixed employment form. The router must remain deterministic: it may suggest a module from structured facts and controlled routing signals, but it must not decide that a legal rule applies. The user confirms which suggested modules are assessed.

This design preserves the existing separation of responsibilities:

- the questionnaire collects and normalizes facts;
- the router identifies reachable implemented modules;
- the requirement validator identifies facts missing from confirmed modules;
- assessment rules produce legal findings;
- the workflow resolves evidence and builds immutable reports;
- the UI presents routing, questions, findings, and unsupported paths.

The router must not use an LLM for formal routing, write localized labels into `AssessmentFacts`, or treat the absence of a finding as proof that no legal risk exists.

## 2. Current-state audit

### 2.1 Current Streamlit questions

The blank assessment form in `assessment_app.py` currently renders only three substantive fact controls.

| Current question | Canonical target | Current input | Rules supported | Defect |
|---|---|---|---|---|
| In which context is the AI system used? | `use_context.domain` | `UseDomain` single choice | AI Act employment routing; contextual signal for all future rules | Useful as universal intake, but it does not itself select candidate modules. |
| What task does the AI system perform? | `use_context.task` plus UI normalization metadata | free text | AI Act employment function matching; controlled signals for GDPR and Data Act facts | One field is being used both as a factual description and as an implicit router. Unsupported text cannot produce a route. |
| Does the AI output materially influence an employment-related decision? | `use_context.materially_influences_decision` | `TriState` | AI Act employment and GDPR Article 22 | The wording is employment-specific although GDPR reuses the same neutral fact path. It is inappropriate for lending, education, healthcare, or other individual decisions. |

Case creation separately collects:

| Current question | Storage target | Limitation |
|---|---|---|
| Case name | `AssessmentCase.name` | This is case metadata, not `system.name`. |
| Case description | `AssessmentCase.description` | It does not populate intended purpose or other system facts. |

The current blank form does **not** directly ask for:

- `system.name` or `system.intended_purpose`;
- affected persons or relevant parties;
- neutral decision/operational impact;
- personal-data processing;
- automated individual decision-making;
- meaningful human involvement before effect;
- connected product, related service, or generated data.

### 2.2 Existing questionnaire foundation

The UI-neutral questionnaire package already provides useful foundations:

- `Question` maps one question to one `AssessmentFacts` leaf path;
- `QuestionRegistry` validates paths and preserves deterministic order;
- `QuestionnaireEngine` converts `RuleRequirementResult` objects into unanswered questions;
- `FactRequirementValidator` distinguishes unknown, not-provided, and invalid paths.

However, it is not connected to `assessment_app.py`. Its current models also have material limitations for dynamic routing:

- `Question.text` and option labels are single-language strings;
- only one question may be registered per fact path, preventing context-specific wording for a shared fact;
- questions have no owning module or rule ID;
- there are no eligibility hints, dependencies, invalidation rules, help text, or unsupported-path messages;
- `QuestionnaireEngine` receives missing requirements only after all registered rules have been considered;
- there is no user-confirmed candidate-module state;
- `QuestionnairePlan` cannot represent suggested, deselected, unsupported, or completed modules.

The foundation should be extended compatibly rather than replaced.

### 2.3 Current rule requirements

| Implemented rule | Framework | Required facts |
|---|---|---|
| `AI_ACT_HIGH_RISK_EMPLOYMENT` | `EU_AI_ACT` | `use_context.domain`, `use_context.task`, `use_context.materially_influences_decision` |
| `GDPR_ARTICLE22_RELEVANCE` | `GDPR` | `data_protection.personal_data_processed`, `data_protection.automated_individual_decision`, `use_context.materially_influences_decision` |
| `EU_DATA_ACT_RELEVANCE` | `EU_DATA_ACT` | `data_act.connected_product`, `data_act.related_service`, `data_act.data_generated` |

The shared `use_context.materially_influences_decision` path demonstrates why question wording must be module-aware: the same canonical fact needs employment wording in the AI Act module and neutral legal/significant-effect wording in the GDPR module.

### 2.4 Demo-specific assumptions

The fixed form currently inherits these recruitment assumptions:

- employment is treated as the primary decision context;
- material influence is described only as influence on an employment decision;
- recruitment task phrases are the most complete controlled free-text mappings;
- personal-data and Data Act facts are expected to arrive from fixtures or controlled task phrases rather than explicit questions;
- all registered rules are executed or reported as missing even though no routing choice was made.

The Recruitment demo works because its fixture already contains the employment facts. The Industrial demo works because its fixture already contains Data Act facts. These fixture-complete paths mask the blank-case routing gap.

### 2.5 Stale dependent facts

Within one custom case, an upstream edit currently clears the report and selected finding but does not invalidate dependent facts. Stale values can therefore survive:

- changing `use_context.domain` from employment to essential services leaves the old employment task and material-influence answer;
- replacing a task containing a controlled phrase does not clear earlier normalization-derived fact updates that are absent from the new text;
- changing personal-data processing to `NO` does not clear a questionnaire-derived automated-decision answer;
- changing connected-product and related-service answers to `NO` does not clear a previously derived `data_act.data_generated` answer;
- module selection does not exist, so no module-specific question state can be invalidated;
- widget values and normalization metadata may retain the semantic context of an earlier answer.

Invalidation must use provenance and explicit dependency declarations. It must not indiscriminately erase facts that were independently confirmed by the user.

## 3. Three questionnaire layers

### 3.1 Layer A — universal intake

Universal intake establishes a neutral system profile and a small set of routing facts. These questions appear for every custom case and must not contain framework-specific legal conclusions.

| Intake topic | Canonical target or routing state | Notes |
|---|---|---|
| System name | `system.name` | Distinct from case name. |
| Intended purpose | `system.intended_purpose` | Plain factual description. |
| Use domain | `use_context.domain` | Canonical `UseDomain`; localized labels only. |
| Task description | `use_context.task` plus original/normalized UI metadata | Free text is preserved. Controlled mappings may create routing tags only when deterministic. |
| Relevant parties | `use_context.affected_persons` | Canonical `AffectedPerson` values. |
| Decision or operational impact | `use_context.materially_influences_decision` | Neutral wording: impact on an individual decision or material operational outcome. Module-specific follow-ups may refine the context. |
| Meaningful human involvement before effect | `use_context.human_review_before_effect` | Useful for the loan acceptance case and future fuller Article 22 analysis; it is not currently a required fact of the preliminary GDPR rule. |
| Personal data may be involved | `data_protection.personal_data_processed` | Tri-state routing fact. |
| Connected product may be involved | `data_act.connected_product` | Tri-state routing fact. |
| Related service may be involved | `data_act.related_service` | Tri-state routing fact. |

Universal questions use neutral bilingual prompts and canonical enum values. `Unknown` is a valid answer. The router must not infer `NO` from an unanswered control.

### 3.2 Layer B — rule-routing questions

Routing questions resolve whether an implemented module should be suggested. They are short factual gates, not legal tests. Examples include:

- Is the decision about an identifiable individual?
- Is the outcome produced solely or substantially through automated processing?
- Does the scenario involve a connected product or a related service?
- Is the system used for recruitment, candidate selection, worker evaluation, ranking, or recommendation?

Routing produces module states, not findings:

- `suggested` — deterministic eligibility hints match;
- `needs_confirmation` — a broad signal matches or a relevant fact is unknown;
- `confirmed` — the user elected to assess the implemented module;
- `deselected` — the user declined the suggestion for this run;
- `not_reached` — deterministic gates do not currently reach the module;
- `unsupported` — a recognized legal route has no implemented rule.

The UI lists implemented suggestions and unsupported paths separately. It must explain that confirming a module starts a preliminary assessment and is not an admission that the regulation applies.

### 3.3 Layer C — rule-specific follow-ups

For every confirmed implemented module:

1. retrieve the corresponding `AssessmentRule` from `RuleRegistry`;
2. validate the current facts against its `required_fact_paths`;
3. pass only that rule's `RuleRequirementResult` to `QuestionnaireEngine`;
4. render mapped questions for missing facts in registry order;
5. show a localized unsupported-path message for any required fact without a question mapping;
6. recompute requirements after answers are saved.

Optional contextual questions, such as meaningful human involvement for the preliminary GDPR module, may be shown after required questions. They must be marked as contextual and must not be represented as a current rule requirement.

This prevents the AI Act employment question from appearing in a loan or industrial case: the employment module is not confirmed unless its deterministic employment hints are reached.

## 4. Rule-question registry design

### 4.1 Keep legal rules and questionnaire metadata separate

Question wording and routing UX should not be added as UI fields on `AssessmentRule`. Instead, create a companion `RuleQuestionnaireDefinition` registry keyed by the same stable `rule_id`. Registry validation must ensure that:

- the rule exists in `RuleRegistry`;
- framework metadata matches;
- declared required paths exactly match `rule.required_fact_paths`;
- every required path has a question or an explicit unsupported-path message;
- dependencies refer to known canonical paths or routing signals;
- invalidation targets are valid and acyclic.

This gives each rule a declarative questionnaire definition without coupling legal evaluation code to Streamlit or translation strings.

### 4.2 Proposed schema

```python
@dataclass(frozen=True, slots=True)
class RuleQuestionnaireDefinition:
    module_id: str
    module_version: str
    rule_id: str
    framework: RegulatoryFramework
    title_key: str
    description_key: str
    eligibility_hints: tuple[EligibilityHintGroup, ...]
    required_fact_paths: tuple[str, ...]
    questions: tuple[QuestionSpec, ...]
    unsupported_path_message_key: str

@dataclass(frozen=True, slots=True)
class QuestionSpec:
    question_id: str
    fact_path: str
    prompt_key: str
    help_key: str | None
    answer_type: AnswerType
    options: tuple[QuestionOptionSpec, ...]
    dependencies: tuple[QuestionCondition, ...]
    invalidates: tuple[InvalidationSpec, ...]
    required_for_rule: bool

@dataclass(frozen=True, slots=True)
class QuestionOptionSpec:
    value: str                 # canonical stored value
    label_key: str             # localized presentation key

@dataclass(frozen=True, slots=True)
class EligibilityHint:
    source: str                # fact path or confirmed routing tag
    operator: HintOperator     # equals, in, any_yes, contains_tag
    expected: object

@dataclass(frozen=True, slots=True)
class InvalidationSpec:
    target_fact_path: str
    when: QuestionCondition
    provenance_question_ids: tuple[str, ...]
```

Use central i18n keys rather than storing English and Chinese text directly in domain objects. A catalogue-completeness test must require both `en` and `zh-CN` entries. Options always store canonical values such as `employment`, `yes`, `no`, and `unknown`.

The current `Question` model can remain supported. `QuestionSpec` can compile into a localized `Question` at presentation time, preserving the current engine API while adding context and dependencies. The uniqueness constraint should move from `fact_path` to `(module_id, question_id)` so that two modules may use different wording for the same canonical path.

### 4.3 Example definitions

```python
RuleQuestionnaireDefinition(
    module_id="ai-act-employment",
    module_version="1.0",
    rule_id="AI_ACT_HIGH_RISK_EMPLOYMENT",
    framework=RegulatoryFramework.EU_AI_ACT,
    eligibility_hints=(
        all_of(fact("use_context.domain").equals("employment")),
        any_of(tag("recruitment"), tag("candidate_ranking"), tag("worker_evaluation")),
    ),
    required_fact_paths=(
        "use_context.domain",
        "use_context.task",
        "use_context.materially_influences_decision",
    ),
    questions=(...),
    unsupported_path_message_key="router.unsupported.ai_act_employment",
)
```

```python
RuleQuestionnaireDefinition(
    module_id="gdpr-article22-relevance",
    module_version="1.0",
    rule_id="GDPR_ARTICLE22_RELEVANCE",
    framework=RegulatoryFramework.GDPR,
    eligibility_hints=(
        any_of(
            fact("data_protection.personal_data_processed").equals("yes"),
            fact("data_protection.personal_data_processed").equals("unknown"),
        ),
        any_of(tag("individual_decision"), tag("credit_decision"), tag("recruitment")),
    ),
    required_fact_paths=(
        "data_protection.personal_data_processed",
        "data_protection.automated_individual_decision",
        "use_context.materially_influences_decision",
    ),
    questions=(...),
    unsupported_path_message_key="router.unsupported.gdpr_article22",
)
```

The `unknown` GDPR hint leads to `needs_confirmation`, not automatic confirmation. A confirmed `NO` for personal-data processing makes the module `not_reached` unless the user explicitly requests it.

### 4.4 Unsupported-path catalogue

Recognized but unimplemented routes need their own declarative registry:

```python
UnsupportedModuleDefinition(
    module_id="ai-act-essential-services-credit",
    framework=RegulatoryFramework.EU_AI_ACT,
    title_key="unsupported.ai_act.credit.title",
    message_key="unsupported.ai_act.credit.message",
    eligibility_hints=(
        fact("use_context.domain").equals("essential_services"),
        tag("credit_decision"),
    ),
)
```

The message should say that a dedicated AI Act credit/essential-services assessment module is not implemented. It must not say that the AI Act is inapplicable or that the system is compliant.

Future Article 6(1) product-safety routing can be added as another definition using `high_risk.is_safety_component_or_product`, `high_risk.product_covered_by_annex_i`, and `high_risk.requires_third_party_conformity_assessment`, without changing the router algorithm. No Article 6(1) rule is implemented in this phase.

## 5. Deterministic routing algorithm

### 5.1 Inputs and output

Inputs:

- current `AssessmentFacts` snapshot;
- fact provenance;
- confirmed controlled routing tags from the existing normalization layer;
- rule registry;
- rule-question registry;
- unsupported-module registry;
- prior user module selections.

Output is an immutable `QuestionnaireRoutePlan` containing:

- universal questions;
- module suggestions and reasons;
- confirmed and deselected module IDs;
- rule-specific unanswered questions;
- unsupported modules and localized message keys;
- invalidated fact paths;
- unmapped required paths;
- deterministic registry/router version metadata.

### 5.2 Algorithm

1. **Normalize intake values.** Retain original text. Use only canonical structured values and confirmed controlled mappings as routing inputs.
2. **Apply dependency invalidation.** Compare the saved upstream values with the prior route snapshot and clear only provenance-linked dependent answers.
3. **Evaluate eligibility hints.** Evaluate definitions in registry order with three-valued logic: matched, not matched, or unknown.
4. **Build suggestions.** A complete deterministic match becomes `suggested`; an incomplete but plausible match becomes `needs_confirmation`; a recognized unimplemented route becomes `unsupported`.
5. **Obtain user confirmation.** Display why each module was suggested and allow confirm or deselect. Do not silently activate a legal module.
6. **Validate confirmed rules.** Use `FactRequirementValidator` against only the confirmed implemented rules.
7. **Build follow-ups.** Map missing paths through the question registry, filter by dependencies, and preserve deterministic definition order.
8. **Repeat after save.** Recompute until no invalidation or candidate-state change occurs. This is a deterministic fixed-point calculation, not an LLM conversation.
9. **Create an execution plan.** Pass the confirmed rule IDs and routing version into the assessment run boundary.
10. **Run assessment.** Only implemented, confirmed rules create findings. Deselected or unsupported modules are presented as not assessed, never as `does_not_apply`.

### 5.3 Selection-aware execution compatibility

The current `AssessmentEngine.run(facts)` executes all registered rules. A later implementation should introduce an optional execution plan, for example `run(facts, rule_ids=None)`, where `None` retains today's all-rule behavior for demos and existing tests. A supplied ordered tuple executes the corresponding registry subset without changing any rule implementation.

Alternatively, a small application service may create a deterministic registry view for confirmed rule IDs. The selection must be recorded in the immutable run snapshot so that reports can distinguish:

- assessed and completed;
- assessed but missing facts;
- explicitly deselected;
- recognized but unsupported;
- not reached by routing.

The workflow must not construct rule branches inside Streamlit.

## 6. User confirmation and unsupported routes

The workspace should show two separate lists.

### Implemented assessment modules

Each suggestion displays:

- localized module name and framework;
- factual reason for suggestion;
- current state;
- confirm/deselect control;
- missing-fact count after confirmation.

Suggested modules are unchecked until explicitly confirmed for a custom case. Demo fixtures may carry a predefined confirmed-module list to preserve one-click behavior.

### Recognized paths not yet implemented

Each entry displays:

- the recognized scenario;
- the relevant framework at a high level;
- a clear statement that the dedicated assessment is unavailable;
- an instruction to obtain further legal assessment.

Unsupported paths never produce a negative legal finding.

## 7. Dependency invalidation model

### 7.1 Principles

- Invalidation follows an explicit directed acyclic dependency graph.
- A dependent value is cleared only when its `FactMetadata.question_id` or UI normalization metadata shows that it came from the invalidated branch.
- Independently confirmed facts are preserved and flagged for review if they conflict with the new route.
- Module selection derived from invalidated facts is reset to `needs_confirmation` or `not_reached`.
- Any fact or routing change invalidates the current report and selected finding.
- Language, case identity, unrelated facts, and immutable historical runs are preserved.

### 7.2 Minimum dependency rules

| Upstream change | Dependent state to invalidate | State to preserve |
|---|---|---|
| `use_context.domain` leaves `employment` | employment-specific material-influence answer, employment question widget, employment routing tags and module confirmation | personal-data facts, Data Act facts, language, case metadata |
| task description changes | prior task-derived controlled tags and only those fact updates whose provenance identifies the old normalization mapping | explicitly answered structured facts |
| `personal_data_processed` becomes `NO` | GDPR-only automated-decision and Article 22 follow-up widgets if they were collected under that dependency | unrelated decision-impact facts and original task text |
| both connected product and related service become `NO` | Data Act generated-data answer if it came from the Data Act branch | operational data descriptions outside the Data Act namespace |
| a suggested module is deselected | its unanswered question state and execution-plan membership | already confirmed facts unless an upstream dependency also changed |

### 7.3 State transition sequence

For an upstream change such as employment → essential services:

1. persist the new canonical domain;
2. compute invalidations from the prior route plan;
3. clear provenance-linked employment answers and widgets;
4. clear `assessment_report`, report ownership metadata, selected finding, and case-derived UI caches;
5. recompute candidate and unsupported modules;
6. rebuild the workflow/runtime bundle if its execution plan is composition-specific, rehydrating the same case ID and current facts first;
7. preserve `ui_language`, case name/description, unrelated facts, and navigation intent;
8. render the neutral universal intake and newly reached modules.

For the current in-memory prototype, bundle replacement must never discard the case. Prefer a selection-aware engine using one stable bundle. If a bundle must be rebuilt, snapshot and recreate the case atomically before replacing session state.

## 8. Custom personal-loan acceptance flow

### 8.1 Universal intake

The user records:

- domain: `essential_services`;
- task: analysis of income, occupation, credit history, and account transactions; calculation of a credit score; automatic loan amount or rejection;
- affected person: `consumer`;
- personal data: `YES`;
- decision or significant effect: `YES`;
- connected product / related service: `NO` or unknown as appropriate.

Controlled bilingual tags may identify `credit_decision` and `individual_decision`, but the UI shows the mapping and asks the user to confirm it. Unsupported free text remains unclassified.

### 8.2 Routing result

The router should produce:

- `GDPR_ARTICLE22_RELEVANCE`: suggested, because personal data and an individual credit decision are indicated;
- `AI_ACT_HIGH_RISK_EMPLOYMENT`: not reached, because the domain is not employment;
- `EU_DATA_ACT_RELEVANCE`: not reached when connected product and related service are confirmed `NO`;
- `ai-act-essential-services-credit`: recognized but unsupported.

The unsupported message states that the prototype does not yet implement the dedicated AI Act credit/essential-services classification route. It does not state that the AI Act is inapplicable.

### 8.3 GDPR follow-ups

After the user confirms the GDPR module, ask:

1. Are personal data processed when assessing the applicant?
   - target: `data_protection.personal_data_processed`;
2. Is the decision made solely through automated processing?
   - target: `data_protection.automated_individual_decision`;
3. Does the decision produce legal effects or similarly significantly affect the individual?
   - target: `use_context.materially_influences_decision`;
4. Is there meaningful human involvement before the decision takes effect?
   - target: `use_context.human_review_before_effect`;
   - contextual for future full Article 22 analysis, not a required fact of the current trigger rule.

Human review available only after an objection should not automatically be treated as meaningful involvement before effect. The user records the factual timing; the preliminary rule still evaluates only its current required facts.

### 8.4 Result behavior

If the three current GDPR rule requirements are complete, the preliminary GDPR finding may be produced. The unsupported AI Act credit route remains visible separately. If the implemented GDPR module cannot be completed, show:

> No substantive assessment was produced. Additional facts or an implemented assessment module are required.

Chinese:

> 尚未形成实质性评估结论。需要补充事实，或当前场景尚需相应的评估模块支持。

This message replaces any zero-findings copy that could be understood as zero legal risk.

## 9. Regression behavior

### Recruitment

- universal intake reaches the employment module from canonical employment context and confirmed recruitment functions;
- the GDPR module is separately suggested when personal data and individual decision signals are present;
- AI Act employment and GDPR questions use context-appropriate wording even where they share a canonical fact;
- Data Act questions are absent unless connected-product or related-service routing signals are present;
- existing demo findings and evidence remain unchanged.

### Industrial connected machinery

- connected-product or related-service signals reach the Data Act module;
- its three required facts drive follow-up questions;
- the employment module is not reached;
- GDPR is suggested only when personal-data and individual-decision routing facts make it plausible;
- the existing Data Act demo finding and evidence remain unchanged.

### Future product-safety route

The router definition format supports a future Article 6(1) module, but the UI must currently present it as unsupported when recognized. No product-safety rule, finding, or legal conclusion is added by this work.

## 10. Bilingual behavior

- Every module, question, option, help message, routing reason, validation message, unsupported message, and empty state uses centralized i18n keys.
- English and Simplified Chinese controls store identical canonical values.
- Original free text is preserved separately from confirmed routing tags.
- Changing language rebuilds display labels only; it does not recompute facts, selections, requirements, report ownership, or findings.
- Missing translation keys fall back safely to English and are detected by catalogue tests.
- Legal identifiers, citations, official excerpts, rule IDs, and stable Evidence IDs remain untranslated in technical/audit views.

## 11. Report and empty-state behavior

The report presentation should consume the execution/routing plan rather than infer module state from a zero-length findings list.

- confirmed completed modules show their findings;
- confirmed modules with missing facts show scoped information gaps;
- deselected modules show `Not assessed — user did not select this module`;
- unsupported modules show their localized limitation message;
- unrelated modules do not create mandatory-looking information gaps;
- technical details preserve raw rule IDs, missing paths, routing reasons, and registry versions.

When there is no substantive finding, use the required neutral message:

- EN: `No substantive assessment was produced. Additional facts or an implemented assessment module are required.`
- ZH-CN: `尚未形成实质性评估结论。需要补充事实，或当前场景尚需相应的评估模块支持。`

## 12. Implementation plan

### 12.1 Files to add

| Proposed file | Responsibility |
|---|---|
| `src/assessment/questionnaire/routing_models.py` | Candidate-module states, eligibility conditions, route plan, invalidation records, unsupported definitions |
| `src/assessment/questionnaire/router.py` | Pure deterministic routing and three-valued hint evaluation |
| `src/assessment/questionnaire/definitions.py` | Declarative definitions for the three implemented modules and recognized unsupported routes |
| `src/assessment/questionnaire/invalidation.py` | Provenance-aware dependency graph and invalidation calculation |
| `src/ui/questionnaire.py` | Streamlit-neutral adapters from route plans to localized controls/view models |
| `tests/assessment/questionnaire/test_router.py` | Router and module-state tests |
| `tests/assessment/questionnaire/test_invalidation.py` | Dependency/provenance tests |
| `tests/ui/test_dynamic_questionnaire.py` | End-to-end Streamlit questionnaire routing tests |

### 12.2 Files to modify

| Existing file | Minimum change |
|---|---|
| `src/assessment/questionnaire/models.py` | Add localized/spec metadata or a compatible compiled-question representation; extend `QuestionnairePlan` with module context. |
| `src/assessment/questionnaire/registry.py` | Register module-aware question variants and validate definitions against `RuleRegistry`. |
| `src/assessment/questionnaire/engine.py` | Build plans from confirmed rule requirements only and apply dependency filters. |
| `src/assessment/questionnaire/__init__.py` | Export routing contracts. |
| `src/assessment/engine.py` | Later phase: optional ordered rule selection with `None` preserving current behavior. |
| `src/assessment/results.py` / assessment run model | Later phase: retain execution-plan and routing-version metadata if required for report audit. |
| `src/assessment/demo/factory.py` | Wire the router and definitions once; do not branch rules in the UI. |
| `assessment_app.py` | Replace the fixed form with universal intake, module confirmation, and generated follow-ups. |
| `src/ui/i18n.py` | Add bilingual prompts, routing states, unsupported messages, and neutral zero-finding copy. |
| `tests/assessment/questionnaire/test_engine.py` | Preserve existing foundation behavior and add module-aware compatibility tests. |
| `tests/ui/test_assessment_app.py` | Keep scenario isolation, language, report ownership, and evidence regressions. |

No changes should be needed in legal rule implementations, evidence services, retrievers, report builders, corpus artifacts, or official source documents.

### 12.3 Migration path

1. Register the three current rules in the companion definition registry without changing execution.
2. Add router models and pure routing tests.
3. Convert the existing three fixed controls into universal intake controls while keeping the same fact paths and canonical values.
4. Add explicit structured personal-data, automated-decision, connected-product, related-service, and data-generation questions.
5. Drive rule-specific follow-ups from confirmed modules and `FactRequirementValidator`.
6. Add provenance-aware invalidation and stale-report clearing.
7. Add optional selection-aware execution while retaining `run(facts)` as the backward-compatible default.
8. Add unsupported AI Act credit/essential-services and judicial-path messages.
9. Switch report empty-state and module-status presentation to the route/execution plan.
10. Keep Recruitment and Industrial demos preconfigured so their current one-click flows remain stable.

### 12.4 Phased milestones

**Milestone 1 — Registry and pure router (medium)**

- routing models and declarative definitions;
- bilingual question/message keys;
- candidate and unsupported module calculation;
- no Streamlit changes yet.

**Milestone 2 — Dynamic custom-case questionnaire (high)**

- universal intake;
- user module confirmation;
- missing-fact follow-ups;
- provenance-aware invalidation;
- custom loan acceptance flow.

**Milestone 3 — Selection-aware runs and reporting (medium-high)**

- immutable execution-plan snapshot;
- selected-rule execution with backward-compatible defaults;
- supported/deselected/unsupported report presentation;
- neutral zero-substantive-finding behavior.

**Milestone 4 — Regression and UX hardening (medium)**

- Recruitment and Industrial regression fixtures;
- bilingual and accessibility checks;
- stale-state and report ownership tests;
- unsupported judicial and product-safety scenarios.

### 12.5 Backward compatibility

- Existing `Question`, `QuestionRegistry`, and `QuestionnaireEngine` behavior remains available during migration.
- `AssessmentEngine.run(facts)` continues to mean all registered rules unless an explicit execution plan is supplied.
- Demo fixtures are not rewritten merely for presentation or routing.
- Existing facts, provenance, findings, evidence bindings, citations, and IDs are unchanged.
- Unknown facts still skip legal rules rather than creating findings.
- Language switching remains presentation-only.

## 13. Test matrix

| Scenario | Routing expectation | Questionnaire expectation | State/report expectation |
|---|---|---|---|
| Recruitment demo | AI Act employment confirmed; GDPR independently suggested/confirmed when facts support it | Employment questions plus GDPR facts when relevant; no Data Act questions | Existing findings, seven-record AI Act evidence regression, and report ownership unchanged |
| Industrial demo | Data Act confirmed | Connected product, related service, generated data; no employment question | Data Act remains primary; Data Act evidence binding unchanged |
| Custom loan case | GDPR suggested; AI Act credit route unsupported; employment not reached | Ask personal data, solely automated decision, significant effect, and human involvement | No full AI Act conclusion; neutral empty state until an implemented module completes |
| Unsupported judicial scenario | Recognize unsupported AI Act justice route; optionally suggest GDPR only from independent data/decision facts | No employment or Data Act questions | Show limitation, not `does_not_apply` or compliant |
| Domain employment → essential services | Remove employment candidate | Invalidate provenance-linked employment decision question and hide employment follow-up | Clear stale report/selection; preserve language and unrelated facts |
| Ambiguous Chinese task | No unconfirmed task tag | Show clarification/confirmation; no silent candidate activation | No positive finding from ambiguous text |
| Unknown required facts | Confirmed module remains active | Ask only its mapped missing facts in deterministic order | No finding until requirements complete; missing reasons preserved |
| Language switch | Route plan unchanged | All prompts/options relocalize with identical canonical values | Preserve case, facts, module selection, report, evidence, and navigation |
| Scenario switch | Recompute from new fixture | Old module questions disappear | Preserve language; clear old case report, bundle-derived state, and selected finding |
| Stale-report injection | Route/case ownership mismatch detected | N/A | Reject report before rendering |
| Equivalent EN/ZH loan answers | Same canonical routing facts and confirmed module IDs | Labels differ only in presentation | Identical GDPR outcome and missing-fact plan |
| Controlled task text replaced | Old mapping provenance invalidated | Questions reflect only new confirmed tags | No stale positive facts or findings |

## 14. Acceptance criteria

The design is successfully implemented when:

1. a custom loan case never receives employment-specific wording;
2. GDPR Article 22 is suggested and its missing facts are explicitly collected;
3. the unsupported AI Act credit route is disclosed without a legal conclusion;
4. Recruitment and Industrial demos retain their current outcomes and evidence;
5. upstream changes invalidate only provenance-linked dependent answers;
6. module confirmation and routing versions are auditable;
7. English and Chinese inputs produce identical canonical facts and route plans;
8. zero findings use the neutral no-substantive-assessment message;
9. no LLM participates in formal routing;
10. no legal rule contains UI branching or translation logic.
