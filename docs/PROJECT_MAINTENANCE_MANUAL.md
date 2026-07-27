# EU Digital Regulation Assessment Platform
# Project Maintenance Manual

## Document metadata

- **Document version:** 2.0
- **Applies to application release:** `v0.5.0-prototype`
- **Application release commit:** `83c772349c0b14c0747bafcaf4f929780be66520`
- **Drafted against repository baseline:** `290534d385eb6366986ba24257f22700634abca2`
- **Last verified date:** 2026-07-27
- **Production URL:** https://eu-regulatory-assessment.streamlit.app/

This manual describes the deterministic assessment application shipped in
`v0.5.0-prototype`. It is a maintenance reference for the current prototype,
not a statement that the platform provides complete regulatory coverage.

## 1. Document control

This Markdown file is the canonical operational manual. Update it whenever a
change affects any of the following:

- the production entry point or supported runtime;
- registered rule IDs, rule versions, or required facts;
- questionnaire routing or module-confirmation behavior;
- Finding, report, or Evidence serialization;
- committed runtime Evidence packs or manifests;
- deployment, test, release, or rollback procedures; or
- the stated legal scope and limitations.

### Implementation source of truth

1. application code and versioned rule configuration;
2. automated tests and scenario fixtures;
3. Evidence manifests and runtime Evidence packs;
4. this manual;
5. the repository `README.md`; and
6. archived historical documents.

Code and tests determine what the software currently does. They do not
override, amend, or conclusively establish what the law requires.

### Legal-authority hierarchy

1. official legal instruments and effective amendments identified through
   authoritative EU sources;
2. authoritative guidance, decisions, and case law when incorporated into the
   reviewed legal corpus;
3. reviewed repository representations or excerpts of official legal sources
   identified through manifests and runtime Legal Evidence;
4. encoded rules and tests; and
5. explanatory documentation.

Repository Legal Evidence has no independent legal authority. Its authority
derives from the identified original legal source, and any discrepancy must be
resolved in favor of the authoritative original source.

Legal-source drift can make a technically passing release legally outdated.
Maintainers must therefore verify both implementation behavior and the
continuing authority and currency of the legal sources on which it relies.

The test counts in this document are a dated release snapshot. They must be
updated after verification and must never be treated as a permanent invariant.

## 2. Product purpose and legal scope

The platform is a legal-engineering prototype for structured preliminary
screening under selected pathways of the EU AI Act, GDPR, and EU Data Act.
Users provide facts about an AI system and its use. Versioned rules evaluate
those facts and produce traceable, framework-specific Findings.

The formal result is determined by code, not by a language model:

> **Rules decide. Evidence supports. LLM explains.**

No language model participates in the released formal assessment result.
“LLM explains” describes a bounded future explanatory role. Any such feature
must remain downstream of canonical Findings and Legal Evidence, must be
clearly presented as non-authoritative, and cannot alter facts, authorization,
Findings, statuses, reason codes, legal bases, or Legal Evidence bindings.

The present maturity level is **Level 1 relevance and classification
screening**. The platform does not provide:

- complete coverage of the EU AI Act, GDPR, or EU Data Act;
- a complete role, obligation, exception, or safeguard analysis;
- certification, assurance, or conformity assessment;
- a final compliance determination; or
- legal advice.

A negative Finding is limited to the predicate evaluated by its named rule. It
does not establish that no other regulatory pathway applies.

## 3. Supported regulatory modules

The shared workflow factory registers four rules, each at version `2026.1`.

| Framework | Rule ID | Current screen | Principal legal basis |
|---|---|---|---|
| EU AI Act | `AI_ACT_HIGH_RISK_EMPLOYMENT` | Preliminary employment-related high-risk pathway under Article 6(2) | Article 6(2); Annex III point 4(a) |
| EU AI Act | `AI_ACT_HIGH_RISK_PRODUCT_SAFETY` | Article 6(1) product or safety-component route | Article 6(1)(a)-(b); selected Annex I point; Article 3(14) when relevant |
| GDPR | `GDPR_ARTICLE22_RELEVANCE` | Preliminary Article 22(1) relevance screen | Article 22(1) |
| EU Data Act | `EU_DATA_ACT_RELEVANCE` | Connected-product or related-service scope/relevance screen | Article 1(1)(a); Article 2(5)-(6) |

The employment rule implements one preliminary Article 6(2) pathway read with
Annex III point 4(a). Its current authored `legal_basis` retains the broader
`Article 6` reference alongside the atomic Annex reference. The module does
not implement a complete analysis of every Article 6(3) derogation, profiling
consequence, provider documentation obligation, or other Article 6 high-risk
route. A negative Finding therefore excludes only this implemented
employment-screening predicate, not all Article 6 pathways.

The product-safety rule checks encoded confirmations for Article 6(1)(a) and
(b), the selected Annex I catalogue instrument, and Article 3(14) when the
safety-component branch is relied upon. It does not interpret the full
underlying product legislation or perform a conformity assessment. A
predicate result records the supplied confirmations; it does not independently
determine the underlying product-law position.

The GDPR rule screens preliminary relevance under Article 22(1). It does not
determine Article 22(2) exceptions, Article 22(3) safeguards, Article 22(4)
special-category restrictions, lawful basis, transparency duties, DPIA
requirements, or overall GDPR compliance.

The Data Act rule is a preliminary scope/relevance screen for the implemented
connected-product and related-service pathway. It does not determine a
specific access entitlement, data-holder duties, user or third-party role
allocation, trade-secret treatment, contractual validity, compensation, or
full Chapter II compliance.

These modules are independent. A multi-framework case may produce more than
one Finding, and each Finding keeps its own facts, reasons, legal basis, and
Evidence binding.

The router may disclose an unsupported route for audit and product
transparency. An unsupported route does not become an executable rule and
cannot enter the canonical report.

## 4. Release and runtime baseline

The verified release baseline is:

| Item | Value |
|---|---|
| Release | `v0.5.0-prototype` |
| Application release commit | `83c772349c0b14c0747bafcaf4f929780be66520` |
| Drafted against repository baseline | `290534d385eb6366986ba24257f22700634abca2` |
| Production branch | `main` |
| Production entry point | `assessment_app.py` |
| Python | 3.12 |
| Streamlit | 1.56.0 |
| Production database | None |

The deterministic assessment workflow requires:

- no API key;
- no runtime Chroma dependency;
- no external corpus directory at runtime; and
- no required outbound call from the assessment logic to an LLM API, remote
  vector database, remote legal corpus, or other external decision service.

The hosted Streamlit application is nevertheless accessed over a network, and
user inputs are transmitted to and processed within its hosting environment.
The absence of external model or API calls is not equivalent to offline,
private, or confidential execution.

Cases, workflow history, and reports are held in memory. Restarting the
Streamlit process clears that state. Do not describe the current prototype as
providing persistent case management.

## 5. System architecture

The released application follows this flow:

```text
User or demo facts
        ↓
AssessmentCaseService
        ↓
QuestionnaireRouter + QuestionnaireEngine
        ↓
Explicit module confirmation
        ↓
Authorized rule_ids
        ↓
AssessmentWorkflowService
        ↓
AssessmentEngine + FactRequirementValidator
        ↓
Canonical Findings / missing requirements / failures
        ↓
EvidenceService
        ↓
ReportBuilder
        ↓
AssessmentReport + per-Finding Evidence Trace
```

`src/assessment/demo/factory.py` is the composition root used by the
Streamlit application and demonstrations. It wires:

- `AssessmentCaseService`;
- `RuleRegistry`;
- the four registered rules;
- `AssessmentEngine`;
- `MultiCorpusLegalEvidenceRetriever`;
- `InMemoryEvidenceService`;
- `ReportBuilder`; and
- `AssessmentWorkflowService`.

The layers must remain separate:

- questionnaires collect facts;
- routing identifies implemented and unsupported paths;
- users authorize implemented modules;
- rules decide formal preliminary outcomes;
- Evidence services resolve authored legal references;
- reports preserve the canonical audit record; and
- the UI presents but does not redefine domain results.

## 6. Assessment facts, TriState values, and provenance

`AssessmentFacts` is the canonical typed input model. Its schema version is
`2.0.0`. It separates facts into these namespaces:

- `case`;
- `system`;
- `scope`;
- `organisation`;
- `supply_chain`;
- `use_context`;
- `practices`;
- `high_risk`;
- `product_regulation`;
- `data_protection`; and
- `data_act`.

Legal booleans use `TriState`:

- `yes` — the fact is affirmatively confirmed;
- `no` — the fact is affirmatively rejected; and
- `unknown` — the user cannot currently confirm either answer.

`unknown` is an answer, not the same thing as an unanswered widget. Requirement
validation treats unresolved required facts explicitly and prevents a rule
from inventing facts.

`FactProvenance` records where an answer came from, including the question ID
and recording time where available. Presentation translations and normalized
input labels must not alter the canonical fact value.

When adding a fact:

1. add it to the correct namespace with a backward-compatible default;
2. preserve JSON serialization;
3. define provenance behavior;
4. add questionnaire definitions only where the fact is collected;
5. declare it in rule requirements only where legally necessary; and
6. add serialization, unknown-value, invalidation, and fixture regressions.

## 7. Case and assessment-run lifecycle

An `AssessmentCase` stores the current mutable facts for one case. The
`AssessmentCaseService` currently uses in-memory storage and returns deep
copies to preserve service boundaries.

An `AssessmentRun` is a separate execution record that is treated as immutable
after creation and storage. The model is not a frozen Python object: the
workflow completes its status, Findings, and timestamps before storing a deep
copy. It contains:

- the case ID;
- a deep-copied facts snapshot;
- ruleset and questionnaire versions;
- authorized rule IDs;
- an input fingerprint;
- status and timestamps;
- Findings; and
- an error message when execution fails.

The workflow fingerprint is a SHA-256 digest over:

- canonical serialized facts;
- authorized rule IDs; and
- engine version.

This fingerprint allows the UI to identify a stale report after facts, engine
version, or module authorization changes. It does not hash the rule source,
each rule version, or the legal corpus independently. Stored runs are isolated
through deep copies, and assessment runs and reports must never be silently
rewritten when a case is edited.

Operationally, the fingerprint does not independently hash rule source,
individual rule versions, Evidence-pack versions, manifests, or the legal
corpus. Changing only those assets may therefore fail to invalidate a
previously stored report automatically. A change to rule behavior, execution
semantics, or the reviewed legal-source baseline must either increment the
engine version or introduce an explicit report invalidation or migration
mechanism. Maintainers must not rely on the fingerprint alone to establish
report validity across releases.

## 8. Questionnaire routing and explicit module confirmation

Questionnaire routing is deterministic and UI-neutral. Definitions live under
`src/assessment/questionnaire/` and map questions to canonical fact paths.

The router:

- evaluates declarative eligibility hints;
- distinguishes supported modules from unsupported informational paths;
- returns missing required questions in deterministic order;
- reflects confirmed and declined modules; and
- exposes routing state for progress and audit presentation.

Routing is not legal adjudication. A routing suggestion means that an
implemented module may be worth assessing. It does not itself create a
Finding.

Before execution, the user must explicitly confirm the implemented modules
that may run. Confirmation is distinct from answering the module's factual
questions.

Dependency invalidation clears downstream answers, module state, and stale
reports when an upstream fact changes. Language changes preserve the active
case, route, report, and selected Finding because localization is presentation
state rather than legal input.

## 9. Authorized rule execution scope

The UI converts confirmed modules into an explicit ordered set of
`authorized_rule_ids`. The workflow passes that scope to the engine.

The engine:

- rejects unknown or duplicate rule IDs;
- resolves requested IDs in registry order;
- executes only authorized rules;
- records the authorized IDs in the result and report; and
- includes authorization in the run fingerprint.

An explicit empty scope executes no rules. Omitting a scope preserves the
full-registry behavior for non-UI callers, so UI callers must always pass the
confirmed scope.

A rule that was suggested but not confirmed cannot emit a canonical Finding.
A screened-out or unsupported route remains outside the report.

## 10. Deterministic rule engine

`AssessmentRule` defines the contract for every legal rule:

- `framework`;
- `rule_id`;
- `version`;
- `category`;
- `required_fact_paths`;
- `legal_basis`;
- `required_fact_paths_for(facts)`; and
- `evaluate(facts) -> Finding`.

Rules are evaluated in stable registry order. Each rule receives a deep copy
of the fact snapshot. A rule must not mutate the case or hide questionnaire
behavior.

The `FactRequirementValidator` runs before legal evaluation. Missing facts
produce structured `RuleRequirementResult` data and no substantive Finding.
The product-safety rule uses transparent conditional requirements for its OR
predicate; other rules retain static requirement behavior.

One rule failure is captured as `RuleExecutionFailure` without crashing later
rules. The engine validates that a returned Finding matches the registered
rule's category, framework, ID, and version.

When changing a rule:

1. state the legal route narrowly;
2. update the rule version when behavior changes;
3. preserve atomic legal references;
4. ensure negative conclusions are limited to the evaluated route;
5. cover positive, negative, unknown, inconsistent, and malformed inputs;
6. test required-fact ordering and non-mutation;
7. test cross-framework isolation; and
8. update this manual and the public scope description; and
9. increment the engine version or add an explicit report invalidation or
   migration mechanism when rule behavior, execution semantics, or the reviewed
   legal-source baseline changes.

## 11. Canonical Findings and AssessmentReport

A `Finding` is the formal output of one rule. It includes:

- framework and category;
- issue code and status;
- title and summary;
- rule ID and version;
- fact references and missing-fact references;
- reason codes;
- legal basis;
- review status;
- trace entries; and
- Evidence references.

Each status is scoped to one implemented rule and rule version:

| Status | Internal rule-engine meaning | What it does not establish legally |
|---|---|---|
| `applies` | The rule has returned its affirmative `applies` state for the evaluated predicate. The domain schema makes this state available, although none of the four current rules returns it. | It does not turn this prototype into a final legal decision system or establish overall compliance, liability, or every consequence of the cited law. |
| `does_not_apply` | Complete facts establish that the named rule's encoded positive predicate is not satisfied. | It is not an overall no-risk result and does not exclude another rule, route, exception analysis, or regulatory framework. |
| `potentially_applies` | The named rule's preliminary positive-screen conditions are satisfied. This is the affirmative status used by the current relevance and classification screens. | It is not a final applicability, non-compliance, certification, or legal-advice conclusion. |
| `undetermined` | The rule has identified an inconsistent or unresolved legal-fact state for which it deliberately returns a Finding rather than a positive or negative screen. | It does not resolve the missing or inconsistent facts in either direction. |
| `not_assessed` | No canonical assessment was performed for that route. The domain schema can represent this state, although the current registered rules do not return it as a substantive result. | It is not a negative Finding and does not mean the route is irrelevant or inapplicable. |

Missing required facts ordinarily remain structured missing information and
produce no substantive Finding. Availability of `applies` and `not_assessed`
in the domain schema does not expand the implemented legal scope.

`ReportBuilder` deterministically projects an `AssessmentResult` and
`EvidenceServiceResult` into an `AssessmentReport`. The report includes:

- a deterministic report ID;
- assessment-run reference and generation time;
- summary;
- canonical Findings;
- Evidence and Finding-to-Evidence bindings;
- missing information;
- recommendations;
- engine and report versions;
- rule-version metadata;
- execution failures;
- Findings grouped by framework;
- assessed frameworks; and
- authorized rule IDs.

The builder validates that every binding points to a known Finding and known
Evidence record. Presentation localization must not mutate
`AssessmentReport.to_dict()`.

## 12. Evidence architecture

Rules author legal issues and citations. The Evidence layer supplies supporting
authority. A language model is not part of this relationship.

In this documentation, **Legal Evidence** is an internal domain-model term. It
means a canonical record containing a reviewed excerpt or structured
representation of an identified authoritative legal source and bound to a
Finding. It does not by itself claim evidentiary status in judicial,
administrative, or other legal proceedings.

A **compliance artefact** or **compliance record** means an organizational
policy, log, assessment, contract, technical record, or other material offered
as proof of actual compliance activity. The current runtime `Evidence` records
are Legal Evidence in the internal sense above. Organizational compliance
records are not authoritative Legal Evidence merely because a future workflow
collects them.

`MultiCorpusLegalEvidenceRetriever` reads independent sources in deterministic
configured order:

1. the legacy `vector_store.json` compatibility asset;
2. the committed EU Data Act metadata-v2 runtime pack; and
3. the committed EU AI Act product-safety metadata-v2 runtime pack.

The retriever:

- resolves legal-source aliases through `config/legal_sources.json`;
- matches instrument and citation metadata;
- expands supported compound citations into atomic citations;
- prefers exact metadata-v2 Evidence over broad legacy records;
- deduplicates by Evidence ID;
- preserves configured ordering; and
- fails closed against malformed or cross-instrument references.

`InMemoryEvidenceService` indexes the configured Evidence by normalized legal
source and atomic citation. It creates `FindingEvidenceBinding` objects without
changing the Finding's authored legal basis.

Each Evidence record exposes:

- `evidence_id`;
- legal source;
- canonical citation;
- excerpt;
- document version; and
- authority level.

The UI presents one Evidence Trace per selected Finding:

```text
Facts
  ↓
Rule application
  ↓
Legal basis
  ↓
Bound Evidence
```

## 13. Runtime Evidence-pack lifecycle

Runtime packs are committed, embedding-free legal Evidence records required by
the clean-checkout application:

- `data/legal_evidence/eu_ai_act_product_safety_metadata_v2.json`
  contains 23 atomic EU AI Act records.
- `data/legal_evidence/eu_data_act_relevance_metadata_v2.json`
  contains 3 atomic EU Data Act records.

Official-source downloads belong in ignored `corpus_sources/`. Generated
candidate corpora belong in ignored `corpus_builds/`. Neither directory is a
production runtime dependency.

Normal candidate builds validate against the committed runtime pack and do not
rewrite it:

```bash
python scripts/build_ai_act_product_safety_candidate_corpus.py \
  corpus_sources/EU_AI_ACT/Regulation_2024_1689.html \
  --output corpus_builds/EU_AI_ACT/product_safety_candidate.json

python scripts/build_data_act_candidate_corpus.py \
  corpus_sources/EU_DATA_ACT/Regulation_2023_2854.html \
  --output corpus_builds/EU_DATA_ACT/data_act_candidate.json
```

Promotion is a reviewed operation:

1. acquire the official EUR-Lex source;
2. confirm the legal-source catalog mapping;
3. build an isolated candidate;
4. inspect citations, excerpts, definitions, and provenance;
5. compare the candidate with the existing runtime pack;
6. obtain substantive legal review for intended changes;
7. regenerate the runtime pack only with the explicit
   `--write-runtime-pack` option;
8. update the matching manifest in the same reviewed change;
9. run Evidence, scenario, UI, and full-suite regressions; and
10. verify a clean checkout without source or candidate directories.

Never replace `vector_store.json` as part of a metadata-v2 pack build.

## 14. Stable Evidence IDs, manifests, hashes, and provenance

Metadata-v2 stable identity is content based. It uses:

```text
instrument_id
+ document_version
+ canonical_citation
+ SHA-256(normalized excerpt)
```

The resulting Evidence ID is namespaced as:

```text
evidence:v2:<first 32 hex characters of identity digest>
```

The source-record ID uses the same digest with the
`legal-chunk:v2:` namespace. Excerpt normalization uses NFC Unicode,
normalizes line endings and whitespace, and preserves punctuation.

Each runtime manifest binds:

- instrument ID;
- document version;
- official source URI and CELEX where applicable;
- canonical citation;
- stable Evidence ID; and
- authoritative excerpt hash.

Runtime loaders validate pack structure, instrument isolation, citations,
stable identities, excerpt hashes, provenance, and manifest equality before
the shared workflow is exposed. Drift must fail explicitly rather than being
silently accepted.

## 15. Local installation and execution

Use a clean Python 3.12 environment:

```bash
git clone https://github.com/miles2005-web/eu-ai-act-legal-rag-prototype.git
cd eu-ai-act-legal-rag-prototype
git switch main
python3.12 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

On Windows PowerShell, activate with:

```powershell
venv\Scripts\Activate.ps1
```

Launch the production assessment entry point:

```bash
streamlit run assessment_app.py
```

Run the recruitment CLI demonstration:

```bash
python scripts/run_demo_assessment.py
```

The deterministic application should start with committed repository assets
and without application secrets or required outbound calls to an LLM API,
remote vector database, remote legal corpus, or external decision service.

## 16. Test architecture and regression matrix

The test suite covers:

- facts, serialization, provenance, and backward compatibility;
- rule metadata, predicates, negative scope, and missing facts;
- authorized engine scope and failure isolation;
- questionnaire definitions, routing, progress, and invalidation;
- case and workflow snapshot behavior;
- deterministic reports and framework grouping;
- citation parsing, retrieval isolation, stable IDs, manifests, and packs;
- bilingual normalization and presentation;
- Streamlit state isolation and Evidence Trace; and
- candidate corpus parsing and validation.

Canonical commands:

```bash
python -m compileall -q assessment_app.py src scripts tests
python -m unittest discover -s tests -p "test_*.py" -q
python -m unittest discover -s tests/ui -p "test_*.py" -q
git diff --check
```

Release snapshot verified on **2026-07-27**:

- **289/289 total tests passed**
- **92/92 UI tests passed**

These counts describe the named release only. Future changes may legitimately
change the counts and must record a new dated snapshot.

Current scenario regression matrix:

| Scenario | Expected canonical result |
|---|---|
| Recruitment AI Screening | 1 Finding / 7 Evidence |
| Industrial AI Monitoring | 1 Data Act Finding / 3 Evidence |
| Industrial Robot Safety and Data Access | 2 independent Findings / 4 AI Act Evidence + 3 Data Act Evidence |
| GDPR lending acceptance | `GDPR_ARTICLE22_RELEVANCE` / `potentially_applies` |

Scenario regressions must also confirm:

- no report is reused across cases;
- language switching preserves the active case and canonical report;
- unsupported routes do not block supported-module completion;
- each Finding binds only its own framework's Evidence; and
- official excerpts, citations, and stable Evidence IDs do not change through
  presentation localization.

## 17. Streamlit staging and production deployment

Production settings:

- repository: `miles2005-web/eu-ai-act-legal-rag-prototype`;
- branch: `main`;
- entry point: `assessment_app.py`;
- Python: 3.12; and
- Streamlit: 1.56.0.

No application secret is required for deterministic assessment. Runtime assets
that must be present include:

- `vector_store.json`;
- `config/legal_sources.json`;
- `config/ai_act_annex_i_instruments.json`;
- both Evidence manifests;
- both metadata-v2 runtime packs; and
- the public demo fixtures.

Before production deployment:

1. install from `requirements.txt` in a clean environment;
2. run `pip check`;
3. compile the application, source, scripts, and tests;
4. run the full and UI suites;
5. run all scenario regressions;
6. launch the Streamlit app;
7. verify English and Simplified Chinese switching;
8. verify each demo and Evidence Trace; and
9. confirm the deployment uses the intended commit.

A staging deployment should use the same entry point and dependency file. It
must not depend on local ignored assets.

## 18. Branch, PR, release, and tag workflow

Use focused branches and preserve reviewable history:

1. fetch `origin`;
2. fast-forward local `main`;
3. create a scoped branch from `origin/main`;
4. modify only files within the stated milestone;
5. run targeted and full validation;
6. inspect `git diff --check` and staged scope;
7. commit with a focused message;
8. push only the current branch;
9. open a Pull Request against `main`;
10. confirm expected HEAD, changed files, mergeability, and checks;
11. merge using the reviewed repository strategy; and
12. fast-forward local `main` after merge.

For a release:

1. verify the exact production commit;
2. confirm that changes to rule behavior, execution semantics, or the reviewed
   legal-source baseline increment the engine version or include an explicit
   report invalidation or migration mechanism;
3. create an annotated tag on that commit;
4. push only the tag;
5. publish a non-draft GitHub Release with validation and scope notes;
6. verify the deployment commit and runtime assets; and
7. preserve the feature branch unless deletion is explicitly approved.

Current release:

- tag: `v0.5.0-prototype`;
- application release commit:
  `83c772349c0b14c0747bafcaf4f929780be66520`; and
- release page:
  https://github.com/miles2005-web/eu-ai-act-legal-rag-prototype/releases/tag/v0.5.0-prototype

## 19. Rollback and recovery

Treat code, rule versions, manifests, and runtime packs as one reviewed
baseline. When moving between baselines, do not treat a matching input
fingerprint as proof that an older report remains valid. If rule behavior,
execution semantics, or the reviewed legal-source baseline changed without an
engine-version increment, apply an explicit report invalidation or migration
mechanism before restoring or presenting stored reports.

If production fails after deployment:

1. record the deployed commit and error;
2. verify required tracked assets exist;
3. reproduce from a clean checkout;
4. distinguish application failure from stale Streamlit session state;
5. compare the change with the last known-good release;
6. revert through a reviewed commit or redeploy the known-good commit;
7. do not rewrite or move an existing release tag; and
8. rerun the complete release verification.

If an Evidence pack fails validation:

1. do not bypass the loader;
2. compare the pack with its manifest;
3. verify citation, excerpt hash, stable ID, instrument ID, and provenance;
4. rebuild a candidate from the official source where available;
5. compare candidate and runtime records; and
6. promote only after substantive review.

If a report appears under the wrong scenario, clear case-owned UI state and
verify its input fingerprint and assessment-run reference. Never present a
report whose fingerprint no longer matches the active case and authorized
rules.

## 20. Troubleshooting

### Application does not start

- Confirm Python 3.12 and Streamlit 1.56.0.
- Run `python -m pip check`.
- Confirm all runtime assets listed in section 17 exist.
- Run the factory and runtime-pack tests before changing code.

### A confirmed module produces no Finding

- Inspect the route's missing fact requirements.
- Confirm `authorized_rule_ids` contains the module rule ID.
- Check whether the run was deliberately executed with gaps.
- Inspect execution failures in the report.
- Do not convert a missing-fact result into a legal Finding.

### A Finding has no Evidence

- Inspect the Finding's instrument and authored citation.
- Confirm the citation is atomic or supported by compound expansion.
- Confirm the legal source catalog resolves the instrument.
- Verify the relevant pack or compatibility record exists.
- Run manifest and multi-corpus isolation tests.
- Do not substitute Evidence from another framework.

### The UI shows a previous scenario's report

- Confirm case-dependent state was reset on scenario change.
- Compare the report fingerprint with the active case.
- Confirm the selected Finding belongs to the current report.
- Run scenario-isolation UI regressions.

### Language switching changes legal output

- Compare `AssessmentReport.to_dict()` before and after the switch.
- Check input normalization and presentation-only translation.
- Verify official excerpts, citations, and Evidence IDs remain unchanged.

### A runtime pack changed unexpectedly

- Stop promotion.
- Compare it with the committed manifest and official-source candidate.
- Review normalized excerpt hashes and stable identities.
- Restore through a reviewed Git change, not manual ID editing.

## 21. Security, privacy, and legal limitations

The released prototype has no user authentication, authorization, production
database, tenant isolation, or durable audit store. The public production
deployment must not be used for:

- personal data relating to real individuals;
- confidential business information;
- privileged legal material;
- client information;
- trade secrets; or
- production-sensitive assessment records.

The deterministic assessment logic makes no required outbound call to an LLM
API, remote vector database, remote legal corpus, or other external decision
service. The hosted Streamlit application is still accessed over a network,
and user inputs are transmitted to and processed within the hosting
environment. The absence of external model or API calls is not equivalent to
an offline, private, or confidential execution environment and is not a
complete security or privacy architecture.

Every user-facing result must retain the following boundaries:

- preliminary screening only;
- implemented pathways only;
- no certification;
- no final compliance determination;
- no final legal advice; and
- professional review required for consequential use.

Legal text, guidance, and implementation timelines can change. A runtime pack
records an identified source and document version; it does not guarantee that
the source remains the latest law. Source updates require authoritative legal
review and a new verified release.

## 22. Legacy RAG appendix

This section documents the historical retrieval prototype. It is not the
production assessment workflow.

The repository retains `app_chroma.py`, `run_pipeline_chroma.py`, ingestion
utilities, and the original vector asset for history and compatibility. The
historical application used:

- OpenRouter for embeddings and generated answers;
- `OPENROUTER_API_KEY` in local or hosted secrets;
- `ChromaDB` for local vector persistence;
- a JSON vector-store fallback for hosted retrieval;
- `Streamlit 1.44.1` during an earlier deployment phase;
- regex self-query routing for Article, Annex, and Recital references; and
- a chat interface in which an LLM generated the answer.

The historical `feature/assessment-engine-v2` branch was a development branch,
not the current deployment branch.

These historical utilities may require additional dependencies, source
documents, credentials, and network access. They are outside the standard
deterministic assessment installation.

The current application may read `vector_store.json` through a metadata-only
compatibility adapter for selected legacy Evidence. That compatibility role
does not restore the old chat architecture as the decision-maker.
