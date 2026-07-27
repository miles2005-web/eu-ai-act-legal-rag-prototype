# From Legal RAG to Deterministic Regulatory Assessment
# Project Development Story

## Document metadata

- **Document version:** 2.0
- **Historical source:** `docs/archive/PROJECT_DEVELOPMENT_STORY_v1.pdf`
- **Current release:** `v0.5.0-prototype`
- **Last verified date:** 2026-07-27

This document preserves the project's real chronology while explaining its
current architecture. The archived PDF records the original RAG project from
its minimum retrieval loop through its first chatbot deployment. Those early
engineering choices are historical phases, not descriptions of the current
production application.

**Editorial note:** Sections 1–24 use a repository-level historical and
technical narrative. Section 25 is intentionally written in the first person
as an application-material summary.

## 1. The original regulatory problem — historical phase

The project began with a practical legal-access question: how could a user
describe an AI system and find the provisions of the EU AI Act most relevant
to classification and compliance?

The law's structure made this harder than ordinary document search. Rules,
definitions, recitals, annexes, exceptions, and cross-references do not behave
like interchangeable passages. A result from the wrong annex can carry a
different legal function even when its language looks semantically similar.

The first project concept was therefore a compliance navigator. A user would
enter a natural-language description, the system would retrieve legal
materials, and a language model would generate a structured answer grounded in
those materials.

The historical source also discussed anticipated application dates. This
version does not repeat those dates as current legal claims because any current
timeline statement should be verified against an identified authoritative
source and the law in force at the time of publication.

## 2. Prior RAG experience and the first minimum loop — historical phase

The work did not start from zero familiarity with retrieval. Before this
repository, the project author had experimented with a local legal RAG setup
using Ollama, DeepSeek 14B, and AnythingLLM. That experience supplied a basic
understanding of vector retrieval, chunking, and prompt design, but it was a
general legal-question-answering system rather than a regulation-specific
workflow.

For this project, the first target was deliberately smaller: prove that one
explicit legal question could move through a complete local loop without
losing legal structure.

```text
Raw PDF
  ↓
Parsed text
  ↓
Legal-structure-aware chunks
  ↓
Local retrieval
  ↓
Streamlit display
```

The initial repository separated raw documents from parsed outputs, used
`src/ingest.py` for extraction, used `src/legal_chunks.py` for structure-aware
chunking, and kept the first `app.py` interface intentionally small. The
minimum loop had no embeddings and no language model. Keyword scoring and
rules were enough to test whether the legal structure survived.

Codex was already used in this phase as a terminal-based engineering assistant
for bounded edits, command execution, and regression checks. The important
practice was to define the goal and verification boundary before accepting a
suggested change.

## 3. Legal-structure-aware parsing and retrieval — historical phase

Generic fixed-length chunking was a poor fit for legislation. Splitting a
provision in the middle could separate a condition from an exception or detach
an annex item from its legal context.

The custom chunking work recognized boundaries such as:

- Chapter;
- Section;
- Article;
- Annex; and
- Recital.

Chunks carried fields such as article number, annex reference, recital
reference, chapter heading, and section heading. The historical source records
an early EU AI Act corpus of 730 structured chunks, later growing to 3,230
records after more materials were added.

The key achievement of this phase was not corpus size. It was a testable legal
structure baseline: explicit Article, Annex, and Recital queries could be
checked against identifiable legal units before semantic retrieval was added.

The limitation was equally clear. Keyword matching could retrieve an explicit
term but could not reliably connect a user's description of an HR screening
tool with concepts such as recruitment or the relevant high-risk route when
the same vocabulary was absent.

## 4. Vector retrieval, OpenRouter, and Chroma — historical phase

Semantic retrieval was introduced in the next phase. The prototype called an
embedding model through OpenRouter and initially attempted to use a dedicated
OpenRouter Python package.

That SDK path failed repeatedly: expected client classes and embedding methods
did not match the installed package. The working historical solution used the
OpenAI SDK with an alternative base URL instead of relying on the unstable
wrapper.

After embedding the early corpus, the prototype stored vectors locally in
ChromaDB. A test question about provider obligations retrieved Article 16
prominently, showing that semantic retrieval could bridge vocabulary
differences that keyword search missed.

The project then added an LLM generation step. The historical prompt required
specific citations, preservation of exceptions, cautious treatment of edge
cases, and confinement to retrieved context. At that point the system had
become a complete legal RAG loop:

```text
Question
  ↓
Semantic retrieval
  ↓
Retrieved legal context
  ↓
LLM-generated compliance answer
  ↓
Streamlit presentation
```

This was a meaningful prototype, but the language model still determined the
wording and practical conclusion of the answer.

## 5. Retrieval precision problems and self-query routing — historical phase

The historical retrieval system exposed an important legal-engineering
failure mode: semantically similar legal text may perform very different legal
functions.

Annex III and Annex IV became the clearest example. One concerns listed
high-risk use cases; the other concerns technical documentation. Their language
and formatting can look similar in vector space, but substituting one for the
other changes the legal analysis.

The project added a self-query layer before semantic retrieval:

1. detect explicit `Article`, `Annex`, or `Recital` references;
2. convert supported references into metadata filters;
3. search only the matching legal unit when a reference is present; and
4. use broader semantic retrieval when no explicit reference is present.

An early attempt asked a language model to extract references as JSON, but
hosted response-format behavior was unstable. The eventual historical solution
used regular expressions.

The archived PDF described this regex approach as “100% reliable.” The current
documentation does not preserve that absolute claim. The defensible statement
is narrower: regex routing was deterministic for the explicitly supported
reference syntax, removed a model call from the routing decision, and could be
covered by defined tests.

The historical prototype also introduced a dynamic token budget so long and
short chunks could be selected within a context allowance rather than by a
fixed number alone.

## 6. Productization of the original chatbot — historical phase

The next phase turned the retrieval form into a multi-turn chat interface.
`app_chroma.py` became the historical application entry point. Custom
Streamlit HTML and CSS created differentiated message bubbles, and generated
answers could be downloaded or regenerated.

The chatbot supported six presentation/output languages:

- English;
- French;
- German;
- Spanish;
- Simplified Chinese; and
- Traditional Chinese.

Language was injected into the model prompt. Existing answers did not change
automatically when the user switched language, so the interface added a
regeneration action.

This was productization of the RAG experience, not yet the structured
assessment architecture that exists today.

## 7. Historical deployment engineering and its limits — historical phase

Local vector persistence worked during development but failed in the original
hosted environment because ChromaDB and parts of its dependency stack were not
compatible with the available Python runtime.

The deployment fallback exported the corpus to `vector_store.json` and
implemented cosine similarity in Python. Two practical problems had to be
solved:

- array values needed conversion before JSON serialization; and
- the first export exceeded GitHub's file-size limit, so numeric precision and
  JSON formatting were reduced.

The historical application also depended on `OPENROUTER_API_KEY` for
embeddings and generated answers. Local shell state and hosted secrets created
their own failure modes.

The archived story records repeated virtual-environment failures caused by
Python-version drift and dependency churn. The durable lesson was to specify
the runtime version, build from a committed dependency file, and verify a
clean environment rather than repairing a polluted environment indefinitely.

These fixes made the chatbot deployable, but they did not solve the deeper
problem: the formal compliance answer was still generated probabilistically.

## 8. The architectural turning point

The decisive shift came from distinguishing two different tasks:

1. locating legal authority; and
2. deciding whether defined legal predicates are satisfied by known facts.

Retrieval can support the first task. It is not, by itself, a safe formal
decision mechanism for the second.

The project therefore changed direction:

### From

A legal RAG prototype that retrieves legal materials and asks a language model
to generate answers.

### To

A deterministic assessment platform in which versioned rules evaluate
structured facts and produce formal preliminary screening Findings,
authoritative legal Evidence supports those Findings, and language models do
not determine the formal result.

This was not a cosmetic UI rewrite. It changed the unit of work from a chat
message to an assessment case, the input from prose alone to canonical facts,
and the output from a generated answer to a serializable Finding and report.

## 9. “Rules decide. Evidence supports. LLM explains.”

The platform adopted a simple separation of responsibilities:

> **Rules decide. Evidence supports. LLM explains.**

“Rules decide” means formal screening outcomes come from inspectable,
versioned code operating on structured facts.

“Evidence supports” means each Finding identifies its legal basis and binds to
authority-specific Evidence without asking retrieval to invent the legal
issue.

“LLM explains” is a boundary for future work. No language model is used in the
released formal assessment result. A future explanatory layer may summarize or
translate a canonical result, but it must not alter facts, execution
authorization, Findings, statuses, reason codes, legal bases, or Legal Evidence
bindings.

## 10. Structured facts and the domain model

The v2 foundation introduced typed models for:

- `AssessmentFacts`;
- `AssessmentCase`;
- `AssessmentRun`;
- `Finding`;
- `AssessmentResult`;
- `Evidence`; and
- `AssessmentReport`.

Facts were divided into legal and operational namespaces rather than stored as
an undifferentiated prompt. Legal booleans use `yes`, `no`, and `unknown`.
Unknown information is explicit, so the engine can request missing facts
without guessing.

Fact provenance records how and when important answers were collected.
Existing scenarios remain serializable, and newly added GDPR, Data Act, and
product-regulation namespaces default safely for older fixtures.

Cases store current facts. A run's facts are deep-copied, and stored runs and
reports are treated as immutable historical snapshots after the workflow has
completed them. The `AssessmentRun` model itself is not a frozen Python object.
This separation made it possible to detect when an edited case had made an
earlier report stale.

## 11. Dynamic questionnaire routing

The first UI used a fixed, recruitment-oriented set of questions. That could
not support a general assessment workspace.

The dynamic questionnaire router introduced:

- question definitions mapped to fact paths;
- declarative eligibility conditions;
- deterministic supported-module suggestions;
- unsupported informational routes;
- missing-requirement routing;
- progress derived from the current route; and
- dependency-aware invalidation.

A change to an upstream fact can clear only the downstream answers and reports
that depend on it. This prevents a case from keeping a legally stale answer
after the scenario changes.

The router suggests where assessment may be useful. It does not itself make a
legal Finding.

## 12. Explicit module confirmation and execution authorization

Routing and legal execution were deliberately separated.

The user explicitly confirms the implemented modules that may enter an
assessment. Confirmed modules are converted into `authorized_rule_ids`, and
the engine executes only that scope.

Authorization is stored in the assessment result, report, run snapshot, and
input fingerprint. An unconfirmed module cannot emit a canonical Finding, even
if facts would satisfy its predicate.

This solved two related problems:

- unrelated rules no longer populated a report merely because they were
  registered; and
- the UI could show unsupported or screened-out routes for audit without
  implying that they had been legally assessed.

## 13. Versioned deterministic rules

The current factory registers four version `2026.1` rules:

- `AI_ACT_HIGH_RISK_EMPLOYMENT`;
- `AI_ACT_HIGH_RISK_PRODUCT_SAFETY`;
- `GDPR_ARTICLE22_RELEVANCE`; and
- `EU_DATA_ACT_RELEVANCE`.

Each rule declares:

- framework;
- ID and version;
- Finding category;
- required fact paths; and
- authored legal basis.

The rule engine validates requirements before evaluation, preserves stable
registry order, gives each rule a copy of the facts, and captures failures
without terminating unrelated rules.

The Article 6(1) product-safety rule introduced conditional requirements for
an OR predicate. One satisfied branch suppresses the unused branch, while
unresolved branches remain visible and deterministic. Negative results are
worded narrowly so they do not imply that all high-risk routes have been
excluded.

The employment rule is a preliminary implementation of the Article 6(2)
pathway read with Annex III point 4(a). Its current authored legal basis uses
the broader `Article 6` label plus the atomic Annex reference. It does not
complete every Article 6(3) derogation, profiling consequence, provider
documentation obligation, or alternative high-risk route; a negative Finding
is limited to its employment predicate.

The product-safety rule checks the encoded Article 6(1)(a)-(b) predicates, the
selected Annex I instrument, and Article 3(14) when the safety-component branch
is used. It does not interpret the full underlying product legislation or
perform a conformity assessment.

The GDPR rule screens Article 22(1) relevance. It does not determine Article
22(2) exceptions, Article 22(3) safeguards, Article 22(4) special-category
restrictions, lawful basis, transparency duties, DPIA requirements, or overall
GDPR compliance.

The Data Act rule screens the implemented connected-product and related-service
scope pathway using Article 1(1)(a) and Article 2(5)-(6). It does not determine
a specific access entitlement, data-holder duties, user or third-party role
allocation, trade-secret treatment, contractual validity, compensation, or
full Chapter II compliance.

## 14. Canonical Findings and reports

A Finding became the formal, inspectable result of one rule. It carries:

- status and framework;
- issue code;
- rule identity and version;
- facts and reason codes;
- legal basis;
- review status; and
- a reasoning trace.

Missing facts do not become legal Findings. They remain structured missing
information.

`ReportBuilder` produces a deterministic `AssessmentReport` containing
Findings, Evidence bindings, missing information, recommendations,
rule-version metadata, failures, assessed frameworks, and authorized rules.
Reports preserve framework grouping while keeping each Finding independently
selectable.

Presentation localization is downstream. The serialized canonical report is
identical before and after switching between English and Simplified Chinese.

## 15. Atomic legal Evidence and stable identities

Evidence modernization moved beyond broad retrieval chunks for the pathways
that required exact support.

Here, **Legal Evidence** is an internal domain-model term. It means a canonical
record containing a reviewed excerpt or structured representation of an
identified authoritative legal source and bound to a Finding. It does not by
itself claim evidentiary status in judicial, administrative, or other legal
proceedings.

This differs from a **compliance artefact** or **compliance record**: an
organizational policy, log, assessment, contract, technical record, or other
material offered as proof of actual compliance activity. The current runtime
`Evidence` records are Legal Evidence in the internal sense above; a future
organizational record would not become authoritative legal material merely
because the platform collected it.

Metadata-v2 records identify:

- legal instrument;
- document version;
- canonical citation;
- authority level;
- source record;
- stable Evidence ID; and
- official-source provenance.

Stable IDs are derived from the instrument, document version, canonical
citation, and normalized excerpt hash. Supported compound references, such as
an Article paragraph range, resolve to atomic Evidence records while the
Finding retains its originally authored legal basis.

Committed clean-checkout runtime packs provide:

- 23 atomic EU AI Act product-safety records; and
- 3 atomic EU Data Act relevance records.

Manifests bind citations, stable IDs, source metadata, and authoritative
excerpt hashes. Runtime loading fails when the pack and manifest drift.

## 16. Expansion from the EU AI Act to GDPR and the EU Data Act

The first deterministic rule screened employment-related high-risk relevance
under the EU AI Act.

The domain model then became regulation neutral. `RegulatoryFramework`
metadata was propagated through rules, Findings, missing requirements,
failures, results, and reports.

GDPR facts and an Article 22 relevance trigger were added without changing
existing AI Act behavior. Data Act facts and a connected-product/related-
service relevance trigger followed.

Multi-framework reports do not collapse these analyses into one generalized
compliance score. Each framework retains its own facts, predicate, status,
legal authority, and Evidence.

## 17. Recruitment, industrial, and multi-framework demonstrations

The release includes three public demonstrations:

### Recruitment AI Screening

An AI system screens and ranks job candidates and materially influences an
employment decision. The expected public result is one employment-screening
Finding with seven Evidence records.

### Industrial AI Monitoring

An AI-enabled system is connected to machinery, provides a related service,
generates operational data, and receives an external access request. The
expected public result is one Data Act Finding with three Evidence records.

### Industrial Robot Safety and Data Access

The scenario independently activates:

- the EU AI Act Article 6(1) product-safety route; and
- the EU Data Act relevance route.

The report contains two independent Findings, four AI Act Evidence records,
and three Data Act Evidence records.

A tested custom lending path also demonstrates a potentially applicable GDPR
Article 22 screen while disclosing an unsupported AI Act credit route without
letting that route block GDPR completion.

## 18. Bilingual compliance workspace

The current interface is not a chatbot. It is a Streamlit compliance workspace
with:

- landing and demo selection;
- structured facts and system profile;
- dynamic questionnaire modules;
- assessment results;
- independently selectable Findings; and
- per-Finding Evidence Trace.

English and Simplified Chinese presentation share the same canonical facts and
domain outputs. Controlled inputs are normalized deterministically, so
equivalent supported English and Chinese answers lead to identical rule
inputs.

Scenario state is isolated. Switching demos clears the previous case's report,
bundle, and selected Finding. Language switching preserves the active case and
assessment state.

## 19. Testing, clean-checkout verification, and release engineering

The project moved from informal retrieval checks to a layered test system:

- rule predicate and required-fact tests;
- engine scope and failure tests;
- questionnaire routing and invalidation tests;
- report determinism and traceability tests;
- Evidence isolation, citation, ID, manifest, and pack tests;
- scenario acceptance tests; and
- Streamlit presentation and state tests.

The dated release snapshot on 2026-07-27 recorded:

- 289/289 total tests passed; and
- 92/92 UI tests passed.

The release was also checked in a fresh Python 3.12 environment installed only
from `requirements.txt`. Compilation, dependency consistency, browser
language switching, scenario outputs, and clean-checkout runtime Evidence were
verified.

These counts describe the release snapshot. They are not permanent guarantees
for future commits.

## 20. Production deployment and v0.5.0-prototype

The deterministic application is deployed at:

https://eu-regulatory-assessment.streamlit.app/

The production entry point is `assessment_app.py` on `main`, using Python 3.12
and Streamlit 1.56.0.

The assessment workflow requires no API key or external corpus directory. Its
deterministic logic makes no required outbound call to an LLM API, remote
vector database, remote legal corpus, or other external decision service. The
Legal Evidence needed by the implemented atomic pathways is committed and
validated locally.

The hosted Streamlit application is still accessed over a network, and user
inputs are transmitted to and processed within its hosting environment. The
absence of external model or API calls is not equivalent to offline, private,
or confidential execution.

The application release commit is:

`83c772349c0b14c0747bafcaf4f929780be66520`

The release is published as:

`v0.5.0-prototype`

## 21. Current maturity and limitations

The current system is a Level 1 relevance and classification-screening
prototype.

It can identify whether confirmed facts satisfy one of four implemented
preliminary screens and show the legal authority supporting that result.

It does not yet provide:

- complete regulatory coverage;
- complete provider, deployer, controller, data-holder, user, or third-party
  role mapping;
- a complete obligation or exception analysis;
- certification or assurance;
- a final compliance determination; or
- legal advice.

An implemented rule's `does_not_apply` result is limited to that rule's route.
An unsupported route is not a negative legal conclusion.

## 22. Lessons in legal engineering

Several lessons survived both architectures.

First, legal structure is not optional metadata. Articles, paragraphs, points,
definitions, recitals, and annexes perform different functions and need
stable, inspectable identities.

Second, simpler deterministic mechanisms can be preferable when the input
syntax and legal purpose are bounded. The historical regex router and the
current versioned rule predicates reflect the same discipline at different
levels.

Third, retrieval quality and legal decision quality are different problems.
Better retrieval can improve access to authority but does not automatically
make a generated conclusion reproducible or auditable.

Fourth, missing information must remain visible. Treating unknown facts as
negative facts creates false certainty.

Fifth, AI-assisted engineering still requires verification. The historical
project used ChatGPT and Claude for architecture review and implementation
ideas, and Codex for repository-local execution and regression work. Some
model suggestions were useful; others proposed obsolete APIs, nonexistent
classes, or unnecessary complexity. Tests and source inspection remained the
deciding evidence.

Finally, scope language is part of the engineering. A narrowly correct
negative Finding is safer than a broad claim unsupported by the implemented
predicate.

## 23. Next stage: role, obligation, safeguard, and compliance-record consequence chains

The next stage is not simply to add more relevance triggers. It is to model the
legal consequences that follow an entry-screening result.

Planned areas include:

- AI Act provider and deployer roles;
- high-risk risk-management, data-governance, documentation, logging,
  transparency, human-oversight, robustness, and cybersecurity duties;
- GDPR controller/processor roles, lawful basis, Article 22 exceptions,
  safeguards, rights, transparency, and DPIA relevance;
- Data Act user, data-holder, and third-party roles, access rights, sharing,
  trade secrets, contracts, and compensation; and
- structured compliance artefacts or compliance records supplied by the
  assessed organization.

These are planned consequence chains, not current capabilities.

## 24. The limited future role of language models

A future language model could assist with:

- explaining a canonical Finding in plainer language;
- summarizing already-bound Evidence;
- translating non-authoritative presentation text;
- suggesting which documents a reviewer may want to collect; or
- drafting a narrative for professional review.

It must not:

- decide whether a rule applies;
- alter canonical facts;
- replace explicit module confirmation;
- invent legal authority;
- overwrite a Finding or stable Evidence identity; or
- present generated text as certification or final legal advice.

The canonical result must remain reproducible without the model.

## 25. Application-material narrative summary

I began with a concrete technology-law problem: legal rules are difficult to
navigate because their conditions, exceptions, definitions, and annexes are
distributed across a structured instrument. My first answer was a legal RAG
prototype. I built a minimum retrieval loop, designed a legal-structure-aware
chunker, added semantic retrieval, and deployed a multilingual chatbot that
generated answers from retrieved legal text.

That work revealed both the value and the limit of retrieval. The system could
locate relevant authority, but a language model still produced the practical
compliance conclusion. The result was difficult to reproduce, test, and audit.
Annex III/Annex IV confusion showed that semantic similarity alone could not
represent legal function.

I therefore redesigned the project around structured legal assessment.
Versioned rules now evaluate canonical facts, explicit user confirmation
controls which modules may run, and each result is preserved as a traceable
Finding. Authoritative Legal Evidence supports the Finding through stable,
source-aware identities. The interface presents independent EU AI Act, GDPR,
and Data Act analyses in English and Simplified Chinese without allowing
translation to change the legal result.

The central achievement is not that the system automates law. It is that the
system makes its limited legal reasoning inspectable: the facts are visible,
the rule is versioned, missing information remains unresolved, the legal basis
is explicit, and the Evidence can be traced. That is the transition from a
legal-information chatbot to a legal-engineering assessment prototype.
