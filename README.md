# EU Digital Regulation Assessment Platform

A deterministic, evidence-grounded legal engineering prototype for selected regulatory pathways under the EU AI Act, GDPR, and EU Data Act.

The project began as a legal retrieval experiment. It now treats retrieval as supporting infrastructure for structured legal assessment rather than as the decision-maker:

> **Rules decide. RAG supports. LLM explains.**

Deterministic, versioned rules produce formal assessment Findings. Dynamic questionnaires collect canonical facts, users explicitly confirm which assessment modules may run, and only authorized rules enter the canonical report. Legal Evidence then supports each Finding with traceable authority.

LLM output is not used to determine the formal legal result. If an explanatory LLM layer is added or used later, it must remain downstream of the rules and Evidence. Every Finding is a preliminary screening conclusion—not legal advice, certification, or a final compliance determination.

## How the platform works

```text
AI system and use-case facts
            ↓
Dynamic questionnaire routing
            ↓
Explicit module confirmation
            ↓
Authorized deterministic rules
            ↓
Framework-specific Findings
            ↓
Atomic legal Evidence binding
            ↓
Canonical assessment report
```

The layers have separate responsibilities:

- **Questionnaires** collect and normalize facts without deciding legal outcomes.
- **Routing** suggests implemented or unsupported pathways deterministically.
- **Users** confirm the implemented modules that may enter the assessment.
- **Rules** evaluate only authorized pathways and generate inspectable Findings.
- **Evidence services** resolve the legal references authored by each rule.
- **Reports** preserve the relationship between facts, rules, Findings, missing information, recommendations, and Evidence.

Assessment runs are immutable snapshots. Switching scenarios clears case-owned outputs, while switching language preserves the active case, report, and selected Finding.

## Current assessment coverage

The prototype currently implements four preliminary screening modules. It does not provide complete coverage of any regulation.

### EU AI Act employment high-risk screening

Rule ID: `AI_ACT_HIGH_RISK_EMPLOYMENT`

- Screens employment, worker-management, and recruitment contexts.
- Evaluates the currently implemented Article 6 and Annex III point 4(a) pathway.
- Considers covered employment functions and material influence on employment-related decisions.
- Produces an independent EU AI Act Finding and Evidence Trace.

### EU AI Act Article 6(1) product and safety-component screening

Rule ID: `AI_ACT_HIGH_RISK_PRODUCT_SAFETY`

- Evaluates whether AI is itself a product or a safety component.
- Validates a selected instrument against the structured Annex I catalogue.
- Requires separate, explicit confirmation that the selected Annex I instrument covers the product.
- Separately evaluates whether third-party conformity assessment is required.
- Authors atomic references to Article 6(1)(a), Article 6(1)(b), the selected Annex I point, and Article 3(14) when the safety-component definition is relied upon.

### GDPR Article 22 relevance screening

Rule ID: `GDPR_ARTICLE22_RELEVANCE`

- Screens personal-data processing.
- Screens automated individual decision-making.
- Considers legal or similarly significant effects through the currently implemented material-influence fact.
- Produces a preliminary Article 22 relevance Finding without deciding applicability, exceptions, safeguards, or compliance.

### EU Data Act relevance screening

Rule ID: `EU_DATA_ACT_RELEVANCE`

- Screens connected products and related services.
- Screens whether product or related-service data are generated.
- Produces an independent Data Act relevance Finding without deciding actor roles, obligations, exemptions, or compliance.

## Public demo scenarios

The Streamlit landing page provides three prepared demonstrations.

| Demo | Open with | Expected preliminary output |
|---|---|---|
| Recruitment AI Screening | **Open recruitment demo** | One `AI_ACT_HIGH_RISK_EMPLOYMENT` Finding; `potentially_applies`; seven Evidence records |
| Industrial AI Monitoring | **Open industrial demo** | One `EU_DATA_ACT_RELEVANCE` Finding; `potentially_applies`; three Evidence records |
| Industrial Robot Safety and Data Access | **Open multi-framework demo** | Two independent Findings; four AI Act Evidence records and three Data Act Evidence records; seven total |

The multi-framework demo does not merge the AI Act and Data Act analyses. Each Finding retains its own:

- factual inputs and legal predicates;
- reason codes and legal references;
- recommendations; and
- independently selectable Evidence Trace.

The repository also contains a tested custom GDPR lending acceptance path. It demonstrates dynamic routing to the implemented GDPR module while presenting the currently unsupported AI Act credit route as informational only.

## Current platform capabilities

- Typed `AssessmentFacts` with separate system, scope, organisation, supply-chain, use-context, practice, high-risk, product-regulation, data-protection, and Data Act namespaces.
- Deterministic, versioned legal-rule evaluation.
- Dynamic questionnaire definitions and rule-driven routing.
- Deterministic controlled routing suggestions and explicit module confirmation.
- English and Simplified Chinese presentation.
- Canonical normalization of controlled inputs.
- Explicit `Unknown` handling, distinct from an unanswered question.
- Dependency-aware invalidation of downstream facts and stale reports.
- Fact provenance for questionnaire and demo inputs.
- Authorized rule execution scope and canonical report scoping.
- Structured missing-information output and incomplete-assessment presentation.
- Multi-framework reports with independently selectable Findings.
- Per-Finding Evidence Trace from facts to rule, legal basis, and source Evidence.
- Framework-isolated, metadata-v2 Evidence with stable IDs.
- Clean-checkout runtime Evidence packs.
- Official EUR-Lex provenance and manifest validation.
- Scenario-state isolation and language-state preservation.

## Evidence architecture

Official EUR-Lex text is used as the authoritative legal source for the committed metadata-v2 Evidence packs. Rules author explicit legal references; the Evidence layer resolves only records that match those references.

The retrieval adapter:

- prioritizes exact atomic metadata-v2 records over broad legacy chunks;
- expands supported compound references into their atomic citations;
- preserves the original legal basis written by the rule;
- isolates Evidence by legal instrument and regulatory framework;
- fails closed against cross-framework Evidence leakage; and
- validates stable Evidence IDs, normalized excerpt hashes, manifests, and source metadata.

Runtime Evidence loading is local and deterministic. It does not require network access, official-source downloads, or candidate corpus builds.

Tracked runtime packs:

- `data/legal_evidence/eu_ai_act_product_safety_metadata_v2.json` — 23 atomic AI Act records.
- `data/legal_evidence/eu_data_act_relevance_metadata_v2.json` — three atomic Data Act records.

Downloaded official sources in `corpus_sources/` and generated candidates in `corpus_builds/` are build inputs, not deployment requirements.

## Current maturity

The present system primarily implements **Level 1: regulatory relevance and classification screening**.

It can currently help answer:

- which implemented regulatory pathway may be relevant;
- whether confirmed facts satisfy a specific implemented classification or relevance screen;
- which legal provisions support that preliminary result; and
- which information remains unresolved.

It does **not** yet fully determine:

- which economic operator holds every relevant legal role;
- the complete set of duties triggered by a classification;
- whether each substantive obligation has been performed;
- whether business documentation is sufficient compliance Evidence; or
- whether the system is legally compliant overall.

This boundary is deliberate. Classification and relevance screens are entry points for deeper legal analysis, not substitutes for it.

## Scope limitations

The platform does not currently provide:

- complete EU AI Act coverage;
- Article 6(2) assessment or complete Annex III coverage;
- Article 6(3) exception analysis;
- Article 5 prohibited-practice assessment;
- complete provider or deployer role and obligation mapping;
- a full conformity-assessment workflow;
- complete GDPR applicability or compliance analysis;
- complete Data Act role and obligation analysis;
- certification or assurance;
- final legal advice; or
- authoritative legal classification by an LLM.

Unsupported controlled routes may be disclosed in the interface without generating a formal Finding. A screened-out implemented module appears only in the routing audit and does not enter the canonical report unless explicitly confirmed and executed.

## Near-term legal deepening

The next development stage is intended to model the consequences of the existing entry-screening rules. The following areas are planned or under development; they are not current platform capabilities and may not map one-to-one to future rules.

### EU AI Act high-risk consequence chain

- Provider and deployer role identification.
- Risk-management, data-governance, technical-documentation, record-keeping, and logging duties.
- Transparency, instructions for use, and human oversight.
- Accuracy, robustness, and cybersecurity.
- Deployer monitoring and use obligations.
- Fundamental-rights impact assessment where relevant.
- Conformity-assessment responsibility boundaries.

### GDPR automated-decision consequence chain

- Controller and processor roles.
- Lawful basis and special-category data.
- Article 22 exceptions and meaningful human involvement.
- Safeguards, data-subject rights, and transparency.
- DPIA relevance, accountability, and compliance Evidence.

### Data Act access and role chain

- User, data-holder, and third-party roles.
- Identification of product and related-service data.
- Access requests, data sharing, and contractual arrangements.
- Trade-secret safeguards.
- Charging, compensation, and implementation Evidence.

## Local installation

Verified environment:

- Python 3.12
- Streamlit 1.56.0

```bash
git clone https://github.com/miles2005-web/eu-ai-act-legal-rag-prototype.git
cd eu-ai-act-legal-rag-prototype
git checkout feature/assessment-engine-v2
python -m venv venv
```

macOS or Linux:

```bash
source venv/bin/activate
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

Install the committed dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the assessment workspace:

```bash
streamlit run assessment_app.py
```

The assessment application does not require:

- an API key;
- `chromadb`;
- `corpus_sources/`;
- `corpus_builds/`; or
- runtime network access for the committed Evidence packs.

The recruitment scenario is also available as a CLI demonstration:

```bash
python scripts/run_demo_assessment.py
```

## Test status

Release-audit snapshot: **27 July 2026**.

- Fresh Python 3.12 installation from `requirements.txt`: passed.
- Full suite: 289 tests passed.
- UI suite: 92 tests passed.
- English/Simplified Chinese browser switching: passed.
- English and Chinese equal-height demo-card browser checks: passed.
- Recruitment, Industrial, multi-framework Industrial, and GDPR lending acceptance paths: passed.

Canonical validation commands:

```bash
python -m compileall -q assessment_app.py src scripts tests
python -m unittest discover -s tests -p "test_*.py" -q
python -m unittest discover -s tests/ui -p "test_*.py" -q
git diff --check
```

These figures are a dated verification snapshot, not a permanent guarantee for future commits.

## Deployment readiness

- Streamlit entry point: `assessment_app.py`
- Python version: 3.12
- Dependency source: `requirements.txt`
- Assessment-app secrets: none required
- Runtime legal assets: committed Evidence packs, manifests, Annex I catalogue, demo fixtures, and the legacy vector asset used by existing retrieval

The runtime Evidence packs must remain tracked. `corpus_sources/` and `corpus_builds/` are not deployment inputs. The current feature branch has passed a clean-clone installation, test, launch, and browser smoke audit; this does not imply that a public deployment currently exists.

## Legacy retrieval and ingestion utilities

The repository retains earlier chat and ingestion code for project history and optional development work:

- `app_chroma.py`
- `run_pipeline_chroma.py`

Those legacy executables are not required to launch `assessment_app.py`. Chroma/OpenRouter scripts are legacy or optional development utilities, and `chromadb` is intentionally excluded from the standard assessment dependency set.

The tracked `vector_store.json` is different: it remains a legacy legal-knowledge asset used by the current retrieval compatibility layer and must remain available to the assessment runtime.

Running the legacy ingestion pipeline requires separate dependencies, source data, credentials, and setup. It is outside the standard assessment installation documented above.

## Repository structure

```text
├── assessment_app.py                     # Streamlit assessment workspace
├── requirements.txt                      # Standard assessment dependencies
├── config/
│   ├── ai_act_annex_i_instruments.json   # Structured Annex I catalogue
│   ├── *_evidence_manifest.json           # Runtime-pack validation manifests
│   └── legal_sources.json                 # Legal instrument catalogue
├── data/legal_evidence/                   # Tracked metadata-v2 runtime packs
├── scripts/
│   ├── run_demo_assessment.py             # Recruitment CLI demo
│   └── build_*_candidate_corpus.py        # Official-source candidate builders
├── src/
│   ├── assessment/
│   │   ├── facts.py                       # Facts, namespaces, provenance
│   │   ├── rules/                         # Deterministic regulatory rules
│   │   ├── questionnaire/                 # Definitions, router, invalidation
│   │   ├── product_regulation/            # Annex I catalogue domain
│   │   ├── evidence/                      # Evidence models and services
│   │   ├── report/                        # Canonical report models and builder
│   │   ├── workflow/                      # Authorized assessment orchestration
│   │   └── demo/                          # Reusable dependency factory
│   ├── ui/                                # Bilingual presentation components
│   ├── eurlex_ai_act.py                   # Official AI Act source adapter
│   └── eurlex_html.py                     # Official Data Act source adapter
└── tests/
    ├── assessment/                        # Domain and workflow regressions
    ├── fixtures/                          # Public demonstration cases
    └── ui/                                # Streamlit and presentation tests
```

## Project context

This project explores how legal rules can be represented as transparent, testable computational assessments without hiding legal uncertainty. It is intended as a Law & Technology portfolio project and a prototype for real compliance-workflow design—not final product documentation.

## Disclaimer

This prototype provides preliminary screening guidance only. It does not constitute legal advice, certification, or a definitive regulatory or compliance determination. Findings should be reviewed by qualified professionals against the complete facts and current law.
