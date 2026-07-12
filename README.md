# EU AI Act Compliance Assessment Platform

A legal engineering prototype for structured, evidence-grounded assessment under the EU Artificial Intelligence Act (Regulation (EU) 2024/1689).

The project began as a legal retrieval and chat prototype. It is evolving into a rule-driven AI governance assessment framework in which retrieval supports legal findings rather than acting as the primary decision mechanism.

## Overview

EU AI Act compliance requires more than locating relevant provisions. An organisation must establish facts about an AI system, determine its regulatory role and risk category, identify potentially applicable obligations, and preserve a defensible link between each conclusion and its legal authority.

Document retrieval alone cannot reliably perform that analysis: similar provisions can have different legal effects, classification depends on structured factual conditions, and missing information must remain visible rather than being filled by inference.

This platform therefore combines:

- structured facts collected through a questionnaire;
- deterministic, versioned legal rules;
- legal corpus retrieval for supporting authority;
- traceable findings and missing-information records; and
- standardized compliance reports suitable for review.

The output is a preliminary compliance assessment, not legal advice or a definitive legal classification.

## Architecture

```text
AI System Facts
        ↓
Questionnaire Engine
        ↓
Assessment Workflow
        ↓
Rule Engine
        ↓
Legal Findings
        ↓
Evidence Retrieval
        ↓
Compliance Report
```

The layers have deliberately separate responsibilities. Questionnaires collect facts, rules evaluate legal conditions, evidence services bind findings to authority, and the report layer produces a deterministic and reviewable output. Assessment runs remain immutable snapshots of the facts and rule versions used.

## Core Capabilities

- **EU AI Act legal corpus retrieval** — Legal-structure-aware chunking and vector retrieval across articles, recitals, and annexes.
- **Structured assessment workflow** — Case facts, missing-fact validation, assessment execution, evidence resolution, and report generation.
- **Rule-based risk classification** — Typed, reusable, versioned rules with explicit required facts and legal-basis metadata.
- **Evidence-grounded findings** — Legal findings can be connected to citations and excerpts without placing retrieval inside the rule logic.
- **Traceable compliance reports** — Deterministic reports preserve the path from facts to findings, rule versions, and supporting evidence.

## Demo

The Streamlit prototype demonstrates the structured assessment workflow from case creation to an evidence-grounded report. A user can:

1. create an assessment case or load the prepared recruitment AI scenario;
2. review and provide the facts required by the registered rule;
3. run the deterministic assessment workflow;
4. inspect the preliminary classification, legal basis, and reasoning trace; and
5. review the supporting legal evidence and report recommendations.

The interface keeps presentation separate from legal reasoning: Streamlit collects and displays information, while the assessment engine, evidence service, and report builder produce the result.

### Recruitment AI Screening Scenario

The included demonstration models an AI system used by a company to screen and rank job candidates. Its output materially influences access to employment opportunities.

The assessment workflow collects these facts and applies a preliminary employment high-risk rule based on Article 6 and Annex III point 4(a) of the EU AI Act. Where the required conditions are satisfied, the result is expressed cautiously as **potentially applies**, preserving the need for further legal and factual review.

The reusable scenario data is stored in `tests/fixtures/recruitment_ai_case.json` and can be loaded directly from the Streamlit landing page.

### Screenshots

> **Assessment facts input** — Screenshot placeholder for the populated structured-facts form.

> **Assessment report** — Screenshot placeholder for the preliminary classification and report summary.

> **Evidence trace** — Screenshot placeholder for the reasoning trace and supporting legal evidence.

## Project Status

- ✅ Structured assessment workflow implemented
- ✅ Evidence-grounded report generation implemented
- ✅ Streamlit assessment prototype available
- 🚧 Additional EU AI Act rules and regulatory domains in development

## How to Run the Demo

After installing the project dependencies, launch the assessment interface from the repository root:

```bash
streamlit run assessment_app.py
```

Then open `http://localhost:8501` and select **Load Recruitment AI Screening Demo**.

The same scenario can also be executed without a UI:

```bash
python scripts/run_demo_assessment.py
```

## Technical Stack

- Python and typed domain models
- Streamlit for the assessment workflow and existing legal retrieval prototype
- ChromaDB and exported vector data for legal corpus retrieval
- Versioned rule engine architecture
- Legal-structure-aware PDF processing and chunking
- Deterministic assessment workflow and report generation

## Repository Structure

```text
├── assessment_app.py               # Streamlit assessment workflow prototype
├── app_chroma.py                    # Existing Streamlit legal retrieval prototype
├── run_pipeline_chroma.py           # Chroma ingestion pipeline
├── vector_store.json                # Exported legal corpus vectors
├── scripts/
│   ├── evaluate_retrieval.py        # Retrieval evaluation
│   └── run_demo_assessment.py       # End-to-end assessment demonstration
├── src/
│   ├── ingest.py                    # Legal document parsing
│   ├── legal_chunks.py              # Legal-structure-aware chunking
│   └── assessment/
│       ├── case/                    # Assessment case lifecycle
│       ├── evidence/                # Evidence models, service, and retrieval adapter
│       ├── questionnaire/           # Question definitions and missing-fact routing
│       ├── report/                  # Deterministic compliance reports
│       ├── rules/                   # Rule contracts, registry, and legal rules
│       ├── workflow/                # Assessment orchestration
│       ├── engine.py                # Rule execution runtime
│       ├── facts.py                 # Structured assessment facts
│       └── findings.py              # Legal finding model
└── tests/
    ├── assessment/                  # Domain and workflow tests
    └── fixtures/                    # Reusable assessment scenarios
```

## Legal Retrieval Foundation

The corpus pipeline preserves Chapter, Section, Article, Recital, and Annex boundaries rather than splitting legislation only by token length. This matters because EU AI Act classification provisions and their consequences are distributed across linked legal structures—for example, Article 6 and Annex III.

The current retrieval corpus includes the EU AI Act and supporting EU digital-regulation materials. Retrieval is used to supply legal evidence; it does not replace structured fact collection or rule evaluation.

## Running Locally

Create an environment and install the project dependencies:

```bash
git clone https://github.com/miles2005-web/eu-ai-act-legal-rag-prototype.git
cd eu-ai-act-legal-rag-prototype
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the assessment interface:

```bash
streamlit run assessment_app.py
```

Run the existing Streamlit retrieval interface:

```bash
export OPENROUTER_API_KEY="your-key-here"
streamlit run app_chroma.py
```

## Roadmap

- Dynamic questionnaire expansion for additional rule requirements
- Additional EU AI Act classification, role, prohibition, and obligation rules
- GDPR and Data Act cross-regulatory assessment
- Broader support for EU digital regulation
- Persistent cases, review workflows, exports, and enterprise compliance processes

## Project Context

This project explores how legal rules can be represented as transparent computational assessments without obscuring legal uncertainty. It is designed as a demonstrable Law & Technology portfolio project and as a prototype for real compliance workflow design.

Built by a law student at Jilin University (Changchun, China).

## Disclaimer

This prototype provides preliminary compliance guidance only. It does not constitute legal advice, and its findings should be reviewed by qualified legal professionals against the complete facts and current law.
