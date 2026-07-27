# Project documentation

This directory separates current maintainable documentation from historical
project records.

## Current documents

| Document | Status | Applicable release |
|---|---|---|
| [Project Maintenance Manual](PROJECT_MAINTENANCE_MANUAL.md) | `2.0` — current canonical operational and architecture manual | `v0.5.0-prototype` |
| [Project Development Story](PROJECT_DEVELOPMENT_STORY.md) | `2.0` — current canonical Markdown development history; no v2 PDF has been generated | `v0.5.0-prototype` |
| [Repository README](../README.md) | Published project overview and launch instructions | `main` |

Production application:

https://eu-regulatory-assessment.streamlit.app/

Published release:

https://github.com/miles2005-web/eu-ai-act-legal-rag-prototype/releases/tag/v0.5.0-prototype

## Archived documents

| Document | Status | Historical scope |
|---|---|---|
| [Project Maintenance Manual v1](archive/PROJECT_MAINTENANCE_MANUAL_v1.md) | Archived, unchanged | Original legal RAG chatbot maintenance instructions |
| [Project Development Story v1](archive/PROJECT_DEVELOPMENT_STORY_v1.pdf) | Archived, unchanged | Original project history from the minimum retrieval loop through the first chatbot deployment |

Archived documents are retained as historical evidence. They do not describe
the current production architecture, dependency model, entry point, legal
coverage, or deployment requirements.

## Document status

The v2 Markdown documents are the current canonical maintenance manual and
development history. `PROJECT_DEVELOPMENT_STORY.md` remains the maintainable
source for the development story; no v2 PDF has yet been generated.

## Implementation source of truth

When documentation conflicts with the implemented software, use this order:

1. application code and versioned rule configuration;
2. automated tests and scenario fixtures;
3. Evidence manifests and runtime Evidence packs;
4. `PROJECT_MAINTENANCE_MANUAL.md`;
5. root `README.md`; and
6. historical archived documents.

Code and tests determine what the software currently does. They do not
override, amend, or conclusively establish what the law requires.

## Legal-authority hierarchy

For legal propositions and source maintenance, use this order:

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

Documentation updates must verify implementation statements against the
implementation hierarchy and legal propositions against the legal-authority
hierarchy. Legal-source drift can make a technically passing release legally
outdated. Historical claims must remain labelled as historical and must not be
promoted into current runtime or legal claims without current evidence.
