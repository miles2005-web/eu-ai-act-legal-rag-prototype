# EU AI Act Legal RAG Prototype

## Overview

The EU AI Act is a long and structurally dense regulation. For a reader who wants to locate a specific article, annex, recital, or compliance-related obligation, manual navigation is slow and often inefficient. This project explores a narrower and more explainable alternative: a lightweight local legal retrieval prototype that helps a user find relevant passages in the Regulation through a simple question-and-answer interface.

The prototype is built around the EU AI Act as a practical test case because the text combines both precise reference-style queries, such as `Article 6`, `Annex III`, or `Recital 12`, and broader compliance-style queries, such as provider obligations for high-risk AI systems. Rather than presenting itself as a production legal assistant, the project is intentionally scoped as a local, inspectable, and constrained retrieval system for a complex regulatory document.

Its value lies in demonstrating three things clearly:

- legal-structure-aware text processing,
- metadata-informed retrieval for legal references,
- careful scope control in a domain where overclaiming would be misleading.

## Project Purpose

The purpose of this project is to make a complex legal text easier to explore through a simple and transparent interface. Instead of requiring a user to manually scan a long regulation, the app:

- retrieves excerpts from the local source text,
- surfaces provisions that are likely to be relevant to the query,
- shows basic legal metadata such as article, annex, recital, chapter, and section context,
- presents a short structured response grounded in the retrieved material.

The project is meant to support orientation, preliminary reading, and fast navigation within the Regulation. It is not intended to replace professional legal analysis.

## Why the EU AI Act

The EU AI Act is a strong test case for a legal retrieval prototype for three reasons:

- it is lengthy and structurally complex, with chapters, sections, articles, annexes, and recitals,
- it contains both precise legal-reference queries and broader compliance-oriented questions,
- it is a timely and policy-relevant regulation at the intersection of law, governance, and artificial intelligence.

That combination makes it especially suitable for demonstrating both the usefulness and the limits of lightweight legal retrieval.

## Current Features

The current prototype includes:

- a local Streamlit interface for question-answering over the EU AI Act text,
- ingestion of `.txt` and basic `.pdf` files from `data/raw/`,
- plain-text export into `data/parsed/`,
- conservative OCR-noise cleanup for repeated page artifacts, broken legal headings, and a limited set of obvious parsing errors,
- legal-structure-aware chunking based on chapter, section, article, annex, and recital boundaries where available,
- lightweight per-chunk metadata including:
  - `article_number`
  - `annex_ref`
  - `recital_ref`
  - `chapter_heading`
  - `section_heading`
- retrieval based on keyword overlap plus rule-based legal-reference boosts,
- additional lightweight ranking preference for broad obligations-style questions concerning providers of high-risk AI systems, with emphasis on Articles 9, 10, 11, 13, and 14,
- structured answer display with:
  - Key legal point
  - Relevant provisions
  - Possible obligations
  - Information gaps / caveats
- retrieved source excerpts with simple citations, chunk metadata, and scores.

## Technical Approach

The system uses a deliberately simple local pipeline.

1. Source documents are placed in `data/raw/`.
2. The ingestion script parses supported files and writes cleaned plain text into `data/parsed/`.
3. The app loads the parsed text files.
4. The text is split into chunks using legal structure where possible, rather than only fixed-size windows.
5. Each chunk receives lightweight metadata for article, annex, recital, chapter, and section context.
6. Queries are matched to chunks using keyword overlap.
7. The scorer adds rule-based boosts when the query contains clear legal references such as `Article 6`, `Annex III`, or `Recital 12`.
8. For broad obligations-style questions, the scorer adds a small heuristic preference for the core high-risk provider-obligation articles.
9. The app renders the top matches together with a short structured summary.

This design is intentionally conservative. It does not aim to maximise architectural sophistication. Instead, it prioritises inspectability, clear design choices, and a workflow that can be explained without overstating what the system does.

## Why the Project Uses a Lightweight Local Design

The project intentionally does not use:

- vector databases,
- embeddings,
- LLM APIs,
- external retrieval services,
- complex orchestration frameworks.

This is a design choice, not merely an omission. At this stage, the goal is to show how far a small and understandable legal retrieval system can go when it is shaped around the structure of the legal text itself. A local design also makes the retrieval logic easier to inspect, easier to explain, and easier to present honestly.

## Limitations

This prototype has important limitations and should be presented honestly.

- It is a retrieval prototype, not a legal reasoning engine.
- The answer section is generated from retrieved text with simple rules, not deep semantic synthesis.
- The ranking logic is heuristic and may miss relevant provisions for broad or ambiguous questions.
- OCR and PDF parsing noise has been reduced, but not eliminated.
- The system works best on the EU AI Act structure it was tuned around; it has not been generalized or systematically evaluated across many legal corpora.
- It does not verify legal interpretation, reconcile conflicts between provisions, or provide legal advice.
- It should be used as a navigation and reading aid, not as an authoritative legal conclusion.

## Repository Structure

```text
legal-rag/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── ingest.py
│   └── legal_chunks.py
└── data/
    ├── raw/
    └── parsed/
```

Key files:

- `app.py`: Streamlit interface, retrieval flow, and structured answer rendering.
- `src/ingest.py`: document ingestion and plain-text export to `data/parsed/`.
- `src/legal_chunks.py`: text cleanup, legal chunking, metadata extraction, and scoring logic.
- `data/raw/`: input documents.
- `data/parsed/`: parsed text used by the app.

## How to Run Locally

This project is intended to run locally with Python 3.12.

### 1. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

If `python3.12` is not available on your machine, install it first.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add source documents

Place `.txt` or `.pdf` files in `data/raw/`.

### 4. Run ingestion

```bash
python src/ingest.py
```

This creates or updates cleaned plain-text files in `data/parsed/`.

### 5. Start the app

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## Example Questions

- `What does Article 6 say about high-risk AI classification?`
- `What is listed in Annex III?`
- `What does Recital 12 discuss?`
- `What obligations do providers of high-risk AI systems have?`
- `What transparency duties apply under the EU AI Act?`

## Presentation Positioning

For an admissions, portfolio, or project-discussion setting, this project is best described as:

- a focused legal AI prototype,
- a local retrieval system for a complex regulatory document,
- an example of legal-structure-aware text processing,
- an example of rule-based legal text engineering under explicit scope constraints.

Its purpose is not to automate legal judgment. Its purpose is to reduce the time required to locate relevant provisions, surface likely starting points for further analysis, and demonstrate how legal text can be turned into a more navigable technical object without claiming more certainty than the system can justify.

It should not be described as:

- a full legal assistant,
- a production RAG platform,
- a system capable of reliable legal advice,
- or a substitute for professional legal interpretation.

## Disclaimer

This project is a legal retrieval and navigation prototype for educational and demonstration purposes only. It does not provide legal advice, does not verify legal conclusions, and should not be relied upon as an authoritative compliance assessment. Any real-world legal use would require qualified professional review.
