# ⚖️ EU AI Act Compliance Navigator

A RAG-based compliance assessment tool for the EU Artificial Intelligence Act (Regulation 2024/1689). Users describe their AI system in natural language, and the tool retrieves relevant legal provisions and generates a structured compliance report.

**Live Demo:** [eu-ai-act-legal-rag-prototype.streamlit.app](https://eu-ai-act-legal-rag-prototype-hgryem2gsyrmyz7m6tda6c.streamlit.app)

## What It Does

1. **Risk Classification** — Input an AI system description → receive risk tier (Unacceptable / High-Risk / Limited / Minimal) with article citations
2. **Obligation Mapping** — Outputs specific compliance requirements with article references
3. **Cross-Regulatory Analysis** — Identifies overlapping obligations from GDPR, product safety, and cybersecurity frameworks
4. **Source Transparency** — Every answer cites specific articles, recitals, and annexes
5. **Smart Routing** — Automatically detects legal references (Article, Annex, Recital) in queries for precision metadata-filtered retrieval

## Technical Architecture


