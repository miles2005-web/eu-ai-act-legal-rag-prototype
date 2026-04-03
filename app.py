from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.legal_chunks import build_structured_chunks
from src.legal_chunks import extract_query_metadata
from src.legal_chunks import is_provider_obligations_query
from src.legal_chunks import narrow_chunks_for_query
from src.legal_chunks import score_chunk


PARSED_DATA_DIR = Path("data/parsed")
LEGACY_DATA_DIR = Path("data")


def load_documents(data_dir: Path) -> list[dict]:
    """Load all text files from a directory."""
    documents = []

    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append({"source": path.name, "text": text})

    return documents


def load_available_documents() -> tuple[list[dict], str]:
    """
    Prefer parsed documents, but keep the old data/ folder as a fallback.

    This lets the app keep working even before the ingest step is used.
    """
    parsed_documents = load_documents(PARSED_DATA_DIR)
    if parsed_documents:
        return parsed_documents, "data/parsed"

    legacy_documents = load_documents(LEGACY_DATA_DIR)
    return legacy_documents, "data"


def build_index(documents: list[dict]) -> list[dict]:
    """Convert documents into searchable chunks with simple legal metadata."""
    indexed_chunks = []

    for document in documents:
        indexed_chunks.extend(
            build_structured_chunks(
                text=document["text"],
                source=document["source"],
            )
        )

    return indexed_chunks


def retrieve(query: str, indexed_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Return the best matching chunks for the user's question."""
    scored_chunks = []
    candidate_chunks = narrow_chunks_for_query(query, indexed_chunks)

    for chunk in candidate_chunks:
        score = score_chunk(query, chunk)
        if score > 0:
            scored_chunks.append({**chunk, "score": score})

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)
    return scored_chunks[:top_k]


def generate_answer_sections(question: str, matches: list[dict]) -> dict[str, str]:
    """
    Build a small structured answer from retrieved text.

    This remains rule-based and uses the top retrieved chunks only.
    """
    if not matches:
        fallback = (
            "I could not find a matching passage in the local documents. "
            "Try asking about high-risk AI systems, prohibited practices, transparency, "
            "or provider obligations."
        )
        return {
            "key_legal_point": fallback,
            "relevant_provisions": "No clear provision was retrieved.",
            "possible_obligations": "No obligations could be extracted from the current matches.",
            "information_gaps": "The local sample did not return a strong match for this question.",
        }

    top_match = matches[0]
    top_preview = _preview_text(top_match["text"], max_chars=340)
    provisions = []

    for match in matches:
        provision = _format_provision_label(match)
        if provision and provision not in provisions:
            provisions.append(provision)

    obligations_lines = []
    for match in matches[:3]:
        snippet = _extract_obligation_snippet(match["text"])
        if snippet and snippet not in obligations_lines:
            obligations_lines.append(f"- {snippet}")

    if not obligations_lines:
        obligations_lines.append("- The retrieved text may describe classification, scope, or context rather than a direct obligation.")

    caveat_parts = []
    if is_provider_obligations_query(question):
        caveat_parts.append(
            "This looks like a broad obligations query, so the ranking now prefers Articles 9, 10, 11, 13, and 14 when they are relevant."
        )
    if len(matches) > 1:
        caveat_parts.append("The answer is based on the top retrieved excerpts, not the full Regulation.")
    caveat_parts.append("Some OCR or parsing noise may still remain in the source text.")

    return {
        "key_legal_point": top_preview,
        "relevant_provisions": ", ".join(provisions) if provisions else "No specific provision label was detected.",
        "possible_obligations": "\n".join(obligations_lines),
        "information_gaps": " ".join(caveat_parts),
    }


def _preview_text(text: str, max_chars: int = 300) -> str:
    compact = text.replace("\n", " ").strip()
    preview = compact[:max_chars].rstrip()
    if len(compact) > max_chars:
        preview += "..."
    return preview


def _format_provision_label(match: dict) -> str | None:
    if match.get("canonical_citation"):
        return match["canonical_citation"]
    if match.get("article_number"):
        return f"Article {match['article_number']}"
    if match.get("annex_ref"):
        return f"Annex {match['annex_ref']}"
    if match.get("recital_ref"):
        return f"Recital {match['recital_ref']}"
    return None


def _extract_obligation_snippet(text: str) -> str | None:
    compact = text.replace("\n", " ").strip()
    sentences = [part.strip() for part in compact.split(".") if part.strip()]

    obligation_markers = (
        "shall",
        "must",
        "required to",
        "ensure",
        "designed",
        "developed",
        "draw up",
        "maintain",
        "provide",
    )
    for sentence in sentences:
        lowered = sentence.lower()
        if any(marker in lowered for marker in obligation_markers):
            return sentence[:220].rstrip() + ("..." if len(sentence) > 220 else "")

    return sentences[0][:220].rstrip() + ("..." if sentences and len(sentences[0]) > 220 else "") if sentences else None


st.set_page_config(page_title="EU AI Act Legal RAG Demo", page_icon="⚖️", layout="wide")

st.title("EU AI Act Legal RAG Demo")
st.write(
    "Ask a question about the local EU AI Act materials. "
    "This beginner-friendly demo uses local text retrieval with legal-structure-aware chunks."
)

documents, active_data_source = load_available_documents()
indexed_chunks = build_index(documents)

if not documents:
    st.error("No text files were found in `data/parsed` or the legacy `data/` folder.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("Loaded from:", active_data_source)
    st.write("Documents loaded:", len(documents))
    st.write("Searchable chunks:", len(indexed_chunks))
    st.write("Chunking: chapter / section / article / annex / recital aware")
    st.write("Retrieval method: keyword overlap plus simple metadata boosts")
    st.write("Metadata: citation, chapter/section numbers, article title, parent context")

question = st.text_input(
    "Your question",
    placeholder="Example: What does the EU AI Act say about high-risk AI systems?",
)

submitted = st.button("Ask")

if submitted and question.strip():
    results = retrieve(question, indexed_chunks)
    answer_sections = generate_answer_sections(question, results)
    query_metadata = extract_query_metadata(question)

    st.subheader("Answer")
    st.markdown(f"**Key legal point**  \n{answer_sections['key_legal_point']}")
    st.markdown(f"**Relevant provisions**  \n{answer_sections['relevant_provisions']}")
    st.markdown(f"**Possible obligations**  \n{answer_sections['possible_obligations']}")
    st.markdown(f"**Information gaps / caveats**  \n{answer_sections['information_gaps']}")

    if any(query_metadata.values()):
        active_filters = []
        if query_metadata["article_number"]:
            article_ref = f"Article {query_metadata['article_number']}"
            if query_metadata.get("paragraph_number"):
                article_ref += f"({query_metadata['paragraph_number']})"
            if query_metadata.get("point_label"):
                article_ref += f"({query_metadata['point_label'].lower()})"
            active_filters.append(article_ref)
        if query_metadata["annex_ref"]:
            active_filters.append(f"Annex {query_metadata['annex_ref']}")
        if query_metadata["chapter_number"]:
            active_filters.append(f"Chapter {query_metadata['chapter_number']}")
        if query_metadata["section_number"]:
            active_filters.append(f"Section {query_metadata['section_number']}")
        if query_metadata["recital_ref"]:
            active_filters.append(f"Recital {query_metadata['recital_ref']}")
        st.caption("Detected legal reference in query: " + ", ".join(active_filters))
    elif is_provider_obligations_query(question):
        st.caption(
            "Detected broad obligations query: ranking gives extra weight to Articles 9, 10, 11, 13, and 14."
        )

    st.subheader("Retrieved Source Excerpts")
    if results:
        for result in results:
            citation = f"{result['source']} (chunk {result['chunk_id']})"
            with st.expander(citation, expanded=True):
                if result.get("canonical_citation"):
                    st.markdown(f"**Citation**  \n{result['canonical_citation']}")

                if result.get("parent_citation"):
                    st.caption("Parent context: " + result["parent_citation"])

                metadata_bits = []
                if result.get("chapter_heading"):
                    chapter_label = result["chapter_heading"]
                    if result.get("chapter_number"):
                        chapter_label += f" [number: {result['chapter_number']}]"
                    metadata_bits.append(chapter_label)
                if result.get("section_heading"):
                    section_label = result["section_heading"]
                    if result.get("section_number"):
                        section_label += f" [number: {result['section_number']}]"
                    metadata_bits.append(section_label)
                if result.get("article_title"):
                    metadata_bits.append(f"Article title: {result['article_title']}")
                if result.get("paragraph_number"):
                    metadata_bits.append(f"Paragraph: {result['paragraph_number']}")
                if result.get("point_label"):
                    metadata_bits.append(f"Point: ({result['point_label']})")
                if result.get("annex_ref"):
                    metadata_bits.append(f"Annex {result['annex_ref']}")
                if result.get("recital_ref"):
                    metadata_bits.append(f"Recital {result['recital_ref']}")

                if metadata_bits:
                    st.caption("Chunk metadata: " + " | ".join(metadata_bits))
                st.write(result["text"])
                st.caption(f"Source: {citation} | Score: {result['score']:.2f}")
    else:
        st.write("No relevant source excerpts were found.")
elif submitted:
    st.warning("Please enter a question first.")
