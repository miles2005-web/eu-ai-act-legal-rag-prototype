import os, time, streamlit as st
from openai import OpenAI
import chromadb
import re

st.set_page_config(page_title="EU AI Act Compliance Navigator", page_icon="⚖️", layout="wide")
st.title("⚖️ EU AI Act Compliance Navigator")
st.write("Describe your AI system or ask about the EU AI Act. The system retrieves relevant provisions and generates a structured compliance assessment.")

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    st.error("Set OPENROUTER_API_KEY before running.")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
db = chromadb.PersistentClient(path="./chroma_db")
col = db.get_collection("eu_ai_act")

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of provisions to retrieve", min_value=3, max_value=15, value=5)
    show_distances = st.checkbox("Show similarity distances", value=True)

    st.markdown("---")
    st.header("System Info")
    st.write(f"Records in ChromaDB: {col.count()}")
    st.write("Embedding: text-embedding-3-small")
    st.write("LLM: GPT-4o-mini via OpenRouter")

    st.markdown("---")
    st.header("Example Questions")
    examples = {
        "High-risk classification": "What does Article 6 say about high-risk AI classification?",
        "HR screening tool": "An AI system that screens job applicants' CVs and ranks candidates for recruitment.",
        "Provider obligations": "What obligations do providers of high-risk AI systems have?",
        "Annex III": "What AI systems are listed in Annex III?",
        "Transparency": "What transparency requirements apply to AI systems?",
        "Prohibited practices": "What AI practices are prohibited under the EU AI Act?",
        "Medical AI device": "A machine learning model that analyzes chest X-rays to detect pneumonia in a hospital.",
    }
    for label, q in examples.items():
        if st.button(label):
            st.session_state["question"] = q

    # Query history
    if st.session_state.get("history"):
        st.markdown("---")
        st.header("Query History")
        for i, h in enumerate(reversed(st.session_state["history"][-10:])):
            st.caption(f"{h['time']} — {h['query'][:50]}...")

# ---- Highlight helper ----
def highlight_terms(text, query):
    keywords = set()
    for word in query.lower().split():
        if len(word) > 3 and word not in {"what", "does", "that", "this", "with", "from", "about", "have", "their", "under", "which"}:
            keywords.add(word)
    # Also highlight Article/Annex/Recital references
    for match in re.findall(r'(?:Article|Annex|Recital)\s+\w+(?:\(\w+\))*', text, re.IGNORECASE):
        keywords.add(match.lower())
    result = text
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(lambda m: f"**{m.group()}**", result)
    return result

# ---- Main ----
if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("Your question", value=st.session_state.get("question", ""), placeholder="e.g. What are the obligations for providers of high-risk AI systems?")

if st.button("Analyze", type="primary") and question.strip():
    t0 = time.time()

    with st.spinner("Retrieving relevant provisions..."):
        qr = client.embeddings.create(model="openai/text-embedding-3-small", input=question)
        results = col.query(query_embeddings=[qr.data[0].embedding], n_results=top_k)
    t_retrieve = time.time() - t0

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0] if "distances" in results else [None]*len(docs)

    context = "\n\n---\n\n".join([
        f"[{metas[i].get('canonical_citation','N/A')}]\n{doc}"
        for i, doc in enumerate(docs)
    ])

    with st.spinner("Generating compliance assessment..."):
        llm_response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": (
                    "You are an EU AI Act compliance analyst. Based ONLY on the retrieved provisions below, "
                    "provide a structured assessment. Use this format:\n\n"
                    "## Risk Classification\n[classification + reasoning]\n\n"
                    "## Applicable Legal Basis\n[articles with brief explanations]\n\n"
                    "## Key Compliance Obligations\n[numbered list with article references]\n\n"
                    "## Cross-Regulatory Considerations\n[GDPR or other overlaps if found]\n\n"
                    "## Information Gaps\n[what the retrieved context did not cover]\n\n"
                    "Cite specific article numbers for every claim. "
                    "If the retrieved context does not contain enough information, say so explicitly. "
                    "Do NOT use knowledge beyond the retrieved text."
                )},
                {"role": "user", "content": f"RETRIEVED PROVISIONS:\n{context}\n\nQUESTION:\n{question}"}
            ],
        )
        answer = llm_response.choices[0].message.content
    t_total = time.time() - t0

    # Save to history
    st.session_state["history"].append({
        "time": time.strftime("%H:%M"),
        "query": question,
    })

    # Display assessment
    st.subheader("Compliance Assessment")
    st.markdown(answer)

    # Performance
    st.caption(f"Retrieval: {t_retrieve:.1f}s | Total: {t_total:.1f}s | Provisions retrieved: {len(docs)}")

    # Retrieved provisions
    st.markdown("---")
    st.subheader("Retrieved Source Provisions")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
        dist_label = f" | distance: {dist:.3f}" if (dist is not None and show_distances) else ""
        with st.expander(f"[{i+1}] {citation}{dist_label}", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                if meta.get("article_number"):
                    st.caption(f"Article: {meta['article_number']}")
            with col2:
                if meta.get("annex_ref"):
                    st.caption(f"Annex: {meta['annex_ref']}")
            with col3:
                if meta.get("recital_ref"):
                    st.caption(f"Recital: {meta['recital_ref']}")
            highlighted = highlight_terms(doc, question)
            st.markdown(highlighted)

    st.markdown("---")
    st.caption("⚠️ This tool provides preliminary guidance only and does not constitute legal advice. Always consult qualified legal counsel for compliance decisions.")
