import os, streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="EU AI Act Legal RAG", page_icon="⚖️", layout="wide")
st.title("⚖️ EU AI Act Compliance Navigator")
st.write("Describe your AI system or ask about the EU AI Act. The system retrieves relevant provisions and generates a structured compliance assessment.")

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    st.error("Set OPENROUTER_API_KEY before running.")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
db = chromadb.PersistentClient(path="./chroma_db")
col = db.get_collection("eu_ai_act")

with st.sidebar:
    st.header("About")
    st.write(f"Records in ChromaDB: {col.count()}")
    st.write("Embedding: text-embedding-3-small")
    st.write("LLM: GPT-4o-mini via OpenRouter")
    st.write("Retrieval: semantic vector search")
    st.markdown("---")
    st.header("Example Questions")
    examples = {
        "High-risk classification": "What does Article 6 say about high-risk AI classification?",
        "HR screening tool": "An AI system that screens job applicants' CVs and ranks candidates for recruitment.",
        "Provider obligations": "What obligations do providers of high-risk AI systems have?",
        "Annex III": "What AI systems are listed in Annex III?",
        "Transparency": "What transparency requirements apply to AI systems?",
    }
    for label, q in examples.items():
        if st.button(label):
            st.session_state["question"] = q

question = st.text_input("Your question", value=st.session_state.get("question", ""), placeholder="e.g. What are the obligations for providers of high-risk AI systems?")

if st.button("Analyze", type="primary") and question.strip():
    with st.spinner("Retrieving relevant provisions..."):
        qr = client.embeddings.create(model="openai/text-embedding-3-small", input=question)
        results = col.query(query_embeddings=[qr.data[0].embedding], n_results=5)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0] if "distances" in results else [None]*len(docs)

    context = "\n\n---\n\n".join([
        f"[{metas[i].get('canonical_citation','N/A')}]\n{doc}"
        for i, doc in enumerate(docs)
    ])

    with st.spinner("Generating assessment..."):
        llm_response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": (
                    "You are an EU AI Act compliance analyst. Based ONLY on the retrieved provisions below, "
                    "provide a structured assessment. Cite specific article numbers for every claim. "
                    "If the retrieved context does not contain enough information, say so explicitly. "
                    "Do NOT use knowledge beyond the retrieved text."
                )},
                {"role": "user", "content": f"RETRIEVED PROVISIONS:\n{context}\n\nQUESTION:\n{question}"}
            ],
        )
        answer = llm_response.choices[0].message.content

    st.subheader("Compliance Assessment")
    st.markdown(answer)

    st.markdown("---")
    st.subheader("Retrieved Source Provisions")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
        score_info = f" | distance: {dist:.3f}" if dist is not None else ""
        with st.expander(f"[{i+1}] {citation}{score_info}", expanded=False):
            if meta.get("article_number"):
                st.caption(f"Article: {meta['article_number']}")
            if meta.get("annex_ref"):
                st.caption(f"Annex: {meta['annex_ref']}")
            if meta.get("recital_ref"):
                st.caption(f"Recital: {meta['recital_ref']}")
            st.write(doc)

    st.markdown("---")
    st.caption("⚠️ This tool provides preliminary guidance only and does not constitute legal advice.")
