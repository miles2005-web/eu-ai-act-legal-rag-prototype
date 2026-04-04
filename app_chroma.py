import os, time, re, streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="EU AI Act Compliance Navigator", page_icon="⚖️", layout="wide")
st.title("⚖️ EU AI Act Compliance Navigator")
st.write("Describe your AI system or ask about the EU AI Act. Mention specific articles or annexes for precision retrieval.")

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    st.error("Set OPENROUTER_API_KEY before running.")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
db = chromadb.PersistentClient(path="./chroma_db")
col = db.get_collection("eu_ai_act")

def extract_legal_references(query):
    refs = {"has_references": False}
    art = re.search(r'Article\s+(\d+)', query, re.IGNORECASE)
    if art:
        refs["article"] = f"Article {art.group(1)}"
        refs["has_references"] = True
    anx = re.search(r'Annex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII)', query, re.IGNORECASE)
    if anx:
        refs["annex"] = f"Annex {anx.group(1)}"
        refs["has_references"] = True
    rec = re.search(r'Recital\s+(\d+)', query, re.IGNORECASE)
    if rec:
        refs["recital"] = f"Recital {rec.group(1)}"
        refs["has_references"] = True
    return refs

def estimate_tokens(text):
    return len(text) // 4

def apply_token_budget(docs, metas, dists, budget):
    sel_d, sel_m, sel_dist = [], [], []
    total = 0
    for doc, meta, dist in zip(docs, metas, dists):
        t = estimate_tokens(doc)
        if total + t > budget and sel_d:
            break
        sel_d.append(doc); sel_m.append(meta); sel_dist.append(dist)
        total += t
    return sel_d, sel_m, sel_dist, total

def highlight_terms(text, query):
    keywords = set()
    for word in query.lower().split():
        if len(word) > 3 and word not in {"what","does","that","this","with","from","about","have","their","under","which","they","been","also","shall","into"}:
            keywords.add(word)
    for match in re.findall(r'(?:Article|Annex|Recital)\s+\w+(?:\(\w+\))*', text, re.IGNORECASE):
        keywords.add(match.lower())
    result = text
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        result = pattern.sub(lambda m: f"**{m.group()}**", result)
    return result

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Initial retrieval count", 3, 20, 10)
    token_budget = st.slider("Context token budget", 2000, 12000, 6000, step=1000)
    similarity_cutoff = st.slider("Similarity cutoff (distance)", 0.5, 2.0, 1.2, step=0.1)
    show_routing = st.checkbox("Show Self-Query routing info", value=True)
    st.markdown("---")
    st.header("System Info")
    st.write(f"Records: {col.count()}")
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
        "Biometrics + Annex III": "Does my biometric identification system fall under Annex III?",
        "Article 9 risk mgmt": "What does Article 9 require for risk management?",
    }
    for label, q in examples.items():
        if st.button(label):
            st.session_state["question"] = q
    if st.session_state.get("history"):
        st.markdown("---")
        st.header("Query History")
        for h in reversed(st.session_state.get("history", [])[-10:]):
            st.caption(f"{h['time']} — {h['query'][:50]}...")

if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("Your question", value=st.session_state.get("question", ""),
    placeholder="e.g. What are the obligations for providers of high-risk AI systems?")

if st.button("Analyze", type="primary") and question.strip():
    t0 = time.time()
    refs = extract_legal_references(question)
    where_filter = None
    route_msg = "🔍 No specific legal reference detected → full vector search"
    if refs["has_references"]:
        conditions = []
        if refs.get("article"):
            conditions.append({"article_number": {"$eq": refs["article"]}})
        if refs.get("annex"):
            conditions.append({"annex_ref": {"$eq": refs["annex"]}})
        if refs.get("recital"):
            conditions.append({"recital_ref": {"$eq": refs["recital"]}})
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$or": conditions}
        detected = ", ".join(v for v in [refs.get("article"), refs.get("annex"), refs.get("recital")] if v)
        route_msg = f"🎯 Detected: {detected} → metadata-filtered search"
    if show_routing:
        st.info(route_msg)
    with st.spinner("Retrieving provisions..."):
        qr = client.embeddings.create(model="openai/text-embedding-3-small", input=question)
        query_args = {"query_embeddings": [qr.data[0].embedding], "n_results": top_k}
        if where_filter:
            query_args["where"] = where_filter
        try:
            results = col.query(**query_args)
        except Exception:
            results = col.query(query_embeddings=[qr.data[0].embedding], n_results=top_k)
            if show_routing:
                st.warning("Metadata filter failed, fell back to vector search.")
    t_retrieve = time.time() - t0
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0] if "distances" in results else [0.0]*len(docs)
    filtered = [(d, m, dist) for d, m, dist in zip(docs, metas, dists) if dist <= similarity_cutoff]
    if not filtered:
        filtered = [(docs[0], metas[0], dists[0])] if docs else []
    docs_f = [x[0] for x in filtered]
    metas_f = [x[1] for x in filtered]
    dists_f = [x[2] for x in filtered]
    docs_b, metas_b, dists_b, used_tokens = apply_token_budget(docs_f, metas_f, dists_f, token_budget)
    context = "\n\n---\n\n".join([
        f"[{metas_b[i].get('canonical_citation','N/A')}]\n{doc}"
        for i, doc in enumerate(docs_b)
    ])
    with st.spinner("Generating compliance assessment..."):
        llm_response = client.chat.completions.create(
            model="openai/gpt-4o-mini", temperature=0.1,
            messages=[
                {"role": "system", "content": (
                    "You are an EU AI Act compliance analyst. Based ONLY on the retrieved provisions below, "
                    "provide a structured assessment using this format:\n\n"
                    "## Risk Classification\n[classification + reasoning with article citations]\n\n"
                    "## Applicable Legal Basis\n[articles with brief explanations]\n\n"
                    "## Key Compliance Obligations\n[numbered list with article references]\n\n"
                    "## Cross-Regulatory Considerations\n[GDPR or other overlaps if found]\n\n"
                    "## Information Gaps\n[what the retrieved context did not cover]\n\n"
                    "Cite specific article/recital numbers for every claim. "
                    "If the context is insufficient, say so. Do NOT use external knowledge."
                )},
                {"role": "user", "content": f"RETRIEVED PROVISIONS:\n{context}\n\nQUESTION:\n{question}"}
            ],
        )
        answer = llm_response.choices[0].message.content
    t_total = time.time() - t0
    st.session_state["history"].append({"time": time.strftime("%H:%M"), "query": question})
    st.subheader("Compliance Assessment")
    st.markdown(answer)
    st.caption(
        f"Retrieval: {t_retrieve:.1f}s | Total: {t_total:.1f}s | "
        f"Retrieved: {len(docs)} → After cutoff: {len(docs_f)} → After budget: {len(docs_b)} (~{used_tokens} tokens)"
    )
    st.markdown("---")
    st.subheader("Retrieved Source Provisions")
    for i, (doc, meta, dist) in enumerate(zip(docs_b, metas_b, dists_b)):
        citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
        with st.expander(f"[{i+1}] {citation} | dist: {dist:.3f}", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                if meta.get("article_number"): st.caption(f"Article: {meta['article_number']}")
            with cols[1]:
                if meta.get("annex_ref"): st.caption(f"Annex: {meta['annex_ref']}")
            with cols[2]:
                if meta.get("recital_ref"): st.caption(f"Recital: {meta['recital_ref']}")
            st.markdown(highlight_terms(doc, question))
    st.markdown("---")
    st.caption("⚠️ This tool provides preliminary guidance only and does not constitute legal advice.")
