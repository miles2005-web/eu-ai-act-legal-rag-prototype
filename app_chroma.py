import os, time, re, json, math, streamlit as st
from openai import OpenAI

st.set_page_config(page_title="EU AI Act Compliance Navigator", page_icon="⚖️", layout="wide")
st.title("⚖️ EU AI Act Compliance Navigator")
st.write("Describe your AI system or ask about the EU AI Act. Mention specific articles or annexes for precision retrieval.")

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
if not api_key:
    st.error("Set OPENROUTER_API_KEY.")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

@st.cache_data
def load_vectors():
    with open("vector_store.json") as f:
        return json.load(f)

store = load_vectors()

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def search(query_emb, top_k=5, where=None):
    results = []
    for item in store:
        if where:
            match = False
            for key, cond in where.items():
                if key == "$or":
                    for c in cond:
                        for mk, mv in c.items():
                            if isinstance(mv, dict):
                                val = mv.get("$eq","")
                            else:
                                val = mv
                            if val and val in str(item["metadata"].get(mk,"")):
                                match = True
                else:
                    if isinstance(cond, dict):
                        val = cond.get("$eq","")
                    else:
                        val = cond
                    if val and val in str(item["metadata"].get(key,"")):
                        match = True
            if not match:
                continue
        sim = cosine_sim(query_emb, item["embedding"])
        results.append({"doc": item["document"], "meta": item["metadata"], "score": sim})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

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

def apply_token_budget(items, budget):
    selected = []
    total = 0
    for item in items:
        t = estimate_tokens(item["doc"])
        if total + t > budget and selected:
            break
        selected.append(item)
        total += t
    return selected, total

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
    show_routing = st.checkbox("Show Self-Query routing info", value=True)
    st.markdown("---")
    st.header("System Info")
    st.write(f"Records: {len(store)}")
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
        query_emb = qr.data[0].embedding
        results = search(query_emb, top_k=top_k, where=where_filter)
    t_retrieve = time.time() - t0
    budgeted, used_tokens = apply_token_budget(results, token_budget)
    context = "\n\n---\n\n".join([
        f"[{item['meta'].get('canonical_citation','N/A')}]\n{item['doc']}" for item in budgeted
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
    st.caption(f"Retrieval: {t_retrieve:.1f}s | Total: {t_total:.1f}s | Provisions: {len(budgeted)} (~{used_tokens} tokens)")
    st.markdown("---")
    st.subheader("Retrieved Source Provisions")
    for i, item in enumerate(budgeted):
        meta = item["meta"]
        citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
        with st.expander(f"[{i+1}] {citation} | sim: {item['score']:.3f}", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                if meta.get("article_number"): st.caption(f"Article: {meta['article_number']}")
            with cols[1]:
                if meta.get("annex_ref"): st.caption(f"Annex: {meta['annex_ref']}")
            with cols[2]:
                if meta.get("recital_ref"): st.caption(f"Recital: {meta['recital_ref']}")
            st.markdown(highlight_terms(item["doc"], question))
    st.markdown("---")
    st.caption("⚠️ This tool provides preliminary guidance only and does not constitute legal advice.")
