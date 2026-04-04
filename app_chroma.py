import os, time, re, json, math, streamlit as st
from openai import OpenAI

st.set_page_config(page_title="EU AI Act Compliance Navigator", page_icon="⚖️", layout="wide")

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

LANG_OPTIONS = {
    "English": "Respond in English.",
    "Français": "Réponds en français.",
    "Deutsch": "Antworte auf Deutsch.",
    "Español": "Responde en español.",
    "简体中文": "请用简体中文回答。",
    "繁體中文": "請用繁體中文回答。",
}

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def search(query_emb, top_k=10, where=None):
    results = []
    for item in store:
        if where:
            match = False
            for key, cond in where.items():
                if key == "$or":
                    for c in cond:
                        for mk, mv in c.items():
                            val = mv.get("$eq","") if isinstance(mv, dict) else mv
                            if val and val in str(item["metadata"].get(mk,"")):
                                match = True
                else:
                    val = cond.get("$eq","") if isinstance(cond, dict) else cond
                    if val and val in str(item["metadata"].get(key,"")):
                        match = True
            if not match: continue
        sim = cosine_sim(query_emb, item["embedding"])
        results.append({"doc": item["document"], "meta": item["metadata"], "score": sim})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def extract_legal_references(query):
    refs = {"has_references": False}
    art = re.search(r'Article\s+(\d+)', query, re.IGNORECASE)
    if art: refs["article"] = f"Article {art.group(1)}"; refs["has_references"] = True
    anx = re.search(r'Annex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII)', query, re.IGNORECASE)
    if anx: refs["annex"] = f"Annex {anx.group(1)}"; refs["has_references"] = True
    rec = re.search(r'Recital\s+(\d+)', query, re.IGNORECASE)
    if rec: refs["recital"] = f"Recital {rec.group(1)}"; refs["has_references"] = True
    return refs

def estimate_tokens(text): return len(text) // 4

def apply_token_budget(items, budget=6000):
    selected, total = [], 0
    for item in items:
        t = estimate_tokens(item["doc"])
        if total + t > budget and selected: break
        selected.append(item); total += t
    return selected, total

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚖️ EU AI Act Navigator")
    lang = st.selectbox("Language / 语言", list(LANG_OPTIONS.keys()), index=0)
    top_k = st.selectbox("Retrieval count", [5, 8, 10, 15], index=2)
    token_budget = st.selectbox("Token budget", [3000, 6000, 8000, 10000], index=1)
    st.markdown("---")
    st.caption(f"Records: {len(store)} | Embedding: text-embedding-3-small | LLM: GPT-4o-mini")
    st.markdown("---")
    st.markdown("**Quick queries**")
    quick = {
        "⚡ High-risk classification": "What does Article 6 say about high-risk AI classification?",
        "📋 HR screening tool": "An AI system that screens job applicants' CVs and ranks candidates.",
        "📜 Provider obligations": "What obligations do providers of high-risk AI systems have?",
        "📎 Annex III list": "What AI systems are listed in Annex III?",
        "🔍 Transparency": "What transparency requirements apply to AI systems?",
        "🚫 Prohibited practices": "What AI practices are prohibited under the EU AI Act?",
        "🏥 Medical AI": "A machine learning model that analyzes chest X-rays to detect pneumonia.",
        "🎯 Biometrics + Annex III": "Does my biometric identification system fall under Annex III?",
        "⚙️ Article 9 risk mgmt": "What does Article 9 require for risk management?",
    }
    for label, q in quick.items():
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Display chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Retrieved provisions", expanded=False):
                st.markdown(msg["sources"])

# ---- Chat input ----
if prompt := st.chat_input("Describe your AI system or ask about the EU AI Act..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.time()

        # Self-Query routing
        refs = extract_legal_references(prompt)
        where_filter = None
        if refs["has_references"]:
            conditions = []
            if refs.get("article"): conditions.append({"article_number": {"$eq": refs["article"]}})
            if refs.get("annex"): conditions.append({"annex_ref": {"$eq": refs["annex"]}})
            if refs.get("recital"): conditions.append({"recital_ref": {"$eq": refs["recital"]}})
            where_filter = conditions[0] if len(conditions) == 1 else {"$or": conditions}
            detected = ", ".join(v for v in [refs.get("article"), refs.get("annex"), refs.get("recital")] if v)
            st.caption(f"🎯 Detected: {detected} → metadata-filtered search")
        else:
            st.caption("🔍 Full vector search")

        # Retrieve
        with st.spinner("Searching..."):
            qr = client.embeddings.create(model="openai/text-embedding-3-small", input=prompt)
            results = search(qr.data[0].embedding, top_k=top_k, where=where_filter)
        t_retrieve = time.time() - t0

        budgeted, used_tokens = apply_token_budget(results, budget=token_budget)
        context = "\n\n---\n\n".join([f"[{item['meta'].get('canonical_citation','N/A')}]\n{item['doc']}" for item in budgeted])

        lang_instruction = LANG_OPTIONS.get(lang, "Respond in English.")

        # Generate
        with st.spinner("Generating assessment..."):
            llm_response = client.chat.completions.create(
                model="openai/gpt-4o-mini", temperature=0.1,
                messages=[
                    {"role": "system", "content": (
                        f"{lang_instruction}\n\n"
                        "You are an EU AI Act compliance analyst. Based ONLY on the retrieved provisions below, "
                        "provide a structured assessment:\n\n"
                        "## Risk Classification\n[classification + reasoning]\n\n"
                        "## Applicable Legal Basis\n[articles with explanations]\n\n"
                        "## Key Compliance Obligations\n[numbered list with article references]\n\n"
                        "## Cross-Regulatory Considerations\n[overlaps if found]\n\n"
                        "## Information Gaps\n[what was not covered]\n\n"
                        "Cite specific articles for every claim. Do NOT use external knowledge."
                    )},
                    {"role": "user", "content": f"RETRIEVED PROVISIONS:\n{context}\n\nQUESTION:\n{prompt}"}
                ],
            )
            answer = llm_response.choices[0].message.content
        t_total = time.time() - t0

        st.markdown(answer)
        st.caption(f"⏱ {t_total:.1f}s | {len(budgeted)} provisions (~{used_tokens} tokens)")

        # Build sources text
        sources_md = ""
        for i, item in enumerate(budgeted):
            meta = item["meta"]
            citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
            sources_md += f"**[{i+1}] {citation}** (sim: {item['score']:.3f})\n\n"
            if meta.get("article_number"): sources_md += f"Article: {meta['article_number']}  \n"
            if meta.get("annex_ref"): sources_md += f"Annex: {meta['annex_ref']}  \n"
            if meta.get("recital_ref"): sources_md += f"Recital: {meta['recital_ref']}  \n"
            sources_md += f"\n{item['doc'][:300]}...\n\n---\n\n"

        with st.expander("📄 Retrieved provisions", expanded=False):
            st.markdown(sources_md)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources_md,
        })

    st.caption("⚠️ This tool provides preliminary guidance only and does not constitute legal advice.")
