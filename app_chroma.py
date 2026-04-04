import os, time, re, json, math, streamlit as st
from openai import OpenAI

st.set_page_config(page_title="EU AI Act Compliance Navigator", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
.user-msg {
    display: flex; justify-content: flex-end; margin: 0.5rem 0;
}
.user-bubble {
    background: #2b5797; color: white; padding: 0.7rem 1rem;
    border-radius: 1rem 1rem 0.2rem 1rem; max-width: 75%; text-align: left;
}
.bot-msg {
    display: flex; justify-content: flex-start; margin: 0.5rem 0;
}
.bot-bubble {
    background: #333; color: #eee; padding: 0.7rem 1rem;
    border-radius: 1rem 1rem 1rem 0.2rem; max-width: 85%; text-align: left;
}
.chat-area { max-height: 70vh; overflow-y: auto; padding: 1rem 0; }
</style>
""", unsafe_allow_html=True)

api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key:
    try: api_key = st.secrets["OPENROUTER_API_KEY"]
    except: pass
if not api_key:
    st.error("API key not configured. Add OPENROUTER_API_KEY to Streamlit Secrets (Settings → Secrets).")
    st.stop()

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

@st.cache_data
def load_vectors():
    with open("vector_store.json") as f:
        return json.load(f)

store = load_vectors()

LANGS = {
    "English": {"instruction": "Respond in English.", "title": "EU AI Act Compliance Navigator", "desc": "Describe your AI system or ask about the EU AI Act.", "placeholder": "Describe your AI system or ask about the EU AI Act...", "sources": "Retrieved provisions", "disclaimer": "This tool provides preliminary guidance only and does not constitute legal advice.", "generating": "Generating assessment...", "clear": "Clear chat"},
    "Français": {"instruction": "Réponds en français.", "title": "Navigateur de conformité EU AI Act", "desc": "Décrivez votre système d'IA ou posez une question.", "placeholder": "Décrivez votre système d'IA...", "sources": "Dispositions récupérées", "disclaimer": "Orientations préliminaires uniquement.", "generating": "Génération...", "clear": "Effacer"},
    "Deutsch": {"instruction": "Antworte auf Deutsch.", "title": "EU AI Act Compliance Navigator", "desc": "Beschreiben Sie Ihr KI-System oder fragen Sie zum EU AI Act.", "placeholder": "Beschreiben Sie Ihr KI-System...", "sources": "Abgerufene Bestimmungen", "disclaimer": "Nur vorläufige Orientierung.", "generating": "Wird erstellt...", "clear": "Löschen"},
    "Español": {"instruction": "Responde en español.", "title": "Navegador de cumplimiento EU AI Act", "desc": "Describa su sistema de IA o pregunte.", "placeholder": "Describa su sistema de IA...", "sources": "Disposiciones recuperadas", "disclaimer": "Orientación preliminar.", "generating": "Generando...", "clear": "Borrar"},
    "简体中文": {"instruction": "请用简体中文回答。", "title": "欧盟AI法案合规导航", "desc": "描述你的AI系统或询问欧盟AI法案相关问题。", "placeholder": "描述你的AI系统或询问欧盟AI法案...", "sources": "检索到的条款", "disclaimer": "本工具仅提供初步指导，不构成法律建议。", "generating": "生成评估中...", "clear": "清除对话"},
    "繁體中文": {"instruction": "請用繁體中文回答。", "title": "歐盟AI法案合規導航", "desc": "描述你的AI系統或詢問歐盟AI法案。", "placeholder": "描述你的AI系統...", "sources": "檢索到的條款", "disclaimer": "僅提供初步指導。", "generating": "生成中...", "clear": "清除"},
}

def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0: return 0.0
    return dot / (na * nb)

def search_store(query_emb, top_k=10, where=None):
    results = []
    for item in store:
        if where:
            match = False
            for key, cond in where.items():
                if key == "$or":
                    for c in cond:
                        for mk, mv in c.items():
                            val = mv.get("$eq","") if isinstance(mv, dict) else mv
                            if val and val in str(item["metadata"].get(mk,"")): match = True
                else:
                    val = cond.get("$eq","") if isinstance(cond, dict) else cond
                    if val and val in str(item["metadata"].get(key,"")): match = True
            if not match: continue
        sim = cosine_sim(query_emb, item["embedding"])
        results.append({"doc": item["document"], "meta": item["metadata"], "score": sim})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def extract_legal_references(query):
    refs = {"has_references": False, "count": 0}
    art = re.search(r'Article\s+(\d+)', query, re.IGNORECASE)
    if art: refs["article"] = f"Article {art.group(1)}"; refs["has_references"] = True; refs["count"] += 1
    anx = re.search(r'Annex\s+(I{1,3}|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII)', query, re.IGNORECASE)
    if anx: refs["annex"] = f"Annex {anx.group(1)}"; refs["has_references"] = True; refs["count"] += 1
    rec = re.search(r'Recital\s+(\d+)', query, re.IGNORECASE)
    if rec: refs["recital"] = f"Recital {rec.group(1)}"; refs["has_references"] = True; refs["count"] += 1
    return refs

def auto_token_budget(query, refs, base=6000):
    budget = base
    if refs["count"] >= 2:
        budget = 10000
    long_keywords = ["obligations", "requirements", "prohibited", "classification", "transparency", "compliance"]
    if any(kw in query.lower() for kw in long_keywords):
        budget = max(budget, 8000)
    return budget

def apply_token_budget(items, budget=6000):
    selected, total = [], 0
    for item in items:
        t = len(item["doc"]) // 4
        if total + t > budget and selected: break
        selected.append(item); total += t
    return selected, total

def run_query(prompt, lang_key, top_k):
    L = LANGS[lang_key]
    refs = extract_legal_references(prompt)
    where_filter = None
    route = "🔍 Vector search"
    if refs["has_references"]:
        conditions = []
        if refs.get("article"): conditions.append({"article_number": {"$eq": refs["article"]}})
        if refs.get("annex"): conditions.append({"annex_ref": {"$eq": refs["annex"]}})
        if refs.get("recital"): conditions.append({"recital_ref": {"$eq": refs["recital"]}})
        where_filter = conditions[0] if len(conditions) == 1 else {"$or": conditions}
        detected = ", ".join(v for v in [refs.get("article"), refs.get("annex"), refs.get("recital")] if v)
        route = f"🎯 {detected}"
    budget = auto_token_budget(prompt, refs)
    qr = client.embeddings.create(model="openai/text-embedding-3-small", input=prompt)
    results = search_store(qr.data[0].embedding, top_k=top_k, where=where_filter)
    budgeted, used_tokens = apply_token_budget(results, budget=budget)
    context = "\n\n---\n\n".join([f"[{item['meta'].get('canonical_citation','N/A')}]\n{item['doc']}" for item in budgeted])
    llm_response = client.chat.completions.create(
        model="openai/gpt-4o-mini", temperature=0.1,
        messages=[
            {"role": "system", "content": (
                f"{L['instruction']}\n\n"
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
    sources_md = ""
    for i, item in enumerate(budgeted):
        meta = item["meta"]
        citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
        sources_md += f"**[{i+1}] {citation}** (sim: {item['score']:.3f})\n\n{item['doc'][:300]}...\n\n---\n\n"
    return answer, sources_md, route, len(budgeted), used_tokens, budget

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚖️ EU AI Act Navigator")
    lang = st.selectbox("Language / 语言", list(LANGS.keys()), index=0)
    top_k = st.selectbox("Retrieval count", [5, 8, 10, 15], index=2)
    st.caption("Token budget auto-adjusts based on query complexity")
    st.markdown("---")
    st.caption(f"Records: {len(store)} | LLM: GPT-4o-mini")
    st.markdown("---")
    L = LANGS[lang]
    if st.button(f"🗑 {L['clear']}", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
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
            st.session_state["pending_query"] = q

L = LANGS[lang]
st.title(f"⚖️ {L['title']}")
st.write(L["desc"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Render chat history with custom bubbles ----
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg"><div class="bot-bubble">', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {L['sources']}", expanded=False):
                st.markdown(msg["sources"])
        if msg.get("meta_info"):
            st.caption(msg["meta_info"])
        st.markdown('</div></div>', unsafe_allow_html=True)

# ---- Input ----
user_input = st.chat_input(L["placeholder"])
if st.session_state.get("pending_query"):
    user_input = st.session_state.pop("pending_query")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-msg"><div class="user-bubble">{user_input}</div></div>', unsafe_allow_html=True)

    t0 = time.time()
    with st.spinner(L["generating"]):
        answer, sources_md, route, n_prov, used_tok, budget = run_query(user_input, lang, top_k)
    t_total = time.time() - t0

    st.markdown(f'<div class="bot-msg"><div class="bot-bubble">', unsafe_allow_html=True)
    st.markdown(answer)
    meta_info = f"{route} | ⏱ {t_total:.1f}s | {n_prov} provisions (~{used_tok} tokens, budget: {budget})"
    st.caption(meta_info)
    with st.expander(f"📄 {L['sources']}", expanded=False):
        st.markdown(sources_md)
    st.markdown('</div></div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_md, "meta_info": meta_info})

st.caption(f"⚠️ {L['disclaimer']}")
