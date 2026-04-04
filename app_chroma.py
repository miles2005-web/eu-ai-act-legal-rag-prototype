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

LANGS = {
    "English": {"instruction": "Respond in English.", "title": "EU AI Act Compliance Navigator", "desc": "Describe your AI system or ask about the EU AI Act.", "placeholder": "Describe your AI system or ask about the EU AI Act...", "analyze": "Compliance Assessment", "sources": "Retrieved provisions", "disclaimer": "This tool provides preliminary guidance only and does not constitute legal advice.", "searching": "Searching...", "generating": "Generating assessment..."},
    "Français": {"instruction": "Réponds en français.", "title": "Navigateur de conformité EU AI Act", "desc": "Décrivez votre système d'IA ou posez une question sur l'EU AI Act.", "placeholder": "Décrivez votre système d'IA...", "analyze": "Évaluation de conformité", "sources": "Dispositions récupérées", "disclaimer": "Cet outil fournit des orientations préliminaires uniquement et ne constitue pas un avis juridique.", "searching": "Recherche...", "generating": "Génération de l'évaluation..."},
    "Deutsch": {"instruction": "Antworte auf Deutsch.", "title": "EU AI Act Compliance Navigator", "desc": "Beschreiben Sie Ihr KI-System oder fragen Sie zum EU AI Act.", "placeholder": "Beschreiben Sie Ihr KI-System...", "analyze": "Konformitätsbewertung", "sources": "Abgerufene Bestimmungen", "disclaimer": "Dieses Tool bietet nur vorläufige Orientierung und stellt keine Rechtsberatung dar.", "searching": "Suche...", "generating": "Bewertung wird erstellt..."},
    "Español": {"instruction": "Responde en español.", "title": "Navegador de cumplimiento EU AI Act", "desc": "Describa su sistema de IA o pregunte sobre la EU AI Act.", "placeholder": "Describa su sistema de IA...", "analyze": "Evaluación de cumplimiento", "sources": "Disposiciones recuperadas", "disclaimer": "Esta herramienta proporciona orientación preliminar y no constituye asesoramiento jurídico.", "searching": "Buscando...", "generating": "Generando evaluación..."},
    "简体中文": {"instruction": "请用简体中文回答。", "title": "欧盟AI法案合规导航", "desc": "描述你的AI系统或询问欧盟AI法案相关问题。", "placeholder": "描述你的AI系统或询问欧盟AI法案...", "analyze": "合规评估", "sources": "检索到的条款", "disclaimer": "本工具仅提供初步指导，不构成法律建议。", "searching": "检索中...", "generating": "生成评估中..."},
    "繁體中文": {"instruction": "請用繁體中文回答。", "title": "歐盟AI法案合規導航", "desc": "描述你的AI系統或詢問歐盟AI法案相關問題。", "placeholder": "描述你的AI系統或詢問歐盟AI法案...", "analyze": "合規評估", "sources": "檢索到的條款", "disclaimer": "本工具僅提供初步指導，不構成法律建議。", "searching": "檢索中...", "generating": "生成評估中..."},
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

def apply_token_budget(items, budget=6000):
    selected, total = [], 0
    for item in items:
        t = len(item["doc"]) // 4
        if total + t > budget and selected: break
        selected.append(item); total += t
    return selected, total

def run_query(prompt, lang_key, top_k, token_budget):
    L = LANGS[lang_key]
    t0 = time.time()
    refs = extract_legal_references(prompt)
    where_filter = None
    route = ""
    if refs["has_references"]:
        conditions = []
        if refs.get("article"): conditions.append({"article_number": {"$eq": refs["article"]}})
        if refs.get("annex"): conditions.append({"annex_ref": {"$eq": refs["annex"]}})
        if refs.get("recital"): conditions.append({"recital_ref": {"$eq": refs["recital"]}})
        where_filter = conditions[0] if len(conditions) == 1 else {"$or": conditions}
        detected = ", ".join(v for v in [refs.get("article"), refs.get("annex"), refs.get("recital")] if v)
        route = f"🎯 {detected}"
    else:
        route = "🔍 Vector search"
    qr = client.embeddings.create(model="openai/text-embedding-3-small", input=prompt)
    results = search_store(qr.data[0].embedding, top_k=top_k, where=where_filter)
    budgeted, used_tokens = apply_token_budget(results, budget=token_budget)
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
    t_total = time.time() - t0
    sources_md = ""
    for i, item in enumerate(budgeted):
        meta = item["meta"]
        citation = meta.get("canonical_citation") or meta.get("article_number") or "N/A"
        sources_md += f"**[{i+1}] {citation}** (sim: {item['score']:.3f})\n\n"
        sources_md += f"{item['doc'][:300]}...\n\n---\n\n"
    return answer, sources_md, route, t_total, len(budgeted), used_tokens

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ⚖️ EU AI Act Navigator")
    lang = st.selectbox("Language / 语言", list(LANGS.keys()), index=0)
    top_k = st.selectbox("Retrieval count", [5, 8, 10, 15], index=2)
    token_budget = st.selectbox("Token budget", [3000, 6000, 8000, 10000], index=1)
    st.markdown("---")
    st.caption(f"Records: {len(store)} | LLM: GPT-4o-mini")
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

# ---- Title ----
st.title(f"⚖️ {L['title']}")
st.write(L["desc"])

# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📄 {L['sources']}", expanded=False):
                st.markdown(msg["sources"])
        if msg.get("meta_info"):
            st.caption(msg["meta_info"])

# ---- Determine input source ----
user_input = st.chat_input(L["placeholder"])

if st.session_state.get("pending_query"):
    user_input = st.session_state.pop("pending_query")

# ---- Process ----
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(L["generating"]):
            answer, sources_md, route, t_total, n_prov, used_tok = run_query(user_input, lang, top_k, token_budget)
        st.markdown(answer)
        meta_info = f"{route} | ⏱ {t_total:.1f}s | {n_prov} provisions (~{used_tok} tokens)"
        st.caption(meta_info)
        with st.expander(f"📄 {L['sources']}", expanded=False):
            st.markdown(sources_md)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources_md, "meta_info": meta_info})

st.caption(f"⚠️ {L['disclaimer']}")
