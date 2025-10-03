# app.py
# GenAI-Tutor — Robust Agentic System (Planner-only tools → Final Synthesis) + RAG + Web + LangSmith (HF-only)

import os, io, re, json, hashlib, requests
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ddgs import DDGS  # web search
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient

from langsmith import Client
from langsmith.run_helpers import tracing_context, trace, get_current_run_tree

# =========================
# Streamlit & Secrets
# =========================
st.set_page_config(page_title="GenAI-Tutor (Agentic, Robust)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Robust Agentic Tutor (HF-only)</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN in Streamlit Secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
os.environ["LANGSMITH_TRACING"] = str(st.secrets.get("LANGSMITH_TRACING", True)).lower()
LS_PROJECT = st.secrets.get("LANGSMITH_PROJECT", os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-Agentic"))
ls_client = Client()

# =========================
# Models & Scenarios
# =========================
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

SCENARIOS: Dict[str, Dict[str, str]] = {
    "Gen-AI Fundamentals": {
        "overview": "Core definitions, model types, tokens, safety basics, and use-cases.",
        "system": "You are GenAI-Tutor, an expert coach for employees learning Gen-AI. Be precise, practical, and safe."
    },
    "Responsible & Secure GenAI at Work": {
        "overview": "Policies, data minimization, privacy, IP, bias, and phishing/social-engineering risks.",
        "system": "You are GenAI-Tutor for responsible, secure Gen-AI usage at work. Emphasize checklists and safety."
    },
    "Prompt Engineering Basics": {
        "overview": "Role/task/context/constraints, few-shot, chain-of-thought, structure, hallucination reduction.",
        "system": "You are GenAI-Tutor for prompt engineering. Provide practical patterns and small exercises."
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# =========================
# RAG corpus (curated, public)
# =========================
DOC_LINKS = [
    {"title": "Ethical & Regulatory Challenges of GenAI in Education (2025) — Frontiers",
     "url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1565938/full", "enabled": True},
    {"title": "Learn Your Way: Reimagining Textbooks with Generative AI (2025) — Google",
     "url": "https://blog.google/outreach-initiatives/education/learn-your-way/", "enabled": True},
    {"title": "Student Generative AI Survey 2025 — HEPI",
     "url": "https://www.hepi.ac.uk/reports/student-generative-ai-survey-2025/", "enabled": True},
    {"title": "Educational impacts of generative AI on learning & performance (2025) — Nature (PDF)",
     "url": "https://www.nature.com/articles/s41598-025-06930-w.pdf", "enabled": True},
    {"title": "Enhancing Retrieval-Augmented Generation: Best Practices — COLING 2025 (PDF)",
     "url": "https://aclanthology.org/2025.coling-main.449.pdf", "enabled": True},
]

# =========================
# RAG plumbing
# =========================
TOP_K = 7
K_CANDIDATES = 30
WORDS_PER_CHUNK = 450
OVERLAP_WORDS = 80

@st.cache_data(show_spinner=False)
def _download(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.content, (r.headers.get("Content-Type", "")).lower()

def _clean_html(html_bytes: bytes) -> str:
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        for t in soup(["script","style","noscript","header","footer","nav","form"]): t.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = html_bytes.decode("utf-8", errors="ignore")
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())

def _clean_pdf(pdf_bytes: bytes) -> str:
    pages = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(ln.strip() for ln in "\n".join(pages).splitlines() if ln.strip())

def fetch_and_clean(url: str) -> str:
    try:
        blob, ctype = _download(url)
        if ".pdf" in url.lower() or "application/pdf" in ctype:
            return _clean_pdf(blob)
        return _clean_html(blob)
    except Exception:
        return ""

def chunk_text(text: str, url: str, title: str) -> List[Dict[str, Any]]:
    if not text: return []
    words = text.split()
    chunks, start, k = [], 0, 0
    while start < len(words):
        end = min(start+WORDS_PER_CHUNK, len(words))
        piece = " ".join(words[start:end])
        chunk_id = f"{hashlib.sha1(url.encode()).hexdigest()}#{k:04d}"
        chunks.append({"chunk_id":chunk_id,"title":title,"url":url,"text":piece})
        if end == len(words): break
        start = max(0, end-OVERLAP_WORDS); k += 1
    return chunks

@st.cache_resource(show_spinner=True)
def load_embedder(): return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=True)
def load_reranker(): return CrossEncoder("BAAI/bge-reranker-v2-m3")

def embed_texts(texts: List[str], model):
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")

@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    import faiss
    d = vectors.shape[1]; index = faiss.IndexFlatIP(d); index.add(vectors); return index

@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, Any]]):
    embedder = load_embedder()
    all_chunks = []
    for doc in doc_links:
        if not doc.get("enabled", True): continue
        raw = fetch_and_clean(doc["url"])
        if not raw: continue
        all_chunks.extend(chunk_text(raw, doc["url"], doc["title"]))
    if not all_chunks: raise RuntimeError("No chunks ingested from sources.")
    vectors = embed_texts([c["text"] for c in all_chunks], embedder)
    index = build_faiss(vectors)
    side = {"chunks": all_chunks, "vectors_shape": vectors.shape}
    return index, side

def retrieve(query: str, index, side: Dict[str, Any]) -> List[Dict[str, Any]]:
    import faiss
    embedder, reranker = load_embedder(), load_reranker()
    qv = embed_texts([query], embedder)
    scores, idxs = index.search(qv, K_CANDIDATES)
    candidates = []
    for rank, (ci, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        if ci < 0: continue
        c = side["chunks"][ci]
        candidates.append({"rank_ann":rank,"score_ann":float(s),**c})
    if not candidates: return []
    pairs = [(query, c["text"]) for c in candidates]
    rers = reranker.predict(pairs, batch_size=64).tolist()
    for c, rs in zip(candidates, rers): c["score_rerank"] = float(rs)
    return sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)[:TOP_K]

# Build global RAG on import (fails silently)
_global_rag = {"index": None, "side": None}
try:
    idx0, side0 = build_rag_index(DOC_LINKS)
    _global_rag["index"], _global_rag["side"] = idx0, side0
except Exception:
    pass

def ensure_rag_ready():
    if _global_rag["index"] is None or _global_rag["side"] is None:
        idx, side = build_rag_index(DOC_LINKS)
        _global_rag["index"], _global_rag["side"] = idx, side

# =========================
# Web search (ddgs)
# =========================
@st.cache_resource(show_spinner=False)
def get_ddg(): return DDGS()

def web_search(query: str, max_results: int = 8, region: str = "wt-wt", lang: str = "en") -> List[Dict[str, str]]:
    results = []
    try:
        ddg = get_ddg()
        for r in ddg.text(keywords=query, max_results=max_results, region=region, safesearch="moderate"):
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            snippet = r.get("body") or ""
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
    except Exception:
        return []
    return results

# =========================
# Tools (callable by planner)
# =========================
def _extract_json_block(text: str) -> Dict[str, Any]:
    try:
        m = re.search(r"\{.*\}", text, flags=re.S)
    except re.error:
        m = None
    if not m:
        return {}
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        # attempt trailing comma cleanup
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return {}

def _summarize_text_for_obs(s: str, limit: int = 1000) -> str:
    return s[:limit] + ("…" if len(s) > limit else "")

def tool_web_search(inp: str) -> Dict[str, Any]:
    """Search the web. Input: natural text or JSON {query,max_results}. Output: {'results': [...]} or {'note': ...}"""
    data = _extract_json_block(inp) or {}
    query = data.get("query") or inp.strip()
    max_results = int(data.get("max_results", 8)) if str(data.get("max_results","")).isdigit() else 8

    res = web_search(query, max_results=max_results)
    if not res:
        res = web_search(query + " site:openai.com OR site:nature.com OR site:aclanthology.org OR site:research.google", max_results=max_results)
    return {"results": res} if res else {"note": "No results found."}

def tool_read_url(inp: str) -> Dict[str, Any]:
    """Read a URL (HTML or PDF). Input: URL or JSON {url,max_chars}. Output: {'title','url','text'}"""
    data = _extract_json_block(inp) or {}
    url = data.get("url") or inp.strip()
    max_chars = int(data.get("max_chars", 9000)) if str(data.get("max_chars","")).isdigit() else 9000
    try:
        r = requests.get(url, headers={"User-Agent":"TutorAI/1.0"}, timeout=25)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type","")).lower()
        text = _clean_pdf(r.content) if ("pdf" in ctype or url.lower().endswith(".pdf")) else _clean_html(r.content)
        return {"title": url, "url": url, "text": text[:max_chars]}
    except Exception as e:
        return {"title": url, "url": url, "text": f"[Error: {e}]"}

def tool_rag_retrieve(inp: str) -> Dict[str, Any]:
    """Retrieve from curated Gen-AI corpus. Input: natural text or JSON {query,top_k}. Output: {'results': [...]} or {'note': ...}"""
    try:
        ensure_rag_ready()
    except Exception as e:
        return {"note": f"RAG index unavailable: {e}"}
    data = _extract_json_block(inp) or {}
    q = data.get("query") or inp.strip()
    try:
        k = int(data.get("top_k", TOP_K)); k = max(1, min(10, k))
    except:
        k = TOP_K
    results = retrieve(q, _global_rag["index"], _global_rag["side"])[:k]
    if not results and k < 9:  # adaptive depth
        results = retrieve(q, _global_rag["index"], _global_rag["side"])[:9]
    if not results:
        return {"note":"No evidence retrieved from corpus."}
    out = [{"title":c["title"], "url":c["url"], "chunk":c["text"][:900], "score":float(c.get("score_rerank",0.0))} for c in results]
    return {"results": out}

TOOLS = {
    "rag_retrieve": tool_rag_retrieve,
    "web_search": tool_web_search,
    "read_url": tool_read_url,
}

# =========================
# HF chat wrapper
# =========================
def hf_chat(model: str, messages: List[Dict[str,str]], max_new_tokens=512, temperature=0.3, top_p=0.9) -> str:
    client = InferenceClient(model=model, token=HF_TOKEN)
    resp = client.chat_completion(messages=messages, max_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    choice = resp.choices[0]
    msg = getattr(choice, "message", None) or choice["message"]
    content = getattr(msg, "content", None) or msg["content"]
    return (content or "").strip()

# =========================
# Planner-only Controller (NO final during planning)
# =========================
def controller_system_text(scenario_sys: str) -> str:
    return (
        "You are the Planner for GenAI-Tutor. Your job is to decide which tools to call to gather evidence. "
        "Do NOT produce the final answer. Keep planning until you have enough evidence (or no useful tools remain), "
        "then return a stop signal.\n\n"
        "Available tools (call names exactly): rag_retrieve, web_search, read_url.\n"
        "RETURN FORMAT (MUST be a single JSON object, no extra text):\n"
        '1) To call a tool: { "tool": "rag_retrieve" | "web_search" | "read_url", "input": "<single string or JSON string>" }\n'
        '2) To stop planning: { "stop": true, "note": "<why you stopped>" }\n\n'
        "Guidelines:\n"
        "- Prefer rag_retrieve for conceptual/how-to/pedagogy/policy questions within the curated corpus.\n"
        "- Use web_search for recent news, pricing, product changes, or topics likely outside the corpus.\n"
        "- After a web_search, consider read_url on the most authoritative link(s).\n"
        "- Avoid duplicate work; request only new, relevant evidence. Do not finalize here."
    )

# =========================
# Evidence Aggregator
# =========================
class EvidenceStore:
    def __init__(self):
        self.search_hits: List[Dict[str,str]] = []
        self.pages: Dict[str, Dict[str,str]] = {}  # url -> {title,url,text}
        self.rag_chunks: List[Dict[str,Any]] = []
        self.seen_urls = set()
        self.seen_hashes = set()

    def add_search(self, items: List[Dict[str,str]]):
        added = 0
        for it in items:
            u = it.get("url","")
            if not u: continue
            if u not in self.seen_urls:
                self.seen_urls.add(u)
                self.search_hits.append(it)
                added += 1
        return added

    def add_page(self, page: Dict[str,str]):
        u = page.get("url","")
        txt = page.get("text","")
        if not u: return 0
        h = hashlib.sha1((u + txt[:1000]).encode()).hexdigest()
        if h in self.seen_hashes: return 0
        self.seen_hashes.add(h)
        self.pages[u] = {"title": page.get("title",u), "url": u, "text": txt}
        return 1

    def add_rag(self, items: List[Dict[str,Any]]):
        added = 0
        for it in items:
            u = it.get("url","")
            chunk = it.get("chunk","")
            h = hashlib.sha1((u + chunk[:400]).encode()).hexdigest()
            if h in self.seen_hashes: continue
            self.seen_hashes.add(h)
            self.rag_chunks.append(it)
            added += 1
        return added

    def context_pack(self, max_chars: int = 16000) -> Tuple[str, List[Dict[str,str]]]:
        """
        Build a compact context string with numbered citations.
        Returns (context_text, citations_list)
        """
        citations: List[Dict[str,str]] = []
        lines = []

        # RAG chunks
        url_to_num = {}
        def cite_for(url, title):
            if url not in url_to_num:
                url_to_num[url] = len(url_to_num) + 1
                citations.append({"title": title or "source", "url": url})
            return url_to_num[url]

        if self.rag_chunks:
            lines.append("### RAG Evidence")
            for it in self.rag_chunks[:10]:
                r = cite_for(it.get("url",""), it.get("title","source"))
                snippet = it.get("chunk","")
                snippet = snippet[:700] + ("…" if len(snippet) > 700 else "")
                lines.append(f"[{r}] {it.get('title','source')}\n{snippet}\n")

        # Web pages
        if self.pages:
            lines.append("### Web Pages")
            # keep up to 6 pages
            for u, p in list(self.pages.items())[:6]:
                r = cite_for(u, p.get("title",u))
                snippet = (p.get("text","") or "")[:900]
                snippet = snippet + ("…" if len(p.get("text","")) > 900 else "")
                lines.append(f"[{r}] {p.get('title',u)}\n{snippet}\n")

        # Web search (snippets) if page not fetched
        if self.search_hits:
            lines.append("### Web Search Snippets")
            kept = 0
            for hit in self.search_hits:
                if kept >= 6: break
                u = hit.get("url",""); t = hit.get("title","source")
                if u in self.pages: continue
                r = cite_for(u, t)
                snippet = (hit.get("snippet","") or "")[:400]
                lines.append(f"[{r}] {t}\n{snippet}\n")
                kept += 1

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n…"
        return text, citations

# =========================
# Run Planner Loop → Gather Evidence → Final Synthesis
# =========================
def run_agent(user_input: str, model_id: str, scenario_sys: str, max_steps: int = 8, stagnation_patience: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Planning loop where LLM ONLY calls tools or stops; never finalizes here.
    After loop, we compile all evidence and run a final synthesis LLM call with full context.
    Returns (final_answer, step_trace).
    """
    steps = []
    ev = EvidenceStore()
    msgs = [
        {"role":"system","content": controller_system_text(scenario_sys)},
        {"role":"user","content": f"User question: {user_input}"}
    ]

    no_new_ctr = 0
    total_added = 0

    for step in range(1, max_steps+1):
        with trace("planner_step", run_type="chain", inputs={"step": step}):
            planner_out = hf_chat(model_id, msgs, max_new_tokens=256, temperature=0.2)
            steps.append({"role":"assistant","content": planner_out})
            parsed = _extract_json_block(planner_out)

            if parsed.get("stop") is True:
                steps.append({"role":"tool","tool":"_STOP","content": parsed.get("note","planner stop")})
                break

            tool = parsed.get("tool")
            tool_inp = parsed.get("input","")
            if tool not in TOOLS:
                # Tell planner it was invalid and continue
                msgs.append({"role":"assistant","content": planner_out})
                msgs.append({"role":"user","content": json.dumps({"note": f"Invalid or missing tool. Try another tool or stop."})})
                no_new_ctr += 1
                if no_new_ctr >= stagnation_patience:
                    break
                continue

            # Execute tool
            try:
                with trace(f"tool:{tool}", run_type="tool", inputs={"input": tool_inp}) as tspan:
                    result = TOOLS[tool](tool_inp)
                    tspan.outputs = {"result": result}
            except Exception as e:
                result = {"error": f"{tool} failed: {e}"}

            obs = json.dumps(result, ensure_ascii=False)
            steps.append({"role":"tool","tool":tool,"content": _summarize_text_for_obs(obs)})

            # Aggregate evidence
            added = 0
            if tool == "web_search" and "results" in result:
                added += ev.add_search(result["results"])
            elif tool == "read_url" and "text" in result:
                added += ev.add_page(result)
            elif tool == "rag_retrieve" and "results" in result:
                added += ev.add_rag(result["results"])

            total_added += added
            no_new_ctr = 0 if added > 0 else no_new_ctr + 1

            # Feed observation back to planner
            msgs.append({"role":"assistant","content": planner_out})
            msgs.append({"role":"user","content": f"Observation:\n{_summarize_text_for_obs(obs)}\nConsider next step. If you have enough, return stop."})

            if no_new_ctr >= stagnation_patience:
                break

    # Final synthesis with COMPLETE evidence bundle
    context_text, cites = ev.context_pack(max_chars=16000)
    cite_lines = "\n".join([f"- [{c['title']}]({c['url']})" for c in cites if c.get("url")])
    final_sys = (
        f"{scenario_sys}\n"
        "You will now answer the user's question using ONLY the provided EVIDENCE below. "
        "Be concise, factual, and cite claims with [n] markers that map to the Sources list. "
        "If evidence is insufficient or ambiguous, say so and explain what else would be needed.\n"
    )
    final_user = (
        f"USER QUESTION:\n{user_input}\n\n"
        f"EVIDENCE:\n{context_text}\n\n"
        "Write the final answer with inline [n] citations and conclude with a 'Sources' list."
    )
    final_answer = hf_chat(
        model_id,
        [
            {"role":"system","content": final_sys},
            {"role":"user","content": final_user}
        ],
        max_new_tokens=700,
        temperature=0.3
    )
    if cite_lines:
        final_answer += "\n\n**Sources:**\n" + cite_lines

    return final_answer, steps

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    max_steps = st.slider("Max planning/tool steps", 1, 12, 8)
    stagnation_patience = st.slider("No-new-evidence patience", 1, 5, 3)
    st.markdown("---")
    st.subheader("📚 RAG Corpus")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build/Refresh RAG Index", use_container_width=True):
            try:
                idx, side = build_rag_index(DOC_LINKS)
                _global_rag["index"], _global_rag["side"] = idx, side
                st.success(f"RAG index built: {side['vectors_shape'][0]} chunks.")
            except Exception as e:
                st.error(f"RAG build failed: {e}")
    with c2:
        st.caption("Curated sources; top-k=7 + reranking. Planner decides if/when to use RAG.")

st.caption(f"Model: **{model_id}**  •  Scenario: **{scenario_name}**")

# =========================
# Chat UI
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"system","content":SCENARIOS[scenario_name]["system"]},
        {"role":"assistant","content":"Hi! I’m GenAI-Tutor. Ask me anything about Gen-AI."}
    ]
if "scenario_prev" not in st.session_state:
    st.session_state.scenario_prev = scenario_name

def _seed_chat():
    st.session_state.messages = [
        {"role":"system","content":SCENARIOS[scenario_name]["system"]},
        {"role":"assistant","content":"Hi! I’m GenAI-Tutor. Ask me anything about Gen-AI."}
    ]

if st.session_state.scenario_prev != scenario_name:
    _seed_chat(); st.session_state.scenario_prev = scenario_name

st.markdown("---")
st.subheader("💬 Tutor Chat (Planner tools → Final Synthesis)")

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

user_q = st.chat_input("Ask a question. The planner may use RAG, Web Search, or ReadURL, then synthesize.")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    with tracing_context(project_name=LS_PROJECT, metadata={"type":"agent_turn","scenario":scenario_name,"model":model_id}):
        with trace("agent_turn", run_type="chain", inputs={"question": user_q, "max_steps": max_steps, "stagnation_patience": stagnation_patience}):
            with st.chat_message("assistant"):
                with st.spinner("Planning with tools → synthesizing…"):
                    try:
                        answer, step_trace = run_agent(
                            user_input=user_q,
                            model_id=model_id,
                            scenario_sys=SCENARIOS[scenario_name]["system"],
                            max_steps=max_steps,
                            stagnation_patience=stagnation_patience
                        )
                    except Exception as e:
                        answer, step_trace = f"⚠️ Agent failed: {e}", []

                st.markdown(answer)
                if step_trace:
                    with st.expander("🔍 Planner Trace (tool JSON + observations)"):
                        for i, step in enumerate(step_trace, start=1):
                            if step.get("role") == "assistant":
                                st.markdown(f"**Step {i}: Planner JSON**")
                                st.code(step["content"], language="json")
                            elif step.get("role") == "tool":
                                st.markdown(f"**Step {i}: Tool `{step.get('tool')}` observation**")
                                st.text_area("Observation", value=step["content"], height=160, key=f"obs_{i}_{step.get('tool')}")

    st.session_state.messages.append({"role":"assistant","content":answer})

# =========================
# Scenario Overview
# =========================
st.markdown("---")
st.subheader("📌 Scenario Overview")
st.markdown(f"**{scenario_name}**  \n{SCENARIOS[scenario_name]['overview']}")

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("GenAI-Tutor is educational. Verify critical info. Follow your organization’s policies.")
