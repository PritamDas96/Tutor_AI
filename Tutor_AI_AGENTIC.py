# app.py
# GenAI-Tutor — Agentic (JSON Controller + Router) + RAG + Tools + LangSmith (HF-only)

import os, io, json, re, hashlib, requests
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient

# LangSmith
from langsmith import Client
from langsmith.run_helpers import tracing_context, trace, get_current_run_tree

# =========================
# Streamlit & Secrets
# =========================
st.set_page_config(page_title="GenAI-Tutor (Agentic + RAG)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Agentic + RAG (HF-only, JSON controller + router)</h1>", unsafe_allow_html=True)

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
    "Prompt Engineering Basics": {
        "overview": "Core prompting concepts, patterns, templates, hallucination reduction.",
        "system": "You are GenAI-Tutor, an expert coach on prompt engineering for employees. Be concise, practical, and safe."
    },
    "Responsible & Secure GenAI at Work": {
        "overview": "Safe inputs, approvals, phishing risks, policy alignment.",
        "system": "You are GenAI-Tutor for responsible GenAI usage at work. Teach checklist-driven guidance."
    },
    "Automating Everyday Tasks with GenAI": {
        "overview": "Drafting, ideation, structuring notes, time-saving workflows.",
        "system": "You are GenAI-Tutor for everyday task automation. Provide templates and quick workflows."
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# =========================
# RAG corpus (public)
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

# Global RAG (auto-build at startup; non-fatal)
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
# Tool helpers & tools
# =========================
def _extract_json_block(text: str) -> Dict[str, Any]:
    # Return the first {...} JSON object in text, if any
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
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return {}

def _summarize_text_for_obs(s: str, limit: int = 1200) -> str:
    return s[:limit] + ("…" if len(s) > limit else "")

def web_search(query: str, max_results: int = 5, region: str = "wt-wt", lang: str = "en") -> List[Dict[str, str]]:
    results = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=max_results, region=region, safesearch="moderate", backend="api"):
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            snippet = r.get("body") or ""
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
    return results

def tool_web_search(inp: str) -> Dict[str, Any]:
    """Search the web. Input: natural language or JSON {query,max_results,region,lang}. Output: {'results': [...]} or {'note': ...}"""
    data = _extract_json_block(inp) or {}
    query = data.get("query") or inp.strip()
    max_results = int(data.get("max_results", 5)) if str(data.get("max_results","")).isdigit() else 5
    region = data.get("region","wt-wt"); lang = data.get("lang","en")

    res = web_search(query, max_results=max_results, region=region, lang=lang)
    if not res:
        if "gen ai" in query.lower() or "genai" in query.lower():
            query2 = "what is generative ai definition site:ibm.com OR site:nvidia.com OR site:microsoft.com"
        else:
            query2 = query + " definition"
        res = web_search(query2, max_results=max_results, region="wt-wt", lang="en")
    return {"results": res} if res else {"note": "No results found. Consider rephrasing the query."}

def tool_read_url(inp: str) -> Dict[str, Any]:
    """Read a URL (HTML or PDF). Input: URL or JSON {url,max_chars}. Output: {'title','url','text'}"""
    data = _extract_json_block(inp) or {}
    url = data.get("url") or inp.strip()
    max_chars = int(data.get("max_chars", 6000)) if str(data.get("max_chars","")).isdigit() else 6000
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
    if not results and k < 7:  # adaptive depth
        results = retrieve(q, _global_rag["index"], _global_rag["side"])[:7]
    if not results:
        return {"note":"No evidence retrieved from corpus for this query."}
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
# Router + Answerability
# =========================
def classify_route(user_input: str, model_id: str, prefer_web: bool = False) -> str:
    """Return 'rag' | 'web' | 'direct'."""
    sys = (
        "Classify the best route to answer.\n"
        "Return ONLY one token: rag | web | direct.\n"
        "- rag: conceptual/how-to/education/pedagogy/policy about Gen-AI\n"
        "- web: current events or outside curated corpus\n"
        "- direct: short general explanation without tools\n"
        f"{'If ambiguous, prefer web.' if prefer_web else ''}"
    )
    out = hf_chat(model_id, [
        {"role":"system","content": sys},
        {"role":"user","content": f"User: {user_input}\nRoute:"}
    ], max_new_tokens=4, temperature=0.0)
    out = out.strip().lower()
    if "web" in out: return "web"
    if "rag" in out: return "rag"
    if "direct" in out: return "direct"
    return "rag"

def can_answer_direct(user_input: str, model_id: str) -> bool:
    """Yes/No if the model can answer confidently without tools."""
    sys = (
        "Answer with YES or NO only.\n"
        "YES if you can confidently answer the question without external tools.\n"
        "NO if you need retrieval or web to be accurate."
    )
    out = hf_chat(model_id, [
        {"role":"system","content": sys},
        {"role":"user","content": f"Question: {user_input}\nAnswer:"}
    ], max_new_tokens=2, temperature=0.0)
    return out.strip().lower().startswith("y")

# =========================
# JSON Controller System Prompt
# =========================
CONTROLLER_SYSTEM = (
    "You are GenAI-Tutor, a safe, practical Gen-AI coach.\n"
    "You can call tools or produce a final answer.\n"
    "Available tools (call names exactly): rag_retrieve, web_search, read_url.\n\n"
    "RETURN FORMAT (MUST be a single JSON object, no trailing text):\n"
    "1) To call a tool:\n"
    '{  "tool": "rag_retrieve" | "web_search" | "read_url",  "input": "<single string>" }\n'
    "2) To finalize:\n"
    '{  "final_answer": "<your answer text>",  "citations": [ {"title": "...", "url": "..."} ] }\n\n'
    "Notes:\n"
    "- Use the provided Route hint: if route=rag, call rag_retrieve first; if route=web, call web_search then read_url; if direct, finalize without tools.\n"
    "- If two tool calls return empty or notes, finalize with best effort and state limitations.\n"
    "- Prefer rag_retrieve for Gen-AI learning topics; then at most one web_search.\n"
    "- Cite sources you relied on. Keep answers concise and factual.\n"
)

# =========================
# Agent Controller
# =========================
def run_agent(user_input: str, model_id: str, max_steps: int, prefer_web: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (final_answer, trace_steps)
    trace_steps: list of {role, content} and tool observations for debug panel.
    """
    steps = []
    citations_pool: List[Dict[str,str]] = []
    empty_hits = 0

    # 0) Try direct answer quickly
    try:
        if can_answer_direct(user_input, model_id):
            direct = hf_chat(model_id, [
                {"role":"system","content":"Answer clearly and concisely. Cite reliable sources if you know them."},
                {"role":"user","content": user_input}
            ], max_new_tokens=512, temperature=0.3)
            return direct, steps
    except Exception:
        pass

    # 1) Decide route once per turn
    try:
        route = classify_route(user_input, model_id, prefer_web=prefer_web)
    except Exception:
        route = "rag"

    controller_msgs = [{"role":"system","content": CONTROLLER_SYSTEM}]
    controller_msgs.append({"role":"user","content": f"Route: {route}\nUser question: {user_input}"})

    for step in range(1, max_steps+1):
        with trace("controller_step", run_type="chain", inputs={"step": step, "route": route}):
            raw = hf_chat(model_id, controller_msgs, max_new_tokens=256, temperature=0.2, top_p=0.9)
            steps.append({"role":"assistant", "content": raw})
            parsed = _extract_json_block(raw)

            if not parsed:
                final = hf_chat(model_id, [
                    {"role":"system","content":"Answer the user's question clearly and concisely, without tools."},
                    {"role":"user","content": user_input}
                ], max_new_tokens=512, temperature=0.3)
                return final, steps

            if "final_answer" in parsed:
                ans = parsed.get("final_answer","").strip()
                cites = parsed.get("citations",[])
                if not cites and citations_pool:
                    seen=set(); dedup=[]
                    for c in citations_pool:
                        if c.get("url") and c["url"] not in seen:
                            dedup.append(c); seen.add(c["url"])
                        if len(dedup) >= 5: break
                    cites = dedup
                if cites:
                    ans += "\n\n**Sources:**\n" + "\n".join([f"- [{c.get('title','source')}]({c['url']})" for c in cites if c.get("url")])
                return ans, steps

            tool = parsed.get("tool")
            tool_inp = parsed.get("input","")
            if tool not in TOOLS:
                empty_hits += 1
                controller_msgs.append({"role":"assistant","content": raw})
                controller_msgs.append({"role":"user","content": json.dumps({"note": f"Invalid tool '{tool}'. Consider finalizing."})})
                if empty_hits >= 2:
                    break
                continue

            # Execute tool
            try:
                with trace(f"tool:{tool}", run_type="tool", inputs={"input": tool_inp}) as tspan:
                    result = TOOLS[tool](tool_inp)
                    tspan.outputs = {"result": result}
            except Exception as e:
                result = {"error": f"{tool} failed: {e}"}

            obs_str = json.dumps(result, ensure_ascii=False)
            steps.append({"role":"tool", "tool": tool, "content": _summarize_text_for_obs(obs_str)})

            # pool citations
            try:
                if tool == "rag_retrieve":
                    for r in result.get("results", []):
                        citations_pool.append({"title": r.get("title","source"), "url": r.get("url","")})
                elif tool == "web_search":
                    for r in result.get("results", []):
                        citations_pool.append({"title": r.get("title","source"), "url": r.get("url","")})
                elif tool == "read_url":
                    citations_pool.append({"title": result.get("title","source"), "url": result.get("url","")})
            except Exception:
                pass

            # track empties
            if (isinstance(result, dict) and not result.get("results")) and ("note" in result or "error" in result or "text" not in result):
                empty_hits += 1
            else:
                empty_hits = 0

            # feedback to controller
            controller_msgs.append({"role":"assistant","content": raw})
            controller_msgs.append({"role":"user","content": f"Observation:\n{_summarize_text_for_obs(obs_str)}\n\nReturn the next JSON. If sufficient, finalize."})

            # Early stop: enough sources gathered
            if len({c.get('url') for c in citations_pool if c.get('url')}) >= 2:
                # Encourage finalize
                controller_msgs.append({"role":"user","content":"You have enough evidence (2+ sources). Finalize now."})

            if empty_hits >= 2:
                break

    # Force finalize if loop exits without explicit final
    final = hf_chat(model_id, [
        {"role":"system","content":"Answer the user's question clearly and concisely. If sources exist, cite them."},
        {"role":"user","content": user_input}
    ], max_new_tokens=512, temperature=0.3)
    if citations_pool:
        seen=set(); dedup=[]
        for c in citations_pool:
            if c.get("url") and c["url"] not in seen:
                dedup.append(c); seen.add(c["url"])
            if len(dedup) >= 5: break
        final += "\n\n**Sources:**\n" + "\n".join([f"- [{c.get('title','source')}]({c['url']})" for c in dedup])
    return final, steps

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIOS.keys(), index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    max_steps = st.slider("Max tool calls per turn", 1, 10, 6)
    prefer_web = st.checkbox("Prefer Web for ambiguous queries", value=False)
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
        st.caption("Uses curated sources; top-k=7 with reranking.")
st.caption(f"Model: **{model_id}**  •  Scenario: **{scenario_name}**")

# =========================
# Chat State & UI
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
st.subheader("💬 Tutor Chat (Agentic JSON + Router)")

for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

user_q = st.chat_input("Ask a question. The agent may use RAG, WebSearch, or ReadURL…")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    with tracing_context(project_name=LS_PROJECT, metadata={"type":"agent_turn","scenario":scenario_name,"model":model_id}):
        with trace("agent_turn", run_type="chain", inputs={"question": user_q, "prefer_web": prefer_web}):
            with st.chat_message("assistant"):
                with st.spinner("Thinking with tools…"):
                    try:
                        answer, step_trace = run_agent(user_q, model_id, max_steps=max_steps, prefer_web=prefer_web)
                    except Exception as e:
                        answer, step_trace = f"⚠️ Agent failed: {e}", []

                st.markdown(answer)
                if step_trace:
                    with st.expander("🔍 Agent Trace (JSON + Observations)"):
                        for i, step in enumerate(step_trace, start=1):
                            if step.get("role") == "assistant":
                                st.markdown(f"**Step {i}: Controller JSON**")
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
