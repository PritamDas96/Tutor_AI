# GenAI-Tutor — Robust Agentic System with Smart Tool Selection + RAG + Web + LangSmith (LangChain-based Web Search, no hardcoding)

import os, io, re, json, hashlib, requests, time
from typing import List, Dict, Any, Tuple, Optional

# --- Disable Streamlit file watcher early (avoids inotify limit errors) ---
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient, InferenceTimeoutError, BadRequestError

from langsmith import Client
from langsmith.run_helpers import tracing_context, trace

# LangChain DuckDuckGo (free) + optional Tool interface
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool

# =========================
# Streamlit & Secrets
# =========================
st.set_page_config(page_title="GenAI-Tutor (Robust Agentic)", layout="wide")
try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass

st.markdown("<h1>🎓 GenAI-Tutor — Robust Agentic System</h1>", unsafe_allow_html=True)

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
MIN_RERANK_SCORE = 0.05  # permissive

@st.cache_data(show_spinner=False)
def _download(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 TutorAI/2.0"})
    r.raise_for_status()
    return r.content, (r.headers.get("Content-Type", "")).lower()

def _clean_html(html_bytes: bytes) -> str:
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        for t in soup(["script","style","noscript","header","footer","nav","form"]): 
            t.decompose()
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
    if len(words) < 100:
        return []
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
def load_embedder(): 
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=True)
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def embed_texts(texts: List[str], model):
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")

@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    import faiss
    d = vectors.shape[1]; index = faiss.IndexFlatIP(d); index.add(vectors); 
    return index

@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, Any]]):
    embedder = load_embedder()
    all_chunks = []
    for doc in doc_links:
        if not doc.get("enabled", True): 
            continue
        raw = fetch_and_clean(doc["url"])
        if not raw or len(raw.split()) < 100:
            continue
        all_chunks.extend(chunk_text(raw, doc["url"], doc["title"]))
    if not all_chunks:
        raise RuntimeError("No chunks ingested from sources (all empty/blocked).")
    vectors = embed_texts([c["text"] for c in all_chunks], embedder)
    try:
        index = build_faiss(vectors)
    except Exception:
        class NpIndex:
            def __init__(self, V): self.V = V
            def search(self, qv, k):
                sims = (qv @ self.V.T)
                idxs = np.argsort(-sims, axis=1)[:, :k]
                scores = np.take_along_axis(sims, idxs, axis=1)
                return scores, idxs
        index = NpIndex(vectors)
    side = {"chunks": all_chunks, "vectors_shape": vectors.shape}
    return index, side

def retrieve(query: str, index, side: Dict[str, Any], top_k: int = TOP_K) -> Tuple[List[Dict[str, Any]], float]:
    embedder, reranker = load_embedder(), load_reranker()
    qv = embed_texts([query], embedder)
    scores, idxs = index.search(qv, K_CANDIDATES)
    candidates = []
    for rank, (ci, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        if ci < 0: continue
        c = side["chunks"][ci]
        candidates.append({"rank_ann":rank,"score_ann":float(s),**c})
    if not candidates: return [], 0.0

    try:
        pairs = [(query, c["text"]) for c in candidates]
        rers = reranker.predict(pairs, batch_size=64).tolist()
        for c, rs in zip(candidates, rers): 
            c["score_rerank"] = float(rs)
    except Exception:
        for c in candidates:
            c["score_rerank"] = float(c["score_ann"])

    quality_results = [c for c in candidates if c["score_rerank"] >= MIN_RERANK_SCORE]
    if not quality_results:
        return [], 0.0

    sorted_results = sorted(quality_results, key=lambda x: x["score_rerank"], reverse=True)[:top_k]
    avg_score = sum(c["score_rerank"] for c in sorted_results) / len(sorted_results)
    return sorted_results, avg_score

# Lazy RAG
_global_rag = {"index": None, "side": None}
def ensure_rag_ready():
    if _global_rag["index"] is None or _global_rag["side"] is None:
        idx, side = build_rag_index(DOC_LINKS)
        _global_rag["index"], _global_rag["side"] = idx, side

# =========================
# Web search (LangChain, per-call, retry; no hardcoding)
# =========================
def _ddg_search_once(query: str, k: int) -> List[Dict[str, str]]:
    wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="y", max_results=k)
    items = wrapper.results(query, max_results=k) or []
    results: List[Dict[str, str]] = []
    for r in items:
        title = (r.get("title") or "").strip()
        url = (r.get("link") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        if title and url:
            results.append({"title": title[:160], "url": url, "snippet": snippet[:400]})
    return results

def web_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """
    Free web search via LangChain's DuckDuckGo wrapper.
    - Fresh wrapper per call (no cached/bad state)
    - One retry on exception (transient robustness)
    - No normalization, no domain pinning, no price heuristics
    """
    if not query or len(query.strip()) < 3:
        return []
    try:
        res = _ddg_search_once(query.strip(), max_results)
    except Exception:
        time.sleep(1.2)
        res = _ddg_search_once(query.strip(), max_results)
    return res

# (Optional) Also expose a LangChain Tool that calls the same function (for future LC/LangGraph agents)
def make_web_search_tool_lc(k: int = 8) -> Tool:
    return Tool(
        name="web_search",
        description="Search the public web (DuckDuckGo). Input a concise query.",
        func=lambda q: json.dumps(web_search(q, max_results=k), ensure_ascii=False),
    )

# =========================
# Improved JSON Parsing
# =========================
def _extract_json_block(text: str) -> Dict[str, Any]:
    try:
        matches = list(re.finditer(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', text, re.DOTALL))
    except re.error:
        matches = []
    if not matches:
        return {}
    for match in reversed(matches):
        s = match.group(0)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and (obj.get("tool") or obj.get("stop") or obj.get("thought")):
                return obj
        except json.JSONDecodeError:
            s_fixed = re.sub(r',\s*([}\]])', r'\1', s)
            try:
                obj = json.loads(s_fixed)
                if isinstance(obj, dict) and (obj.get("tool") or obj.get("stop") or obj.get("thought")):
                    return obj
            except:
                continue
    return {}

def _force_last_json(text: str) -> Dict[str, Any]:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.DOTALL)
    blocks = re.findall(r'\{.*\}', t, flags=re.DOTALL)
    for s in reversed(blocks):
        try:
            obj = json.loads(re.sub(r',\s*([}\]])', r'\1', s))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            continue
    return {}

def _summarize_text(s: str, limit: int = 800) -> str:
    return s[:limit] + ("…" if len(s) > limit else "")

# =========================
# Tools with Quality Checks
# =========================
def tool_web_search(inp: Any) -> Dict[str, Any]:
    if isinstance(inp, dict):
        data = inp
    elif isinstance(inp, str):
        if inp.strip().startswith("{"):
            data = _extract_json_block(inp)
        else:
            data = {}
    else:
        data = {}
    query = data.get("query") or (inp if isinstance(inp, str) else "").strip()
    max_results = int(data.get("max_results", 8)) if str(data.get("max_results","")).isdigit() else 8
    if not query or len(query) < 3:
        return {"error": "Empty or too short search query"}
    query = str(query).strip('"').strip("'")

    # Execute robust search
    try:
        res = web_search(query, max_results=max_results)
    except Exception as e:
        return {"error": f"web_search execution error: {e}"}

    if not res:
        return {
            "error": f"0 results for: {query}",
            "suggestion": "Try broader keywords or provide a URL to read."
        }
    return {"results": res, "summary": f"Found {len(res)} web results for '{query}'"}

def tool_read_url(inp: Any) -> Dict[str, Any]:
    if isinstance(inp, dict):
        data = inp
    elif isinstance(inp, str):
        if inp.strip().startswith("{"):
            data = _extract_json_block(inp)
        else:
            data = {}
    else:
        data = {}
    url = data.get("url") or (inp if isinstance(inp, str) else "").strip()
    max_chars = int(data.get("max_chars", 8000)) if str(data.get("max_chars","")).isdigit() else 8000
    url = str(url).strip('"').strip("'")
    if not url.startswith("http"):
        return {"error": "Invalid URL format (must start with http:// or https://)"}
    try:
        r = requests.get(url, headers={"User-Agent":"TutorAI/2.0 (+https://example.com)"}, timeout=25)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type","")).lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = _clean_pdf(r.content)
            if not text or len(text) < 50:
                return {"error": f"PDF text extraction was empty for {url}",
                        "suggestion": "This PDF may be scanned or blocked. Try an HTML version or another source."}
        else:
            text = _clean_html(r.content)
        if not text or len(text) < 50:
            return {"error": f"No meaningful content extracted from {url}"}
        text_trimmed = text[:max_chars]
        return {
            "title": url.split("/")[-1][:100],
            "url": url,
            "text": text_trimmed,
            "char_count": len(text_trimmed),
            "summary": f"Extracted {len(text_trimmed)} chars from {url}"
        }
    except requests.RequestException as e:
        return {"error": f"Failed to fetch {url}: {str(e)[:100]}"}
    except Exception as e:
        return {"error": f"Failed to process {url}: {str(e)[:100]}"}

def tool_rag_retrieve(inp: Any) -> Dict[str, Any]:
    try:
        ensure_rag_ready()
    except Exception as e:
        return {"error": f"RAG index unavailable: {e}", "suggestion": "Use web_search instead"}
    if isinstance(inp, dict):
        data = inp
    elif isinstance(inp, str):
        if inp.strip().startswith("{"):
            data = _extract_json_block(inp)
        else:
            data = {}
    else:
        data = {}
    q = data.get("query") or (inp if isinstance(inp, str) else "").strip()
    q = str(q).strip('"').strip("'")
    if not q or len(q) < 3:
        return {"error": "Empty or too short RAG query"}
    try:
        k = int(data.get("top_k", TOP_K)); k = max(1, min(10, k))
    except:
        k = TOP_K
    results, avg_score = retrieve(q, _global_rag["index"], _global_rag["side"], top_k=k)
    if not results:
        return {"error": f"No relevant content in RAG corpus for: {q}",
                "suggestion": "Topic may be outside corpus scope. Try web_search for broader coverage.",
                "avg_score": 0.0}
    out = [{
        "title": c["title"],
        "url": c["url"],
        "chunk": c["text"][:900],
        "score": round(c.get("score_rerank", 0.0), 3)
    } for c in results]
    if avg_score >= 0.5: quality_note = " (HIGH quality - good match)"
    elif avg_score >= 0.3: quality_note = " (MEDIUM quality - acceptable)"
    else: quality_note = " (LOW quality - consider web_search)"
    return {"results": out, "avg_score": round(avg_score, 3),
            "summary": f"Retrieved {len(out)} chunks (avg score: {avg_score:.2f}){quality_note}"}

TOOLS = {
    "rag_retrieve": tool_rag_retrieve,
    "web_search": tool_web_search,
    "read_url": tool_read_url,
}

TOOL_DESCRIPTIONS = """
Available tools:
1. rag_retrieve: Search curated Gen-AI education corpus (best for: fundamental concepts, pedagogical approaches, ethical considerations, established best practices)
2. web_search: Search the internet (best for: recent news, current pricing, product updates, latest research, real-time information)
3. read_url: Fetch and read content from a specific URL (use after web_search to get full article content)

Tool selection strategy:
- START with rag_retrieve for conceptual/educational questions about Gen-AI fundamentals
- If RAG returns HIGH quality (avg_score >= 0.5), you have good evidence - consider stopping
- If RAG returns MEDIUM quality (0.3-0.5), you have acceptable evidence - may add web_search for supplemental info
- If RAG returns LOW quality (< 0.3) or error, immediately switch to web_search
- Use web_search for time-sensitive queries (news, pricing, product features)
- After web_search, use read_url on the most promising 1-2 URLs for depth
- Avoid redundant calls: don't search the same query twice with the same phrasing
- STOP when you have 5+ high-quality pieces of evidence OR all strategies exhausted
"""

# =========================
# HF chat wrapper (InferenceClient) with gentle retry + safe stop tokens
# =========================
def hf_chat(model: str, messages: List[Dict[str,str]], max_new_tokens=512, temperature=0.0, top_p=0.9) -> str:
    client = InferenceClient(model=model, token=HF_TOKEN)
    try:
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["```", "\n\nObservation:", "\n\nUser:"],
        )
    except (InferenceTimeoutError, BadRequestError):
        # retry once smaller output (helps some endpoints)
        resp = client.chat_completion(
            messages=messages,
            max_tokens=max(256, int(max_new_tokens/2)),
            temperature=temperature,
            top_p=top_p,
            stop=["```", "\n\nObservation:", "\n\nUser:"],
        )
    choice = resp.choices[0]
    msg = getattr(choice, "message", None) or choice["message"]
    content = getattr(msg, "content", None) or msg["content"]
    return (content or "").strip()

# =========================
# Enhanced Controller System
# =========================
def controller_system_text(scenario_sys: str) -> str:
    return f"""{scenario_sys}

You are the Planner for GenAI-Tutor. Your job is to intelligently select tools to gather evidence, then signal when ready.

{TOOL_DESCRIPTIONS}

RESPONSE FORMAT (must be valid JSON only, no extra text):

1. To think/reflect:
{{"thought": "brief reasoning about what to do next"}}

2. To call a tool:
{{"tool": "rag_retrieve|web_search|read_url", "input": {{"query": "...", ...}}}}

3. To stop planning:
{{"stop": true, "reason": "sufficient evidence gathered | no more useful tools"}}

CRITICAL RULES:
- ALWAYS start with rag_retrieve for educational/conceptual questions
- If rag_retrieve returns error or avg_score < 0.3, immediately try web_search
- Use web_search for recent events, pricing, product features
- After web_search, consider read_url on top 1-2 URLs for depth
- Include "thought" every 2-3 steps to reflect on progress
- Stop when you have enough evidence OR all useful strategies exhausted
- Never call the same tool with the same query twice
- Return ONLY valid JSON, no markdown, no explanation outside JSON

FIRST ACTION RULE:
- On your first turn, you MUST emit exactly one of:
  {{"tool":"rag_retrieve","input":{{"query":"<user question>","top_k":7}}}}
  {{"tool":"web_search","input":{{"query":"<user question>","max_results":8}}}}
No thoughts or stops before the first tool call.
"""

# =========================
# Evidence Store with Quality Tracking
# =========================
class EvidenceStore:
    def __init__(self):
        self.search_hits: List[Dict[str,str]] = []
        self.pages: Dict[str, Dict[str,str]] = {}
        self.rag_chunks: List[Dict[str,Any]] = []
        self.seen_urls = set()
        self.seen_hashes = set()
        self.tool_history: List[Dict[str, Any]] = []

    def add_tool_call(self, tool: str, inp: Any, result: Dict[str, Any], added: int):
        self.tool_history.append({
            "tool": tool,
            "input": (inp if isinstance(inp, str) else json.dumps(inp))[:100],
            "success": "error" not in result,
            "added": added,
            "quality": result.get("avg_score", 0.0) if tool == "rag_retrieve" else None
        })

    def add_search(self, items: List[Dict[str,str]]) -> int:
        added = 0
        for it in items:
            u = it.get("url","")
            if not u or u in self.seen_urls: continue
            self.seen_urls.add(u)
            self.search_hits.append(it)
            added += 1
        return added

    def add_page(self, page: Dict[str,str]) -> int:
        u = page.get("url","")
        txt = page.get("text","")
        if not u or not txt or len(txt) < 50: return 0
        h = hashlib.sha1((u + txt[:1000]).encode()).hexdigest()
        if h in self.seen_hashes: return 0
        self.seen_hashes.add(h)
        self.pages[u] = {"title": page.get("title",u), "url": u, "text": txt}
        return 1

    def add_rag(self, items: List[Dict[str,Any]]) -> int:
        added = 0
        for it in items:
            chunk = it.get("chunk","")
            if not chunk or len(chunk) < 30: continue
            u = it.get("url","")
            h = hashlib.sha1((u + chunk[:400]).encode()).hexdigest()
            if h in self.seen_hashes: continue
            self.seen_hashes.add(h)
            self.rag_chunks.append(it)
            added += 1
        return added

    def get_summary(self) -> str:
        return f"Evidence: {
