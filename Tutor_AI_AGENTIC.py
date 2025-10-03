# GenAI-Tutor — Robust Agentic System with Smart Tool Selection + RAG + Web + LangSmith (Hardened)

import os, io, re, json, hashlib, time, requests
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient

from langsmith import Client
from langsmith.run_helpers import tracing_context, trace

# LangChain DuckDuckGo web search (free)
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# =========================
# Streamlit & runtime hygiene
# =========================
# Reduce noisy inotify/health-check issues in hosted Linux environments
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
# st.set_option("server.fileWatcherType", "none")

st.set_page_config(page_title="GenAI-Tutor (Robust Agentic)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Robust Agentic System</h1>", unsafe_allow_html=True)

# =========================
# Secrets & API Keys
# =========================
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
MIN_RERANK_SCORE = 0.05  # permissive to avoid empty returns

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
    try:
        import faiss
        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        return index
    except Exception:
        class NpIndex:
            def __init__(self, V): self.V = V
            def search(self, qv, k):
                sims = (qv @ self.V.T)
                idxs = np.argsort(-sims, axis=1)[:, :k]
                scores = np.take_along_axis(sims, idxs, axis=1)
                return scores, idxs
        return NpIndex(vectors)

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
    index = build_faiss(vectors)
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
        for c, rs in zip(candidates, rers): c["score_rerank"] = float(rs)
    except Exception:
        for c in candidates: c["score_rerank"] = float(c["score_ann"])

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
# Web search via LangChain DuckDuckGo (fresh per call)
# =========================
def lc_duckduckgo_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    results = []
    def _query(q) -> List[Dict[str, str]]:
        wrapper = DuckDuckGoSearchAPIWrapper()  # create per call
        hits = wrapper.results(q, max_results=max_results)
        out = []
        for h in hits or []:
            # LangChain format: {'title':..., 'link':..., 'snippet':...}
            url = h.get("link") or h.get("url") or ""
            title = h.get("title") or ""
            snippet = h.get("snippet") or ""
            if title and url and url.startswith("http"):
                out.append({"title": title, "url": url, "snippet": snippet})
        return out

    try:
        results = _query(query)
    except Exception:
        results = []
    if not results and len(query.split()) >= 3:
        # A very light broadening retry (no brand/site hardcoding)
        q2 = re.sub(r"\b(202[0-9])\b", "", query).strip()
        q2 = re.sub(r"\s{2,}", " ", q2)
        try:
            results = _query(q2) if q2 and q2 != query else []
        except Exception:
            results = []
    return results

# deterministic top URLs for auto-follow
def pick_top_urls_from_hits(hits: List[Dict[str, str]], n: int = 2) -> List[str]:
    urls = []
    for h in hits:
        u = (h.get("url") or "").strip()
        if u.startswith("http") and u not in urls:
            urls.append(u)
        if len(urls) >= n:
            break
    return urls

# =========================
# JSON helpers
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
        data = _extract_json_block(inp) if inp.strip().startswith("{") else {}
    else:
        data = {}
    query = data.get("query") or (inp if isinstance(inp, str) else "").strip()
    max_results = int(data.get("max_results", 8)) if str(data.get("max_results","")).isdigit() else 8
    if not query or len(query) < 3:
        return {"error": "Empty or too short search query"}
    query = str(query).strip('"').strip("'")

    hits = lc_duckduckgo_search(query, max_results=max_results)
    if not hits:
        return {
            "error": f"No web results found for: {query}",
            "suggestion": "Try simpler keywords or a related term, or provide a direct URL."
        }
    return {"results": hits, "summary": f"Found {len(hits)} web results for '{query}'"}

def tool_read_url(inp: Any) -> Dict[str, Any]:
    if isinstance(inp, dict):
        data = inp
    elif isinstance(inp, str):
        data = _extract_json_block(inp) if inp.strip().startswith("{") else {}
    else:
        data = {}

    url = data.get("url") or (inp if isinstance(inp, str) else "").strip()
    max_chars = int(data.get("max_chars", 8000)) if str(data.get("max_chars","")).isdigit() else 8000
    url = str(url).strip('"').strip("'")
    if not url.startswith("http"):
        return {"error": "Invalid URL format (must start with http:// or https://)"}

    headers_primary = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    def _fetch(u: str) -> Tuple[Optional[str], Optional[int], Optional[bytes], Optional[str]]:
        try:
            r = requests.get(u, headers=headers_primary, timeout=25, allow_redirects=True)
            ctype = (r.headers.get("Content-Type","")).lower()
            status = r.status_code
            if status >= 400:
                return None, status, None, None
            return ctype, status, r.content, r.url
        except requests.RequestException:
            return None, None, None, None

    # Primary tries (once + retry)
    for attempt in (1, 2):
        ctype, status, content, final_url = _fetch(url)
        if content is not None:
            try:
                if "pdf" in (ctype or "") or (final_url or url).lower().endswith(".pdf"):
                    text = _clean_pdf(content)
                else:
                    text = _clean_html(content)
                if not text or len(text) < 50:
                    return {"error": f"No meaningful content extracted from {final_url or url}"}
                text_trimmed = text[:max_chars]
                return {
                    "title": (final_url or url).split("/")[-1][:100],
                    "url": (final_url or url),
                    "text": text_trimmed,
                    "char_count": len(text_trimmed),
                    "summary": f"Extracted {len(text_trimmed)} chars from {final_url or url}"
                }
            except Exception as e:
                if attempt == 2:
                    return {"error": f"Failed to process content from {final_url or url}: {str(e)[:120]}"}
        else:
            if attempt == 2:
                # Generic reader snapshot fallback (domain-agnostic)
                proxy = f"https://r.jina.ai/{url}"
                p_ctype, p_status, p_content, p_final = _fetch(proxy)
                if p_content is not None:
                    try:
                        text = p_content.decode("utf-8", errors="ignore")
                        if not text or len(text) < 50:
                            return {"error": f"No meaningful content extracted via reader for {url}"}
                        text_trimmed = text[:max_chars]
                        return {
                            "title": (p_final or proxy).split("/")[-1][:100],
                            "url": url,  # keep original URL canonical
                            "text": text_trimmed,
                            "char_count": len(text_trimmed),
                            "summary": f"Extracted {len(text_trimmed)} chars via reader for {url}"
                        }
                    except Exception as e:
                        dom = re.sub(r'^https?://', '', url).split('/')[0]
                        return {"error": f"Reader fallback failed for {url}: {str(e)[:120]}", "domain": dom, "http_status": p_status}
                dom = re.sub(r'^https?://', '', url).split('/')[0]
                return {"error": f"Failed to fetch {url}: HTTP {status or 'unknown'}", "http_status": status, "domain": dom}

    dom = re.sub(r'^https?://', '', url).split('/')[0]
    return {"error": f"Failed to fetch {url}: unknown error", "domain": dom}

def tool_rag_retrieve(inp: Any) -> Dict[str, Any]:
    try:
        ensure_rag_ready()
    except Exception as e:
        return {"error": f"RAG index unavailable: {e}", "suggestion": "Use web_search instead"}
    if isinstance(inp, dict):
        data = inp
    elif isinstance(inp, str):
        data = _extract_json_block(inp) if inp.strip().startswith("{") else {}
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
2. web_search: Search the internet (best for: recent news, product updates, latest research, real-time information)
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
# HF chat wrapper (robust)
# =========================
def hf_chat(model: str, messages: List[Dict[str,str]], max_new_tokens=512, temperature=0.0, top_p=0.9) -> str:
    client = InferenceClient(model=model, token=HF_TOKEN)
    # one retry path with reduced tokens on error
    for attempt in (1, 2):
        try:
            resp = client.chat_completion(
                messages=messages,
                max_tokens=max_new_tokens if attempt == 1 else max(256, max_new_tokens // 2),
                temperature=temperature,
                top_p=top_p,
                stop=["```", "\n\nObservation:", "\n\nUser:"],
            )
            choice = resp.choices[0]
            msg = getattr(choice, "message", None) or choice["message"]
            content = getattr(msg, "content", None) or msg["content"]
            return (content or "").strip()
        except Exception:
            if attempt == 2:
                # fallback minimal notice
                return '{"thought":"model_call_failed","stop":true,"reason":"llm_error"}'
            time.sleep(0.5)
    return '{"stop": true, "reason":"unreachable"}'

# =========================
# Controller System (unchanged format)
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
- Stop when you have enough evidence OR all strategies exhausted
- Never call the same tool with the same query twice
- Return ONLY valid JSON, no markdown, no explanation outside JSON

FIRST ACTION RULE:
- On your first turn, you MUST emit exactly one of:
  {{"tool":"rag_retrieve","input":{{"query":"<user question>","top_k":7}}}}
  {{"tool":"web_search","input":{{"query":"<user question>","max_results":8}}}}
No thoughts or stops before the first tool call.
"""

# =========================
# Evidence Store with blocked-domain memory
# =========================
class EvidenceStore:
    def __init__(self):
        self.search_hits: List[Dict[str,str]] = []
        self.pages: Dict[str, Dict[str,str]] = {}
        self.rag_chunks: List[Dict[str,Any]] = []
        self.seen_urls = set()
        self.seen_hashes = set()
        self.tool_history: List[Dict[str, Any]] = []
        self.blocked_domains: Dict[str, int] = {}   # NEW
        self.failed_urls: set[str] = set()          # NEW

    def note_block(self, domain: str):
        if not domain: return
        self.blocked_domains[domain] = self.blocked_domains.get(domain, 0) + 1

    def is_blocked(self, url: str) -> bool:
        dom = re.sub(r'^https?://', '', url).split('/')[0]
        return self.blocked_domains.get(dom, 0) >= 1 or (url in self.failed_urls)

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
        return f"Evidence: {len(self.rag_chunks)} RAG chunks, {len(self.pages)} full pages, {len(self.search_hits)} web snippets"

    def context_pack(self, max_chars: int = 18000) -> Tuple[str, List[Dict[str,str]]]:
        citations: List[Dict[str,str]] = []
        lines = []
        url_to_num = {}
        def cite_for(url, title):
            if url not in url_to_num:
                url_to_num[url] = len(url_to_num) + 1
                citations.append({"title": title or "source", "url": url})
            return url_to_num[url]

        if self.rag_chunks:
            lines.append("### RAG Evidence (Curated Corpus)")
            sorted_chunks = sorted(self.rag_chunks, key=lambda x: x.get("score", 0), reverse=True)
            for it in sorted_chunks[:8]:
                r = cite_for(it.get("url",""), it.get("title","source"))
                snippet = it.get("chunk","")[:750]
                score = it.get("score", 0)
                lines.append(f"[{r}] {it.get('title','source')} (relevance: {score:.2f})\n{snippet}\n")

        if self.pages:
            lines.append("### Web Pages (Full Content)")
            for u, p in list(self.pages.items())[:5]:
                r = cite_for(u, p.get("title",u))
                snippet = (p.get("text",""))[:1200]
                lines.append(f"[{r}] {p.get('title',u)}\n{snippet}\n")

        if self.search_hits:
            lines.append("### Web Search Snippets")
            kept = 0
            for hit in self.search_hits:
                if kept >= 5: break
                u = hit.get("url","")
                if u in self.pages: continue
                r = cite_for(u, hit.get("title",""))
                snippet = hit.get("snippet","")[:400]
                lines.append(f"[{r}] {hit.get('title','')}\n{snippet}\n")
                kept += 1

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[Evidence truncated due to length]"
        return text, citations

# =========================
# Sliding Context Window
# =========================
class SlidingContext:
    def __init__(self, max_interactions: int = 5):
        self.system_msg = None
        self.user_query = None
        self.interactions: List[Tuple[str, str]] = []
        self.max_interactions = max_interactions

    def set_initial(self, system: str, user_query: str):
        self.system_msg = system
        self.user_query = user_query

    def add_interaction(self, planner_resp: str, observation: str):
        self.interactions.append((planner_resp, observation))
        if len(self.interactions) > self.max_interactions:
            self.interactions.pop(0)

    def build_messages(self, reflection: Optional[str] = None) -> List[Dict[str, str]]:
        msgs = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": f"User question: {self.user_query}"}
        ]
        for planner_resp, obs in self.interactions:
            msgs.append({"role": "assistant", "content": planner_resp})
            msgs.append({"role": "user", "content": f"Observation: {obs}"})
        if reflection:
            msgs.append({"role": "user", "content": reflection})
        return msgs

# =========================
# Agent loop (reflection interval fixed to 3 in backend)
# =========================
def run_agent(
    user_input: str,
    model_id: str,
    scenario_sys: str,
    max_steps: int = 10,
    reflection_interval: int = 3  # fixed
) -> Tuple[str, List[Dict[str, Any]]]:
    steps = []
    ev = EvidenceStore()
    ctx = SlidingContext(max_interactions=5)
    ctx.set_initial(controller_system_text(scenario_sys), user_input)

    invalid_call_count = 0
    no_evidence_count = 0
    max_invalid = 2
    max_no_evidence = 4

    for step in range(1, max_steps + 1):
        reflection = None
        if step > 1 and step % reflection_interval == 0:
            recent_tools = [h["tool"] for h in ev.tool_history[-3:]]
            reflection = f"""
🔍 REFLECTION CHECKPOINT (Step {step}):
Evidence gathered: {ev.get_summary()}
Recent tools used: {', '.join(recent_tools)}

Critical questions:
1. Do you have sufficient HIGH-QUALITY evidence to answer the user's question?
2. If RAG returned good results (score >= 0.5), why are you still searching?
3. If web_search keeps failing, have you tried simpler/broader keywords?
4. Are you repeating the same unsuccessful strategy?

Decision: Return {{"thought": "..."}} with your assessment, then either:
- {{"stop": true, "reason": "sufficient evidence"}} if you can answer well
- {{"tool": "...", "input": {{...}}}} with a NEW strategy if gaps remain
"""

        msgs = ctx.build_messages(reflection)

        with trace("planner_step", run_type="chain", inputs={"step": step, "evidence_count": len(ev.rag_chunks) + len(ev.pages)}):
            try:
                planner_out = hf_chat(model_id, msgs, max_new_tokens=220, temperature=0.0, top_p=0.9)
            except Exception as e:
                steps.append({"role": "error", "content": f"LLM call failed: {e}"})
                break

            steps.append({"role": "assistant", "content": planner_out, "step": step})
            parsed = _extract_json_block(planner_out)
            if not parsed or not any(k in parsed for k in ("tool","stop","thought")):
                parsed = _force_last_json(planner_out)

            # If still nothing, run a safe default tool once
            if not parsed or not any(k in parsed for k in ("tool","stop","thought")):
                heuristic_tool = "rag_retrieve"
                tool_inp = {"query": user_input, "top_k": TOP_K}
                try:
                    with trace(f"tool:{heuristic_tool}", run_type="tool", inputs={"input": tool_inp}) as tspan:
                        result = TOOLS[heuristic_tool](tool_inp)
                        tspan.outputs = {"result": result}
                    added = 0
                    obs = ""
                    if "error" in result:
                        no_evidence_count += 1
                        obs = f"❌ {heuristic_tool} failed: {result['error']}"
                    else:
                        added = ev.add_rag(result.get("results", []))
                        obs = result.get("summary", f"Added {added} RAG chunks")
                        if added == 0:
                            no_evidence_count += 1
                        else:
                            no_evidence_count = 0
                    ev.add_tool_call(heuristic_tool, json.dumps(tool_inp), result, added)
                    steps.append({"role": "tool", "tool": heuristic_tool, "content": _summarize_text(obs, 900), "added": added, "step": step})
                    ctx.add_interaction(planner_out, f"Planner returned non-JSON. Ran default rag_retrieve once. {obs}")
                    continue
                except Exception as e:
                    invalid_call_count += 1
                    ctx.add_interaction(planner_out, f"Planner parse failure and default call failed: {e}")
                    if invalid_call_count >= max_invalid:
                        steps.append({"role":"stop","content":"Planner outputs invalid JSON repeatedly. Stopping.", "step": step})
                        break
                    continue

            # Handle thought
            if parsed.get("thought"):
                thought = parsed.get("thought", "")
                steps.append({"role": "thought", "content": thought, "step": step})
                ctx.add_interaction(planner_out, f"Thought noted: {thought[:100]}")
                continue

            # Handle stop
            if parsed.get("stop") is True:
                reason = parsed.get("reason", "planner decided to stop")
                steps.append({"role": "stop", "content": reason, "step": step})
                break

            # Handle tool call
            tool = parsed.get("tool")
            tool_inp = parsed.get("input", "")

            if not tool or tool not in TOOLS:
                invalid_call_count += 1
                error_msg = f"Invalid tool call. Available: {', '.join(TOOLS.keys())}. You said: {tool}"
                steps.append({"role": "error", "content": error_msg, "step": step})
                ctx.add_interaction(planner_out, error_msg)
                if invalid_call_count >= max_invalid:
                    steps.append({"role": "stop", "content": "Too many invalid tool calls, stopping", "step": step})
                    break
                continue

            # Execute tool
            try:
                with trace(f"tool:{tool}", run_type="tool", inputs={"input": tool_inp}) as tspan:
                    result = TOOLS[tool](tool_inp)
                    tspan.outputs = {"result": result}
            except Exception as e:
                result = {"error": f"{tool} execution failed: {str(e)[:200]}"}

            added = 0
            quality_note = ""
            should_consider_stopping = False

            if "error" in result:
                error_msg = result["error"]
                suggestion = result.get("suggestion", "")
                obs = f"❌ {tool} failed: {error_msg}\n" + (f"💡 {suggestion}" if suggestion else "")
                # If read_url error includes domain, remember to avoid re-hitting in this turn
                dom = result.get("domain")
                if dom:
                    ev.note_block(dom)
                url_failed = None
                if isinstance(tool_inp, dict):
                    url_failed = tool_inp.get("url")
                if url_failed:
                    ev.failed_urls.add(url_failed)
                no_evidence_count += 1

            else:
                if tool == "web_search" and "results" in result:
                    added = ev.add_search(result["results"])
                    quality_note = result.get("summary", f"Added {added} web results")

                    # Show preview to the planner
                    preview = "\n".join(f"- {h['title'][:80]} :: {h['url']}" for h in result["results"][:5])
                    ctx.add_interaction('{"tool":"web_search","input":{}}', f"Top search URLs:\n{preview}")

                    # Deterministic auto-follow top 1–2 URLs, skipping blocked domains
                    top_urls = pick_top_urls_from_hits(result["results"], n=2)
                    for u in top_urls:
                        if ev.is_blocked(u):
                            steps.append({"role":"tool","tool":"read_url","content":f"Skip auto-follow for blocked domain URL: {u}", "added":0, "step":step})
                            continue
                        try:
                            with trace("tool:read_url(auto_follow)", run_type="tool", inputs={"input": {"url": u, "max_chars": 8000}}) as tspan2:
                                r2 = TOOLS["read_url"]({"url": u, "max_chars": 8000})
                                tspan2.outputs = {"result": r2}
                            if "error" in r2:
                                dom2 = r2.get("domain")
                                if dom2:
                                    ev.note_block(dom2)
                                ev.failed_urls.add(u)
                                steps.append({"role":"tool","tool":"read_url","content":_summarize_text(f"Auto-follow failed: {r2['error']}",900),"added":0,"step":step})
                            else:
                                added_page = ev.add_page(r2)
                                steps.append({"role":"tool","tool":"read_url","content":_summarize_text(r2.get("summary","Fetched page"),900),"added":added_page,"step":step})
                        except Exception as e:
                            steps.append({"role":"tool","tool":"read_url","content":f"Auto-follow exception: {str(e)[:120]}", "added":0, "step":step})

                elif tool == "read_url" and "text" in result:
                    added = ev.add_page(result)
                    quality_note = result.get("summary", f"Added {added} page")

                elif tool == "rag_retrieve" and "results" in result:
                    added = ev.add_rag(result["results"])
                    avg_score = result.get("avg_score", 0.0)
                    quality_note = result.get("summary", f"Added {added} RAG chunks (avg quality: {avg_score:.2f})")
                    if avg_score >= 0.5 and added >= 5:
                        quality_note += "\n✅ HIGH quality results! This is strong evidence. Consider if you can stop."
                        should_consider_stopping = True
                    elif avg_score >= 0.3:
                        quality_note += "\n✓ MEDIUM quality - acceptable for answering."
                    elif 0 < avg_score < 0.3:
                        quality_note += "\n⚠️ LOW quality - try web_search for better results."

                obs = quality_note if quality_note else json.dumps(result, ensure_ascii=False)[:800]
                if added == 0:
                    no_evidence_count += 1
                else:
                    no_evidence_count = 0

            ev.add_tool_call(tool, tool_inp if isinstance(tool_inp, str) else json.dumps(tool_inp), result, added)
            steps.append({"role": "tool", "tool": tool, "content": _summarize_text(obs, 900), "added": added, "step": step})
            ctx.add_interaction(planner_out, obs)

            # stopping checks
            if no_evidence_count >= max_no_evidence:
                steps.append({"role": "stop", "content": "Multiple tools returned no evidence. Stopping to synthesize what we have.", "step": step})
                break
            total_evidence = len(ev.rag_chunks) + len(ev.pages) + len(ev.search_hits)
            if should_consider_stopping or (total_evidence >= 7 and step >= 3):
                obs += f"\n💡 You now have {total_evidence} pieces of evidence. If this covers the question well, consider stopping."

    # Final synthesis
    context_text, cites = ev.context_pack(max_chars=18000)
    total_evidence = len(ev.rag_chunks) + len(ev.pages) + len(ev.search_hits)

    if not context_text or len(context_text) < 100:
        final_answer = (
            "I couldn't gather enough high-quality evidence to answer confidently.\n\n"
            "**Possible reasons**\n"
            "- The curated corpus lacks this topic\n"
            "- Web search returned weak/blocked results\n"
            "- The question might need rephrasing\n\n"
            "**Try**\n"
            "- Rephrase with simpler keywords\n"
            "- Ask a broader or adjacent question\n"
            "- Provide a specific URL to read"
        )
    else:
        cite_lines = "\n".join([f"[{i+1}] [{c['title']}]({c['url']})" for i, c in enumerate(cites)])
        final_sys = (
            f"{scenario_sys}\n\n"
            "You are now providing the final answer based on gathered evidence.\n"
            "CRITICAL RULES:\n"
            "- Use ONLY information from the EVIDENCE section below\n"
            "- Cite claims with [n] markers matching the source numbers\n"
            "- Be concise but comprehensive\n"
            "- If evidence conflicts, acknowledge it\n"
            "- Note gaps explicitly if evidence is missing\n"
            "- Provide actionable insights when possible\n"
            "- Do NOT invent numbers (e.g., pricing) that are not explicitly present in the EVIDENCE.\n"
        )
        final_user = (
            f"USER QUESTION:\n{user_input}\n\n"
            f"EVIDENCE (from {len(cites)} sources):\n{context_text}\n\n"
            "Write a clear, structured answer with inline [n] citations."
        )
        try:
            with trace("final_synthesis", run_type="llm", inputs={"context_length": len(context_text), "num_sources": len(cites)}):
                final_answer = hf_chat(
                    model_id,
                    [{"role": "system", "content": final_sys},
                     {"role": "user", "content": final_user}],
                    max_new_tokens=800,
                    temperature=0.3
                )
        except Exception as e:
            final_answer = f"Error generating final answer: {e}\n\nEvidence was gathered but synthesis failed."
        if cite_lines:
            final_answer += "\n\n---\n**Sources:**\n" + cite_lines
    return final_answer, steps

# =========================
# Sidebar Configuration (reflection removed from UI)
# =========================
with st.sidebar:
    st.header("⚙️ Configuration")

    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model", HF_MODELS, index=0)

    st.markdown("---")
    st.subheader("🤖 Agent Settings")
    max_steps = st.slider("Max planning steps", 3, 15, 10, help="Maximum tool calls before forcing stop")

    st.markdown("---")
    st.subheader("📚 RAG Corpus")

    if st.button("🔄 Rebuild RAG Index", use_container_width=True):
        with st.spinner("Rebuilding RAG index..."):
            try:
                idx, side = build_rag_index(DOC_LINKS)
                _global_rag["index"], _global_rag["side"] = idx, side
                st.success(f"✅ RAG index rebuilt: {side['vectors_shape'][0]} chunks from {len(DOC_LINKS)} sources")
            except Exception as e:
                st.error(f"❌ RAG build failed: {e}")

    if _global_rag["index"] is not None and _global_rag["side"] is not None:
        st.info(f"📊 Index: {_global_rag['side']['vectors_shape'][0]} chunks ready")

    st.markdown("---")
    st.caption(f"**Model:** {model_id.split('/')[-1]}")
    st.caption(f"**Scenario:** {scenario_name}")

# =========================
# Main Chat Interface
# =========================
st.markdown("---")
st.subheader("💬 Intelligent Tutor Chat")
st.caption("The agent decides which tools to use based on your question. It starts with RAG for concepts, falls back to web search if needed, and can fetch full articles.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "👋 Hi! I'm GenAI-Tutor with an intelligent agentic system.\n\nI can:\n- 📚 Search curated Gen-AI educational content (RAG)\n- 🌐 Search the web for recent information\n- 📄 Fetch and read full articles\n\nI automatically choose the best approach based on your question. Ask me anything!"}
    ]
    st.session_state.scenario_prev = scenario_name

if "scenario_prev" not in st.session_state:
    st.session_state.scenario_prev = scenario_name

# Reset chat if scenario changed
if st.session_state.scenario_prev != scenario_name:
    st.session_state.messages = [
        {"role": "assistant", "content": f"📚 Switched to **{scenario_name}** scenario.\n\n{SCENARIOS[scenario_name]['overview']}\n\nWhat would you like to learn?"}
    ]
    st.session_state.scenario_prev = scenario_name

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask about Gen-AI, prompt engineering, ethics, or current AI news...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Planning strategy and gathering evidence..."):
            try:
                with tracing_context(
                    project_name=LS_PROJECT,
                    metadata={"type": "agent_conversation","scenario": scenario_name,"model": model_id,"max_steps": max_steps}
                ):
                    with trace("agent_turn", run_type="chain", inputs={"question": user_query, "scenario": scenario_name}):
                        answer, step_trace = run_agent(
                            user_input=user_query,
                            model_id=model_id,
                            scenario_sys=SCENARIOS[scenario_name]["system"],
                            max_steps=max_steps,
                            reflection_interval=3  # fixed in backend
                        )
            except Exception as e:
                answer = f"⚠️ **Agent Error**\n\nThe agentic system encountered an error: {str(e)}\n\nPlease try rephrasing your question."
                step_trace = []

        st.markdown(answer)

        if step_trace:
            with st.expander("🔍 Agent Reasoning Trace (Click to expand)", expanded=False):
                st.caption("See how the agent decided which tools to use and what evidence it gathered.")
                for i, step in enumerate(step_trace, start=1):
                    step_num = step.get("step", i)
                    if step.get("role") == "assistant":
                        st.markdown(f"**Step {step_num}: 🧠 Planner Decision**")
                        st.code(step["content"], language="json")
                    elif step.get("role") == "thought":
                        st.markdown(f"**Step {step_num}: 💭 Reflection**")
                        st.info(step["content"])
                    elif step.get("role") == "tool":
                        tool_name = step.get("tool", "unknown")
                        added = step.get("added", 0)
                        emoji = "📚" if tool_name == "rag_retrieve" else "🌐" if tool_name == "web_search" else "📄"
                        st.markdown(f"**Step {step_num}: {emoji} Tool: `{tool_name}` → Added {added} items**")
                        st.text_area("Result", value=step["content"], height=150,
                                     key=f"tool_{step_num}_{tool_name}_{i}", disabled=True)
                    elif step.get("role") == "error":
                        st.markdown(f"**Step {step_num}: ⚠️ Error**")
                        st.error(step["content"])
                    elif step.get("role") == "stop":
                        st.markdown(f"**Step {step_num}: 🛑 Planning Complete**")
                        st.success(step["content"])

                st.divider()
                st.caption(f"Total steps: {len(step_trace)} | Agent used intelligent tool selection based on context")

    st.session_state.messages.append({"role": "assistant", "content": answer})

# =========================
# Information Panel
# =========================
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("📌 Current Scenario")
    st.markdown(f"**{scenario_name}**")
    st.caption(SCENARIOS[scenario_name]['overview'])
with col2:
    st.subheader("🤖 Agent Capabilities")
    st.markdown("""
    **Smart Tool Selection:**
    - 📚 Starts with RAG for educational concepts
    - 🌐 Falls back to web search if RAG quality is low
    - 📄 Fetches full articles when needed
    - 💭 Reflects on progress every few steps (interval fixed to 3)
    - 🎯 Stops when sufficient evidence gathered
    """)

# =========================
# Example Questions
# =========================
st.markdown("---")
st.subheader("💡 Try These Questions")
example_cols = st.columns(3)
with example_cols[0]:
    st.markdown("**📚 Conceptual (uses RAG)**")
    if st.button("What is prompt engineering?", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What is prompt engineering and what are the key techniques?"})
        st.rerun()
with example_cols[1]:
    st.markdown("**🌐 Current Info (uses Web)**")
    if st.button("Latest multimodal model updates?", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What are the latest multimodal model updates?"})
        st.rerun()
with example_cols[2]:
    st.markdown("**🔀 Hybrid (uses both)**")
    if st.button("AI in education ethics", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "What are the ethical considerations of using AI in education?"})
        st.rerun()

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("⚠️ **Educational Use Only** • Verify critical information • Follow your organization's AI policies")
st.caption(f"🔧 Powered by: {model_id} | LangSmith tracing enabled | RAG + Web Search + Agentic Planning")
