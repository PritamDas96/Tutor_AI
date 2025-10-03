# ------------------------------------------------------------
# GenAI-Tutor (Simple Agentic)
# - Tools: rag_retrieve → (gate by confidence) → web_search → read_url
# - No domain bias; resilient search; only cite successfully fetched pages.
# - HF-only (no OpenAI). Streamlit watcher disabled to avoid inotify errors.
# ------------------------------------------------------------

import os
# Disable event-based file watcher BEFORE importing streamlit (fixes inotify)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import re, io, json, hashlib, requests
from typing import List, Dict, Any, Tuple
import numpy as np

import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ddgs import DDGS  # DuckDuckGo search (new package name)
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient

# =========================
# App config
# =========================
st.set_page_config(page_title="GenAI-Tutor (Simple Agentic)", layout="wide")
st.markdown("<h2>🎓 GenAI-Tutor — Simple Agentic System (RAG-gated, HF-only)</h2>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN to Streamlit Secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Chat-capable HF models
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Curated public RAG sources (accessible)
DOC_LINKS = [
    {"title": "Frontiers — Ethical & Regulatory Challenges of GenAI in Education (2025)",
     "url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1565938/full", "enabled": True},
    {"title": "Google Blog — Learn Your Way: Reimagining Textbooks with GenAI (2025)",
     "url": "https://blog.google/outreach-initiatives/education/learn-your-way/", "enabled": True},
    {"title": "HEPI — Student Generative AI Survey 2025",
     "url": "https://www.hepi.ac.uk/reports/student-generative-ai-survey-2025/", "enabled": True},
    {"title": "Nature (PDF) — Educational impacts of generative AI (2025)",
     "url": "https://www.nature.com/articles/s41598-025-06930-w.pdf", "enabled": True},
    {"title": "ACL Anthology (PDF) — Enhancing RAG: Best Practices (COLING 2025)",
     "url": "https://aclanthology.org/2025.coling-main.449.pdf", "enabled": True},
]

# RAG settings
TOP_K = 7
K_CANDIDATES = 30
WORDS_PER_CHUNK = 450
OVERLAP_WORDS = 80
RAG_CONF_THRESHOLD = 0.30  # gate: if mean top-3 rerank score below this, prefer web

# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def _download(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 TutorAI/1.0"})
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
    out = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for p in reader.pages:
        try:
            out.append(p.extract_text() or "")
        except Exception:
            out.append("")
    return "\n".join(ln.strip() for ln in "\n".join(out).splitlines() if ln.strip())

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
        end = min(start + WORDS_PER_CHUNK, len(words))
        piece = " ".join(words[start:end])
        chunk_id = f"{hashlib.sha1(url.encode()).hexdigest()}#{k:04d}"
        chunks.append({"chunk_id": chunk_id, "title": title, "url": url, "text": piece})
        if end == len(words): break
        start = max(0, end - OVERLAP_WORDS); k += 1
    return chunks

@st.cache_resource(show_spinner=True)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=True)
def load_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-v2-m3")

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")

@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    import faiss
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # normalized vecs → IP ~ cosine
    index.add(vectors)
    return index

@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, Any]]):
    embedder = load_embedder()
    all_chunks: List[Dict[str, Any]] = []
    for doc in doc_links:
        if not doc.get("enabled", True): continue
        raw = fetch_and_clean(doc["url"])
        if not raw: continue
        all_chunks.extend(chunk_text(raw, doc["url"], doc["title"]))
    if not all_chunks:
        raise RuntimeError("No chunks ingested from sources.")
    vectors = embed_texts([c["text"] for c in all_chunks], embedder)
    index = build_faiss(vectors)
    side = {"chunks": all_chunks, "vectors_shape": vectors.shape}
    return index, side

# Build on import (non-fatal)
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

def retrieve(query: str, index, side: Dict[str, Any]) -> List[Dict[str, Any]]:
    import faiss  # noqa
    embedder, reranker = load_embedder(), load_reranker()
    qv = embed_texts([query], embedder)
    scores, idxs = index.search(qv, K_CANDIDATES)
    candidates = []
    for rank, (ci, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        if ci < 0: continue
        c = side["chunks"][ci]
        candidates.append({"rank_ann": rank, "score_ann": float(s), **c})
    if not candidates: return []
    pairs = [(query, c["text"]) for c in candidates]
    rer = reranker.predict(pairs, batch_size=64).tolist()
    for c, rs in zip(candidates, rer): c["score_rerank"] = float(rs)
    return sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)[:TOP_K]

def rag_confidence(retrieved: List[Dict[str, Any]]) -> float:
    if not retrieved: return 0.0
    top = sorted((c.get("score_rerank", 0.0) for c in retrieved), reverse=True)[:3]
    return sum(top)/len(top) if top else 0.0

# =========================
# Search + Read tools
# =========================
@st.cache_resource(show_spinner=False)
def get_ddg():
    return DDGS()

EXPANSIONS = [
    "{q}",
    "{q} overview",
    "{q} definition",
    "{q} tutorial",
    "{q} research paper",
    "{q} 2024 OR 2025",
]

def web_search(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    results = []
    ddg = get_ddg()
    for variant in EXPANSIONS:
        q = variant.format(q=query)
        try:
            for r in ddg.text(keywords=q, max_results=max_results, safesearch="moderate"):
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                snippet = r.get("body") or ""
                if title and url:
                    results.append({"title": title, "url": url, "snippet": snippet})
        except Exception:
            pass
        if results: break
    return results

def read_url(url: str, max_chars: int = 12000) -> Dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 TutorAI/1.0",
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type","")).lower()
        if ("pdf" in ctype) or url.lower().endswith(".pdf"):
            text = _clean_pdf(r.content)
            return {"title": url, "url": url, "text": text[:max_chars], "ok": True}
        text = _clean_html(r.content)
        # opportunistic: follow first PDF link on page if present
        try:
            soup = BeautifulSoup(r.content, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.lower().endswith(".pdf"):
                    pdf_url = requests.compat.urljoin(url, href)
                    r2 = requests.get(pdf_url, headers=headers, timeout=20)
                    if r2.ok and "pdf" in (r2.headers.get("Content-Type","")).lower():
                        text_pdf = _clean_pdf(r2.content)
                        text = text + "\n\n[PDF extract]\n" + text_pdf
                        break
        except Exception:
            pass
        return {"title": url, "url": url, "text": text[:max_chars], "ok": True}
    except requests.HTTPError as he:
        code = he.response.status_code if he.response is not None else "?"
        return {"title": url, "url": url, "text": f"[HTTP {code} error]", "ok": False}
    except Exception as e:
        return {"title": url, "url": url, "text": f"[Error: {e}]", "ok": False}

# =========================
# HF Chat wrapper
# =========================
def hf_chat(model: str, messages: List[Dict[str, str]], max_new_tokens=600, temperature=0.3, top_p=0.9) -> str:
    client = InferenceClient(model=model, token=HF_TOKEN)
    resp = client.chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    choice = getattr(resp, "choices", None) or resp["choices"]
    msg = getattr(choice[0], "message", None) or choice[0]["message"]
    content = getattr(msg, "content", None) or msg["content"]
    return (content or "").strip()

# =========================
# Agent pipeline (simple)
# =========================
def build_context(rag_chunks: List[Dict[str,Any]], pages: List[Dict[str,str]]) -> Tuple[str, List[Dict[str,str]]]:
    """Return (context_text, citations) with [n] mapping; only cite ok pages."""
    citations: List[Dict[str,str]] = []
    url_to_idx = {}
    lines = []

    def cite(url: str, title: str):
        if url not in url_to_idx:
            url_to_idx[url] = len(url_to_idx) + 1
            citations.append({"title": title or url, "url": url})
        return url_to_idx[url]

    if rag_chunks:
        lines.append("### RAG Evidence")
        for c in rag_chunks:
            idx = cite(c.get("url",""), c.get("title","source"))
            snippet = (c.get("text") or c.get("chunk",""))[:800]
            lines.append(f"[{idx}] {c.get('title','source')}\n{snippet}\n")

    ok_pages = [p for p in pages if p.get("ok")]
    if ok_pages:
        lines.append("### Web Pages")
        for p in ok_pages[:5]:
            idx = cite(p.get("url",""), p.get("title", p.get("url","")))
            snippet = (p.get("text","") or "")[:1000]
            lines.append(f"[{idx}] {p.get('title', p.get('url',''))}\n{snippet}\n")

    context = "\n".join(lines)
    return context, citations

def agent_answer(user_q: str, model_id: str) -> Tuple[str, Dict[str, Any]]:
    trace = {"steps": []}
    rag_chunks: List[Dict[str,Any]] = []
    web_pages: List[Dict[str,str]] = []

    # 1) Try RAG first (if index available)
    try:
        ensure_rag_ready()
        retrieved = retrieve(user_q, _global_rag["index"], _global_rag["side"])
    except Exception as e:
        retrieved = []
        trace["steps"].append({"tool":"rag_retrieve","input":user_q,"observation":f"RAG unavailable: {e}"})

    if retrieved:
        conf = rag_confidence(retrieved)
        # keep only minimal fields for citations
        rag_chunks = [{"title":c["title"],"url":c["url"],"text":c["text"]} for c in retrieved]
        trace["steps"].append({"tool":"rag_retrieve","input":user_q,"observation":f"{len(retrieved)} chunks; confidence={conf:.2f}"})
    else:
        conf = 0.0
        trace["steps"].append({"tool":"rag_retrieve","input":user_q,"observation":"0 chunks"})

    # 2) Gate: if no RAG or low confidence, do web_search then read_url top 2
    if (not retrieved) or (conf < RAG_CONF_THRESHOLD):
        sr = web_search(user_q, max_results=8)
        trace["steps"].append({"tool":"web_search","input":user_q,"observation":f"{len(sr)} hits"})
        if sr:
            # take first 2 unique URLs and read them
            seen = set()
            for hit in sr:
                u = hit.get("url","")
                if not u or u in seen: continue
                seen.add(u)
                page = read_url(u)
                web_pages.append(page)
                obs = page["text"][:160].replace("\n"," ")
                trace["steps"].append({"tool":"read_url","input":u,"observation":obs + ("…" if len(page['text'])>160 else "")})
                if len(web_pages) >= 2: break

    # 3) Build context and synthesize final answer
    context_text, citations = build_context(rag_chunks, web_pages)
    if not context_text.strip():
        return ("I couldn’t retrieve enough credible evidence to answer that. Try rephrasing the question or asking for a definition/overview.", trace)

    cite_lines = "\n".join([f"- [{c['title']}]({c['url']})" for c in citations if c.get("url")])

    sys = (
        "You are GenAI-Tutor. Answer ONLY with the EVIDENCE provided.\n"
        "Add [n] citations after claims that come from a source, where n matches the Sources list below.\n"
        "Do not invent URLs. If evidence is insufficient, say so explicitly."
    )
    usr = (
        f"USER QUESTION:\n{user_q}\n\n"
        f"EVIDENCE:\n{context_text}\n\n"
        "Write a concise answer with inline [n] citations and finish with a 'Sources' list."
    )

    try:
        ans = hf_chat(model_id, [{"role":"system","content":sys},{"role":"user","content":usr}], max_new_tokens=700)
    except Exception as e:
        ans = f"⚠️ Generation failed: {e}"

    if cite_lines:
        ans += "\n\n**Sources:**\n" + cite_lines
    return ans, trace

# =========================
# UI
# =========================
with st.sidebar:
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    if st.button("Rebuild RAG Index"):
        try:
            idx, side = build_rag_index(DOC_LINKS)
            _global_rag["index"], _global_rag["side"] = idx, side
            st.success(f"RAG ready: {side['vectors_shape'][0]} chunks.")
        except Exception as e:
            st.error(f"RAG build failed: {e}")

st.caption(f"Model: **{model_id}** — Tools: RAG (gated), Web Search, Read URL")

query = st.text_input("Ask a question (the agent will decide tools):", value="")
if st.button("Ask") and query.strip():
    with st.spinner("Thinking with tools…"):
        answer, tool_trace = agent_answer(query.strip(), model_id)
    st.markdown("### Answer")
    st.write(answer)
    with st.expander("🔍 Tool Trace"):
        for i, s in enumerate(tool_trace.get("steps", []), start=1):
            st.markdown(f"**Step {i}: {s['tool']}**")
            st.code(json.dumps({k:v for k,v in s.items() if k!='observation'}, ensure_ascii=False, indent=2))
            st.text(s.get("observation",""))
