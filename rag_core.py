"""
Shared Retrieval-Augmented Generation core for GenAI-Tutor (Versions 2, 3, 4).

Pipeline:
    fetch -> clean (HTML/PDF) -> chunk -> embed (bge-small) -> FAISS (cosine)
    -> rerank (cross-encoder) -> top-k, plus citation helpers.

Streamlit caching is used so models and the index are built once per session and
reused across reruns. This module is the single source of truth for retrieval;
the app files import from it instead of each keeping their own copy.
"""
import io
import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
TOP_K = 7
K_CANDIDATES = 30
WORDS_PER_CHUNK = 450          # roughly 600 tokens
OVERLAP_WORDS = 80
MIN_RERANK_SCORE = 0.05        # used by the agentic edition to drop weak matches

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Curated, publicly accessible corpus (Generative AI in education)
# ---------------------------------------------------------------------------
DOC_LINKS: List[Dict[str, Any]] = [
    {"title": "Ethical & Regulatory Challenges of GenAI in Education (2025) — Frontiers",
     "url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1565938/full",
     "enabled": True},
    {"title": "Learn Your Way: Reimagining Textbooks with Generative AI (2025) — Google",
     "url": "https://blog.google/outreach-initiatives/education/learn-your-way/",
     "enabled": True},
    {"title": "Student Generative AI Survey 2025 — HEPI",
     "url": "https://www.hepi.ac.uk/reports/student-generative-ai-survey-2025/",
     "enabled": True},
    {"title": "Educational impacts of generative AI on learning & performance (2025) — Nature (PDF)",
     "url": "https://www.nature.com/articles/s41598-025-06930-w.pdf",
     "enabled": True},
    {"title": "Enhancing Retrieval-Augmented Generation: Best Practices — COLING 2025 (PDF)",
     "url": "https://aclanthology.org/2025.coling-main.449.pdf",
     "enabled": True},
    {"title": "Large Language Models for Education: A Survey (2024) — arXiv (PDF)",
     "url": "https://arxiv.org/pdf/2403.18105",
     "enabled": True},
    {"title": "Generative AI for Education (GAIED): Advances, Opportunities, Challenges (2024) — arXiv (PDF)",
     "url": "https://arxiv.org/pdf/2402.01580",
     "enabled": True},
]

# ---------------------------------------------------------------------------
# Fetch & clean
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _download(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 TutorAI/2.0"})
    r.raise_for_status()
    return r.content, (r.headers.get("Content-Type", "")).lower()


def _clean_html(html_bytes: bytes) -> str:
    try:
        soup = BeautifulSoup(html_bytes, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
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
    """Download and extract clean text. Returns '' on any failure (skip silently)."""
    try:
        blob, ctype = _download(url)
        if ".pdf" in url.lower() or "application/pdf" in ctype:
            return _clean_pdf(blob)
        return _clean_html(blob)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, url: str, title: str,
               target_words: int = WORDS_PER_CHUNK,
               overlap_words: int = OVERLAP_WORDS) -> List[Dict[str, Any]]:
    if not text:
        return []
    words = [w for w in text.replace(" ", " ").split() if w]
    chunks, start, k = [], 0, 0
    while start < len(words):
        end = min(start + target_words, len(words))
        piece = " ".join(words[start:end])
        chunk_id = f"{hashlib.sha1(url.encode()).hexdigest()}#{k:04d}"
        chunks.append({"chunk_id": chunk_id, "title": title, "url": url, "text": piece})
        if end == len(words):
            break
        start = max(0, end - overlap_words)
        k += 1
    return chunks


# ---------------------------------------------------------------------------
# Embeddings & reranker
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner=True)
def load_reranker() -> CrossEncoder:
    # Lightweight cross-encoder (~90MB): fits Streamlit Cloud free tier and is fast on CPU.
    return CrossEncoder(RERANK_MODEL)


def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")


# ---------------------------------------------------------------------------
# Vector index (FAISS, with a NumPy fallback if faiss is unavailable)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    try:
        import faiss
        index = faiss.IndexFlatIP(vectors.shape[1])   # cosine via inner product on normalized vecs
        index.add(vectors)
        return index
    except Exception:
        class NpIndex:
            def __init__(self, V):
                self.V = V

            def search(self, qv, k):
                sims = qv @ self.V.T
                idxs = np.argsort(-sims, axis=1)[:, :k]
                scores = np.take_along_axis(sims, idxs, axis=1)
                return scores, idxs
        return NpIndex(vectors)


@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, Any]]):
    embedder = load_embedder()
    all_chunks: List[Dict[str, Any]] = []
    for doc in doc_links:
        if not doc.get("enabled", True):
            continue
        raw = fetch_and_clean(doc["url"])
        if not raw or len(raw.split()) < 100:   # skip empty / blocked / trivially short pages
            continue
        all_chunks.extend(chunk_text(raw, doc["url"], doc["title"]))
    if not all_chunks:
        raise RuntimeError("No chunks ingested from the selected sources.")
    vectors = embed_texts([c["text"] for c in all_chunks], embedder)
    index = build_faiss(vectors)
    side = {"chunks": all_chunks, "vectors_shape": vectors.shape}
    return index, side


def refresh_rag_cache():
    st.cache_resource.clear()
    st.cache_data.clear()


# ---------------------------------------------------------------------------
# Retrieve -> rerank -> top-k
# ---------------------------------------------------------------------------
def retrieve(query: str, index, side: Dict[str, Any],
             top_k: int = TOP_K, k_candidates: int = K_CANDIDATES,
             min_score: float = 0.0) -> Tuple[List[Dict[str, Any]], float]:
    """
    Two-stage retrieval. Returns (results, avg_rerank_score).
    Pass min_score > 0 to drop weak matches (the agentic edition uses MIN_RERANK_SCORE).
    """
    embedder = load_embedder()
    reranker = load_reranker()

    qv = embed_texts([query], embedder)
    scores, idx = index.search(qv, k_candidates)
    cand_ids, cand_scores = idx[0].tolist(), scores[0].tolist()

    candidates = []
    for pos, (ci, s) in enumerate(zip(cand_ids, cand_scores)):
        if ci < 0:
            continue
        c = side["chunks"][ci]
        candidates.append({"rank_ann": pos + 1, "score_ann": float(s), **c})
    if not candidates:
        return [], 0.0

    try:
        pairs = [(query, c["text"]) for c in candidates]
        rerank_scores = reranker.predict(pairs, batch_size=64).tolist()
        for c, rs in zip(candidates, rerank_scores):
            c["score_rerank"] = float(rs)
    except Exception:
        for c in candidates:
            c["score_rerank"] = float(c["score_ann"])

    kept = [c for c in candidates if c["score_rerank"] >= min_score] if min_score > 0 else candidates
    if not kept:
        return [], 0.0

    kept.sort(key=lambda x: x["score_rerank"], reverse=True)
    kept = kept[:top_k]
    avg = sum(c["score_rerank"] for c in kept) / len(kept)
    return kept, avg


# ---------------------------------------------------------------------------
# Context + citations
# ---------------------------------------------------------------------------
def build_context_and_citations(retrieved: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, Any]]]:
    url_to_ref: Dict[str, int] = {}
    refs: List[str] = []
    blocks: List[str] = []
    for c in retrieved:
        u = c["url"]
        if u not in url_to_ref:
            url_to_ref[u] = len(url_to_ref) + 1
            refs.append(f"[{url_to_ref[u]}] {u} — {c['title']}")
        r = url_to_ref[u]
        snippet = c["text"].strip()
        snippet = (snippet[:800] + "…") if len(snippet) > 800 else snippet
        blocks.append(f"[{r}] {c['title']}\n{snippet}\n")
    return "\n\n".join(blocks), "\n".join(refs), retrieved


def rag_rules() -> str:
    return ("Use ONLY the provided CONTEXT. Cite like [1], [2] after claims tied to evidence. "
            "If context is insufficient, say so and suggest which source to read. Do NOT invent URLs. "
            "End with a 'Sources' list mapping [n] to URL.")
