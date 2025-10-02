# app.py
# GenAI-Tutor (Agentic) — RAG + 3 Tools (WebSearch, URLReader, RAGRetriever) via LangChain
# ----------------------------------------------------------------------------------------
# - HF-only (no OpenAI). Uses Hugging Face Inference and tokenizers locally for counts if needed.
# - Tools:
#     1) WebSearchTool (DuckDuckGo, no API key)  -> top results (title, url, snippet)
#     2) URLReaderTool  (requests + trafilatura/pypdf) -> cleaned text from HTML/PDF
#     3) RAGRetrieverTool (your FAISS+BGE index) -> top-k chunks with URLs/titles
# - Agent: ZERO_SHOT_REACT_DESCRIPTION (ReAct), step-capped, with optional memory.
# - UI: Sidebar settings, RAG management, chat with agent, step trace, and evidence display.
# - Observability: Simple prints to Streamlit; (optional) wire to LangSmith if desired.

import os
import io
import time
import json
import hashlib
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
import streamlit as st

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

# DuckDuckGo search (no key)
from duckduckgo_search import DDGS

# HF chat (direct for simple calls) + tokenizer (optional)
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# LangChain agent + tools
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory

# ===========================
# App Config & Secrets
# ===========================
st.set_page_config(page_title="GenAI-Tutor (Agentic RAG)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Agentic Learning Assistant (RAG + Tools)</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN in Streamlit Secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)

# ===========================
# HF Chat Models (open-source)
# ===========================
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

# ===========================
# Learning Scenarios (for tone)
# ===========================
SCENARIOS: Dict[str, Dict[str, str]] = {
    "Prompt Engineering Basics": {
        "overview": """- Core prompting concepts (role, task, context, constraints)
- Patterns: few-shot, step-by-step, style/format guides
- Practical templates for summaries, emails, brainstorming
- Reducing hallucinations (be specific, ask for sources)""",
        "system": "You are GenAI-Tutor, an expert coach on prompt engineering for employees. Be concise, practical, and safe."
    },
    "Responsible & Secure GenAI at Work": {
        "overview": """- Safe inputs (no confidential/PII), data minimization
- Policy-aligned usage, approvals
- Phishing/social engineering risks
- Checklists and red flags""",
        "system": "You are GenAI-Tutor for responsible, secure GenAI usage at work. Teach practical, checklist-driven guidance."
    },
    "Automating Everyday Tasks with GenAI": {
        "overview": """- Draft emails, notes, briefs, SOPs
- Idea generation & prioritization
- Notes → structured outputs (tables, action items)
- Time-saving workflows""",
        "system": "You are GenAI-Tutor for everyday task automation. Provide templates and quick workflows."
    },
    "Writing & Communication with GenAI": {
        "overview": """- Tone targeting and audience fit
- Rewrite/expand/condense with structure and clarity
- Persuasive & empathetic patterns
- Review checklists""",
        "system": "You are GenAI-Tutor for business writing with Gen-AI. Focus on clarity, inclusivity, and concise structure."
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# ===========================
# Accessible RAG Sources (5)
# ===========================
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

# ============================================================
#                    RAG: Ingestion & Retrieval
# ============================================================
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
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            t.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = html_bytes.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def _clean_pdf(pdf_bytes: bytes) -> str:
    text = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except Exception:
            text.append("")
    lines = [ln.strip() for ln in "\n".join(text).splitlines()]
    return "\n".join([ln for ln in lines if ln])

def fetch_and_clean(url: str) -> str:
    try:
        blob, ctype = _download(url)
        if ".pdf" in url.lower() or "application/pdf" in ctype:
            return _clean_pdf(blob)
        return _clean_html(blob)
    except requests.HTTPError as he:
        code = he.response.status_code if he.response is not None else "?"
        st.info(f"Skipping (HTTP {code}): {url}")
        return ""
    except Exception as e:
        st.info(f"Skipping (fetch error): {url} ({e})")
        return ""

def _to_words(text: str) -> List[str]:
    return [w for w in text.replace("\u00a0", " ").split() if w]

def chunk_text(text: str, url: str, title: str,
               target_words: int = WORDS_PER_CHUNK, overlap_words: int = OVERLAP_WORDS) -> List[Dict[str, Any]]:
    if not text:
        return []
    words = _to_words(text)
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
    index = faiss.IndexFlatIP(d)  # cosine via IP on normalized vectors
    index.add(vectors)
    return index

@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, Any]]):
    embedder = load_embedder()
    all_chunks: List[Dict[str, Any]] = []
    for doc in doc_links:
        if not doc.get("enabled", True):
            continue
        raw = fetch_and_clean(doc["url"])
        if not raw:
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

def retrieve(query: str, index, side: Dict[str, Any],
             top_k: int = TOP_K, k_candidates: int = K_CANDIDATES) -> List[Dict[str, Any]]:
    import faiss  # noqa
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
        return []

    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=64).tolist()
    for c, rs in zip(candidates, rerank_scores):
        c["score_rerank"] = float(rs)
    candidates.sort(key=lambda x: x["score_rerank"], reverse=True)
    return candidates[:top_k]

# ============================================================
#                         Tools
# ============================================================
@st.cache_data(show_spinner=False)
def cached_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Cache DuckDuckGo results for a few minutes."""
    out = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results):
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                snippet = r.get("body") or ""
                if title and url:
                    out.append({"title": title, "url": url, "snippet": snippet})
    except Exception as e:
        st.info(f"Web search error ({e}).")
    return out

@tool
def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web (DuckDuckGo) and return a list of {'title','url','snippet'}."""
    return cached_search(query=query, max_results=max_results)

@st.cache_data(show_spinner=False)
def cached_read_url(url: str, max_chars: int = 6000) -> Dict[str, str]:
    """Fetch and clean a URL (HTML/PDF). Returns {'title','url','text'}."""
    try:
        resp = requests.get(url, headers={"User-Agent": "TutorAI/1.0"}, timeout=25)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type", "")).lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            # PDF
            text = _clean_pdf(resp.content)
            title = url
        else:
            # HTML
            text = _clean_html(resp.content)
            title = url
        text = (text or "")[:max_chars]
        return {"title": title, "url": url, "text": text}
    except Exception as e:
        return {"title": url, "url": url, "text": f"[Error fetching URL: {e}]"}
    
@tool
def read_url(url: str, max_chars: int = 6000) -> Dict[str, str]:
    """Open a URL and extract clean text from HTML or PDF. Returns {'title','url','text'}."""
    return cached_read_url(url=url, max_chars=max_chars)

# RAGRetriever tool depends on an index built at runtime
_global_rag = {"index": None, "side": None}

@tool
def rag_retrieve(query: str, top_k: int = 7) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from TutorAI’s curated Gen-AI corpus. Returns [{'title','url','chunk','score'}]."""
    if _global_rag["index"] is None or _global_rag["side"] is None:
        return [{"error": "RAG index not initialized. Press 'Build/Refresh RAG Index' in the app and retry."}]
    index, side = _global_rag["index"], _global_rag["side"]
    results = retrieve(query, index, side, top_k=top_k, k_candidates=max(2*top_k, 20))
    out = []
    for c in results:
        out.append({
            "title": c["title"],
            "url": c["url"],
            "chunk": c["text"][:1000],
            "score": round(float(c.get("score_rerank", 0.0)), 4),
        })
    return out

# ============================================================
#                    HF Chat (for direct calls)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_tokenizer_for(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id, use_fast=True, token=HF_TOKEN)
    except Exception:
        return None

def count_chat_tokens(model_id: str, messages: List[Dict[str, str]], completion_text: str | None = None) -> tuple[int, int]:
    tok = get_tokenizer_for(model_id)
    if tok is None:
        return 0, 0
    # Try chat template
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt_tokens = len(ids)
    except Exception:
        prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        prompt_tokens = len(tok(prompt_text).input_ids)
    completion_tokens = len(tok(completion_text).input_ids) if completion_text else 0
    return prompt_tokens, completion_tokens

# ============================================================
#                       Sidebar Settings
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    st.caption("HF token comes from Secrets.")
    st.markdown("---")
    st.subheader("🧰 Agent Tools")
    enable_agent = st.checkbox("Enable Tools (Agentic)", value=True)
    max_steps = st.slider("Max tool calls per turn", min_value=1, max_value=5, value=3)
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
                st.error(f"Failed to build RAG index: {e}")
    with c2:
        st.caption("Uses 5 curated sources; top-k=7 + reranker.")

st.caption(f"Model: **{model_id}**  •  Scenario: **{scenario_name}**")

# ============================================================
#                Agent Construction (LangChain)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_agent(model_id: str, use_tools: bool, max_iterations: int):
    # HF endpoint as LangChain ChatModel
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
    )
    llm = ChatHuggingFace(llm=endpoint)

    # Memory: keep running dialogue (optional)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Tool list
    tools = [rag_retrieve] if not use_tools else [web_search, read_url, rag_retrieve]

    # Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # ReAct-style
        verbose=False,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        memory=memory,
    )
    return agent

# ============================================================
#                        Chat State
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
        {"role": "assistant", "content": "Hello! I’m GenAI-Tutor. Ask me anything about Gen-AI."}
    ]
if "scenario_prev" not in st.session_state:
    st.session_state.scenario_prev = scenario_name

def _seed_chat():
    st.session_state.messages = [
        {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
        {"role": "assistant", "content": "Hello! I’m GenAI-Tutor. Ask me anything about Gen-AI."}
    ]

if st.session_state.scenario_prev != scenario_name:
    _seed_chat()
    st.session_state.scenario_prev = scenario_name

# ============================================================
#                       Chat UI + Agent
# ============================================================
st.markdown("---")
st.subheader("💬 Tutor Chat (Agentic)")

# Show history
for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

user_prompt = st.chat_input("Ask a question (the agent may search the web, read a URL, or use RAG)…")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Build agent
    agent = get_agent(model_id=model_id, use_tools=enable_agent, max_iterations=max_steps)

    # Build the agent input using last N turns as context string (agent also has memory)
    # You can keep it simple and just pass the last user question:
    agent_input = {"input": user_prompt}

    with st.chat_message("assistant"):
        with st.spinner("Thinking with tools…"):
            try:
                # Return intermediate steps so we can show tool calls & observations
                result = agent.invoke(agent_input, config={"return_intermediate_steps": True})
                reply = result["output"]
                inter = result.get("intermediate_steps", [])
            except Exception as e:
                reply = f"⚠️ Agent error: {e}"
                inter = []

        st.markdown(reply)

        # Show agent trace
        if inter:
            with st.expander("🔍 Agent Trace (tools & observations)"):
                for i, (tool_call, observation) in enumerate(inter, start=1):
                    st.markdown(f"**Step {i}: {tool_call.tool}**")
                    try:
                        args_str = json.dumps(tool_call.tool_input, ensure_ascii=False)
                    except Exception:
                        args_str = str(tool_call.tool_input)
                    st.code(args_str, language="json")

                    # Observations can be long (e.g., chunks). Truncate for view.
                    obs_view = observation
                    try:
                        if isinstance(observation, (list, dict)):
                            obs_view = json.dumps(observation, ensure_ascii=False)[:1200]
                        elif isinstance(observation, str):
                            obs_view = observation[:1200]
                    except Exception:
                        obs_view = str(observation)[:1200]
                    st.text_area("Observation", value=obs_view, height=160)

        # Evidence view (from RAG tool if used)
        # Parse intermediate steps for rag_retrieve returns:
        rag_cards = []
        for (tool_call, observation) in inter:
            if tool_call.tool == "rag_retrieve" and isinstance(observation, list):
                for item in observation[:TOP_K]:
                    if isinstance(item, dict) and "title" in item and "url" in item and "chunk" in item:
                        rag_cards.append(item)
        if rag_cards:
            with st.expander("📚 Retrieved Evidence (RAG)"):
                for i, it in enumerate(rag_cards, 1):
                    st.markdown(f"**[{i}] [{it['title']}]({it['url']})**")
                    st.write(it["chunk"][:500] + ("…" if len(it["chunk"]) > 500 else ""))

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ============================================================
#                         Scenario Overview
# ============================================================
st.markdown("---")
st.subheader("📌 Scenario Overview")
st.markdown(f"**{scenario_name}**  \n{SCENARIOS[scenario_name]['overview']}")

# ============================================================
#                         Footer
# ============================================================
st.markdown("---")
st.caption("GenAI-Tutor is educational. Verify critical info. Follow your organization’s policies.")
