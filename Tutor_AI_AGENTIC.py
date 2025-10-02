# app.py
# GenAI-Tutor (Agentic) — RAG + Tools + LangSmith (HF-only, with safe fallback)
# ---------------------------------------------------------------------------
# Tools:
#   1) web_search (DuckDuckGo) -> [{'title','url','snippet'}]
#   2) read_url (HTML/PDF fetch + clean) -> {'title','url','text'}
#   3) rag_retrieve (FAISS + BGE + reranker) -> [{'title','url','chunk','score'}]
# Agent: Structured Chat (multi-arg tools supported) with HF chat model
# RAG: top_k=7 with rerank; 5 curated public sources
# Observability: LangSmith tracing + thumbs feedback to latest run
# Reliability: verbose agent, raw result inspector, and direct-HF fallback reply

import os
import io
import json
import time
import hashlib
from typing import List, Dict, Any, Tuple

import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader

from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from huggingface_hub import InferenceClient

# LangSmith
from langsmith import Client
from langsmith.run_helpers import tracing_context, trace, get_current_run_tree

# LangChain (agent + tools + prompt)
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ===========================
# App Config & Secrets
# ===========================
st.set_page_config(page_title="GenAI-Tutor (Agentic RAG + Tools)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Agentic Learning Assistant (RAG + Tools + LangSmith)</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN in Streamlit Secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)

# LangSmith env
os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
os.environ["LANGSMITH_TRACING"] = str(st.secrets.get("LANGSMITH_TRACING", True)).lower()
LS_PROJECT = st.secrets.get("LANGSMITH_PROJECT", os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-Agentic"))
ls_client = Client()

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
# Learning Scenarios
# ===========================
SCENARIOS: Dict[str, Dict[str, str]] = {
    "Prompt Engineering Basics": {
        "overview": """- Core prompting concepts (role, task, context, constraints)
- Few-shot, step-by-step, style/format guides
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
# RAG Sources (5 public)
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
    index = faiss.IndexFlatIP(d)  # cosine via IP on normalized vecs
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
    for pos, (ci, s) in enumerate(zip(cand_ids, cand_scores), start=1):
        if ci < 0:
            continue
        c = side["chunks"][ci]
        candidates.append({"rank_ann": pos, "score_ann": float(s), **c})

    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = reranker.predict(pairs, batch_size=64).tolist()
    for c, rs in zip(candidates, rerank_scores):
        c["score_rerank"] = float(rs)
    candidates.sort(key=lambda x: x["score_rerank"], reverse=True)
    return candidates[:top_k]

# ============================================================
#                         Tools (schema-based)
# ============================================================
@st.cache_data(show_spinner=False)
def cached_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    out = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results, safesearch="moderate"):
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                snippet = r.get("body") or ""
                if title and url:
                    out.append({"title": title, "url": url, "snippet": snippet})
    except Exception as e:
        st.info(f"Web search error ({e}).")
    return out

class WebSearchInput(BaseModel):
    query: str = Field(..., description="The web search query.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return (1-10).")

from langchain_core.tools import tool as lc_tool  # alias to avoid confusion with st.tool

@lc_tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (DuckDuckGo) and return a list of {'title','url','snippet'}."""
    return cached_search(query=query, max_results=max_results)

@st.cache_data(show_spinner=False)
def cached_read_url(url: str, max_chars: int = 6000) -> Dict[str, str]:
    try:
        resp = requests.get(url, headers={"User-Agent": "TutorAI/1.0"}, timeout=25)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type", "")).lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = _clean_pdf(resp.content)
        else:
            text = _clean_html(resp.content)
        return {"title": url, "url": url, "text": (text or "")[:max_chars]}
    except Exception as e:
        return {"title": url, "url": url, "text": f"[Error fetching URL: {e}]"}
    
class ReadUrlInput(BaseModel):
    url: str = Field(..., description="The URL to fetch (HTML or PDF).")
    max_chars: int = Field(6000, ge=500, le=20000, description="Max characters to return.")

@lc_tool("read_url", args_schema=ReadUrlInput)
def read_url_tool(url: str, max_chars: int = 6000) -> dict:
    """Open a URL and extract clean text from HTML or PDF. Returns {'title','url','text'}."""
    return cached_read_url(url=url, max_chars=max_chars)

_global_rag = {"index": None, "side": None}

class RAGRetrieveInput(BaseModel):
    query: str = Field(..., description="The user query to retrieve evidence for.")
    top_k: int = Field(7, ge=1, le=10, description="Top-k chunks to return.")

@lc_tool("rag_retrieve", args_schema=RAGRetrieveInput)
def rag_retrieve_tool(query: str, top_k: int = 7) -> list[dict]:
    """Retrieve top-k chunks from TutorAI’s curated Gen-AI corpus. Returns [{'title','url','chunk','score'}]."""
    if _global_rag["index"] is None or _global_rag["side"] is None:
        return [{"error": "RAG index not initialized. Click 'Build/Refresh RAG Index' and try again."}]
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
#                    HF Tokenizer (optional counts)
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
    try:
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        prompt_tokens = len(ids)
    except Exception:
        prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        prompt_tokens = len(tok(prompt_text).input_ids)
    completion_tokens = len(tok(completion_text).input_ids) if completion_text else 0
    return prompt_tokens, completion_tokens

# ============================================================
#                 Direct HF Fallback (no tools)
# ============================================================
def hf_direct_reply(model_id: str, user_text: str, system_text: str = "") -> str:
    """
    If the agent fails or returns empty, use a simple HF chat completion so the user always gets an answer.
    """
    try:
        client = InferenceClient(model=model_id, token=HF_TOKEN)
        msgs = []
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        msgs.append({"role": "user", "content": user_text})
        resp = client.chat_completion(messages=msgs, max_tokens=512, temperature=0.5, top_p=0.9)
        choice = resp.choices[0]
        msg = getattr(choice, "message", None) or choice["message"]
        content = getattr(msg, "content", None) or msg["content"]
        return (content or "").strip()
    except Exception as e:
        return f"⚠️ Fallback HF error: {e}"

# ============================================================
#                       Sidebar Settings
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    st.caption("HF & LangSmith settings come from Secrets.")
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
        st.caption("Uses 5 curated sources; top-k=7 with reranking.")
st.caption(f"Model: **{model_id}**  •  Scenario: **{scenario_name}**")

# ============================================================
#          Agent Construction (Structured Chat + tools)
# ============================================================
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@st.cache_resource(show_spinner=False)
def get_agent(model_id: str, use_tools: bool, max_iterations: int):
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
    )
    llm = ChatHuggingFace(llm=endpoint)
    tools = [rag_retrieve_tool] if not use_tools else [web_search_tool, read_url_tool, rag_retrieve_tool]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are GenAI-Tutor, a safe and expert assistant for Gen-AI learning.\n"
             "You may use external tools when helpful. Here are the tools you can use:\n{tools}\n\n"
             "When you decide to use a tool, call it with the correct JSON args that match its schema.\n"
             "Available tool names you may call: {tool_names}\n"
             "Cite sources when you rely on external content. If a tool fails, try another or summarize partial results."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,                    # 👈 turn on verbose so you see planning/parsing errors in logs/LangSmith
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        memory=memory,
    )
    return executor

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
if "turn_logs" not in st.session_state:
    st.session_state.turn_logs = []

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

# History
for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

user_q = st.chat_input("Ask a question. The agent may use WebSearch, ReadURL, or RAG…")
if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})

    agent = get_agent(model_id=model_id, use_tools=enable_agent, max_iterations=max_steps)
    scenario_system = SCENARIOS[scenario_name]["system"]
    agent_input = {"input": f"{scenario_system}\n\nUser question: {user_q}"}

    with tracing_context(project_name=LS_PROJECT, metadata={
        "type": "agent_turn", "scenario": scenario_name, "model": model_id, "tools": "on" if enable_agent else "rag_only"
    }):
        with trace("agent_turn", run_type="chain", inputs={"question": user_q}):
            with st.chat_message("assistant"):
                with st.spinner("Thinking with tools…"):
                    result = {}
                    error_text = ""
                    try:
                        result = agent.invoke(
                            agent_input,
                            config={
                                "metadata": {"scenario": scenario_name, "model": model_id, "tools_enabled": enable_agent},
                                "tags": ["TutorAI", "Agentic", "RAG"],
                                "run_name": "TutorAI-Agentic-Turn",
                            },
                        )
                        reply = result.get("output", "")
                    except Exception as e:
                        reply = ""
                        error_text = f"Agent exception: {e}"

                # If the agent produced nothing, do a direct HF fallback so user always sees something
                if not reply or not reply.strip():
                    fallback_prompt = (
                        "Answer clearly. If you used any web content or documents, mention sources.\n\n"
                        f"Question: {user_q}"
                    )
                    reply = hf_direct_reply(model_id, fallback_prompt, system_text=scenario_system)

                st.markdown(reply)

                # Show raw result for debugging if empty or if errors happened
                if error_text or not result:
                    with st.expander("⚙️ Debug (raw agent result / errors)"):
                        if error_text:
                            st.error(error_text)
                        st.json(result or {"note": "No result object returned by agent."})

                # Agent trace (tool calls)
                inter = result.get("intermediate_steps", [])
                if inter:
                    with st.expander("🔍 Agent Trace (tools & observations)"):
                        for i, (tool_call, observation) in enumerate(inter, start=1):
                            st.markdown(f"**Step {i}: `{tool_call.tool}`**")
                            try:
                                args_str = json.dumps(tool_call.tool_input, ensure_ascii=False)
                            except Exception:
                                args_str = str(tool_call.tool_input)
                            st.code(args_str, language="json")
                            # observation preview
                            obs_view = observation
                            try:
                                if isinstance(observation, (list, dict)):
                                    obs_view = json.dumps(observation, ensure_ascii=False)[:1200]
                                elif isinstance(observation, str):
                                    obs_view = observation[:1200]
                            except Exception:
                                obs_view = str(observation)[:1200]
                            st.text_area("Observation", value=obs_view, height=160)

                # Evidence from rag_retrieve (if used this turn)
                rag_cards = []
                for (tool_call, observation) in inter or []:
                    if tool_call.tool == "rag_retrieve" and isinstance(observation, list):
                        for item in observation[:TOP_K]:
                            if isinstance(item, dict) and all(k in item for k in ["title", "url", "chunk"]):
                                rag_cards.append(item)
                if rag_cards:
                    with st.expander("📚 Retrieved Evidence (RAG)"):
                        for i, it in enumerate(rag_cards, 1):
                            st.markdown(f"**[{i}] [{it['title']}]({it['url']})**")
                            st.write(it["chunk"][:500] + ("…" if len(it["chunk"]) > 500 else ""))

                # Thumbs → LangSmith
                with st.expander("Rate this answer"):
                    col1, col2 = st.columns(2)
                    if col1.button("👍 Helpful"):
                        try:
                            rid = str(get_current_run_tree().id)
                            ls_client.create_feedback(run_id=rid, key="user_score", score=1)
                            st.success("Thanks! Logged to LangSmith.")
                        except Exception:
                            st.info("Feedback logging failed (check LangSmith key).")
                    if col2.button("👎 Not helpful"):
                        try:
                            rid = str(get_current_run_tree().id)
                            ls_client.create_feedback(run_id=rid, key="user_score", score=0)
                            st.info("Feedback recorded.")
                        except Exception:
                            st.info("Feedback logging failed (check LangSmith key).")

            # Append to session log
            try:
                rid = str(get_current_run_tree().id)
            except Exception:
                rid = ""
            st.session_state.turn_logs.append({
                "run_id": rid,
                "question": user_q,
                "answer": reply,
                "contexts": [it.get("chunk", "") for it in rag_cards],
            })

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ===========================
#                         Scenario Overview
# ===========================
st.markdown("---")
st.subheader("📌 Scenario Overview")
st.markdown(f"**{scenario_name}**  \n{SCENARIOS[scenario_name]['overview']}")

# ===========================
#                         Footer
# ===========================
st.markdown("---")
st.caption("GenAI-Tutor is educational. Verify critical info. Follow your organization’s policies.")
