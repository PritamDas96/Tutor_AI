# GenAI-Tutor — RAG + LangSmith + RAGAS (Streamlit Cloud)
# -------------------------------------------------------
# - Sidebar: ONLY two dropdowns (Learning Scenario, HF Model)
# - RAG: fetch → clean → chunk (~600 tokens, 80 overlap) → embed (bge-small)
#        → FAISS (cosine via IP on normalized vecs) → rerank (bge-reranker) → top_k=7
# - Observability: LangSmith tracing for chat & retrieval; thumbs feedback; RAGAS eval panel.
# - RAG switch ON => minimal prompts & strict use of retrieved CONTEXT with inline [n] citations.

import os
import io
import time
import json
import hashlib
import requests
import numpy as np
from typing import List, Dict, Any, Tuple

import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from bs4 import BeautifulSoup
from pypdf import PdfReader

# -------- LangSmith (observability) --------
from langsmith import Client, traceable
from langsmith.run_helpers import trace, tracing_context, get_current_run_tree

# -------- RAGAS (evaluation) --------
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper

# -------- LangChain (judge LLM wrapper for RAGAS) --------
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ===========================
# App Config & Secrets
# ===========================
st.set_page_config(page_title="GenAI-Tutor (RAG + Observability)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Intelligent Conversational Learning Assistant</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN in Streamlit Secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)

os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
os.environ["LANGSMITH_TRACING"] = str(st.secrets.get("LANGSMITH_TRACING", True)).lower()
LS_PROJECT = st.secrets.get("LANGSMITH_PROJECT", os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-RAG"))
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
- Patterns: few-shot, step-by-step, style/format guides
- Practical templates for summaries, emails, brainstorming
- Ways to reduce hallucinations (be specific, ask for sources)""",
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
# Sidebar (ONLY two dropdowns)
# ===========================
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    st.caption("HF token & LangSmith settings come from Secrets.")
st.caption(f"Model in use: **{model_id}**  •  Scenario: **{scenario_name}**")

# ===========================
# Session State
# ===========================
if "scenario_prev" not in st.session_state:
    st.session_state.scenario_prev = scenario_name
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""
if "use_rag" not in st.session_state:
    st.session_state.use_rag = True
if "turn_logs" not in st.session_state:
    # For RAGAS: per-turn {'run_id','question','contexts':[...],'answer'}
    st.session_state.turn_logs: List[Dict[str, Any]] = []

def _seed_chat():
    st.session_state.messages = [
        {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
        {"role": "assistant", "content": "Hello! I’m GenAI-Tutor. What would you like to learn today?"}
    ]
if not st.session_state.messages or st.session_state.scenario_prev != scenario_name:
    _seed_chat()
    st.session_state.scenario_prev = scenario_name

# ===========================
# HF Chat Completion
# ===========================
@traceable(run_type="llm", name="hf_chat")
def call_hf_chat(model: str,
                 messages: List[Dict[str, str]],
                 token: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
    last_err = None
    for provider in (None, "hf-inference"):
        try:
            client = InferenceClient(model=model, token=token, provider=provider)
            resp = client.chat_completion(
                messages=messages,
                max_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
            )
            choice = resp.choices[0]
            msg = getattr(choice, "message", None) or choice["message"]
            content = getattr(msg, "content", None) or msg["content"]
            return (content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"Chat completion failed for {model}: {last_err}")

# ============================================================
#                         RAG CORE
# ============================================================
TOP_K = 7
K_CANDIDATES = 30
WORDS_PER_CHUNK = 450     # ~600 tokens (rough)
OVERLAP_WORDS = 80

# 5 accessible sources (public)
DOC_LINKS = [
    {
        "title": "Ethical & Regulatory Challenges of GenAI in Education (2025) — Frontiers",
        "url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1565938/full",
        "enabled": True
    },
    {
        "title": "Learn Your Way: Reimagining Textbooks with Generative AI (2025) — Google",
        "url": "https://blog.google/outreach-initiatives/education/learn-your-way/",
        "enabled": True
    },
    {
        "title": "Student Generative AI Survey 2025 — HEPI",
        "url": "https://www.hepi.ac.uk/reports/student-generative-ai-survey-2025/",
        "enabled": True
    },
    {
        "title": "Educational impacts of generative AI on learning & performance (2025) — Nature (PDF)",
        "url": "https://www.nature.com/articles/s41598-025-06930-w.pdf",
        "enabled": True
    },
    {
        "title": "Enhancing Retrieval-Augmented Generation: Best Practices — COLING 2025 (PDF)",
        "url": "https://aclanthology.org/2025.coling-main.449.pdf",
        "enabled": True
    },
]

# -------- Fetch & Clean --------
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

# -------- Chunking --------
def _to_words(text: str) -> List[str]:
    return [w for w in text.replace("\u00a0", " ").split() if w]

def chunk_text(text: str, url: str, title: str,
               target_words: int = WORDS_PER_CHUNK,
               overlap_words: int = OVERLAP_WORDS) -> List[Dict[str, Any]]:
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

# -------- Embeddings & Reranker --------
@st.cache_resource(show_spinner=True)
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=True)
def load_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-v2-m3")

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")

# -------- FAISS Index --------
@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    import faiss  # lazy import
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via IP on normalized vecs
    index.add(vectors)
    return index

# -------- Build RAG index --------
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

# -------- Retrieve → Rerank → top_k --------
@traceable(run_type="retriever", name="retrieve", metadata={"top_k": TOP_K})
def retrieve(query: str,
             index,
             side: Dict[str, Any],
             top_k: int = TOP_K,
             k_candidates: int = K_CANDIDATES) -> List[Dict[str, Any]]:
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
    return ("Use ONLY the provided CONTEXT. "
            "Cite like [1], [2] after claims tied to evidence. "
            "If context is insufficient, say so and suggest which source to read. "
            "Do NOT invent URLs. End with a 'Sources' list mapping [n] → URL.")

# ===========================
# Overview
# ===========================
st.subheader("📌 Scenario Overview")
st.markdown(f"**{scenario_name}**  \n{SCENARIOS[scenario_name]['overview']}")

# ===========================
# Notes (RAG-aware)
# ===========================
st.markdown("---")
with st.expander("📝 Personalized Study Notes (RAG-aware)", expanded=False):
    ROLE_OPTS = ["General","Manager","Analyst","Engineer/Developer","HR/People","Sales","Marketing",
                 "Operations","Finance","Customer Support","Legal/Compliance","Data/Analytics","Other"]
    TEAM_OPTS = ["General","HR","Finance","Marketing","Sales","IT/Engineering","Operations",
                 "Legal/Compliance","Customer Support","Data/Analytics","Other"]
    GOAL_OPTS = ["Use Gen-AI safely & responsibly","Write effective prompts","Automate routine tasks",
                 "Improve business writing","Summarize long content","Analyze/compare information",
                 "Build evaluation & guardrails","Other (type below)"]
    PAIN_OPTS = ["Unclear prompt structure","Fear of data leaks","Hallucinations/accuracy issues",
                 "Hard to control tone/style","Information overload","Tool overwhelm / where to start",
                 "Other (type below)"]

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        role_choice = st.selectbox("Role", ROLE_OPTS, index=0)
        role_other = st.text_input("Specify Role") if role_choice == "Other" else ""
        level = st.selectbox("Current Level", ["Beginner","Intermediate","Advanced"], index=0)
    with c2:
        team_choice = st.selectbox("Team / Domain", TEAM_OPTS, index=0)
        team_other = st.text_input("Specify Team/Domain") if team_choice == "Other" else ""
        time_per_day = st.text_input("Time Available / Day", value="15 minutes")
    with c3:
        style = st.selectbox("Preferred Style", ["Concise & example-driven","Step-by-step","Visual & analogies"], index=0)
        st.write("")

    def _finalize(choice, other): return other.strip() if (choice=="Other" and other.strip()) else choice
    role_val = _finalize(role_choice, role_other) or "General"
    team_val = _finalize(team_choice, team_other) or "General"

    def _merge(base_list, other_text, max_keep=3):
        fixed = [x for x in base_list if x!="Other (type below)"][:max_keep]
        more = [x.strip() for x in (other_text or "").split(",") if x.strip()]
        seen, out = set(), []
        for x in fixed + more:
            if x not in seen:
                out.append(x); seen.add(x)
        return ", ".join(out) if out else "(not provided)"

    goals_sel = st.multiselect("Your Top 3 Goals", GOAL_OPTS,
                               default=["Use Gen-AI safely & responsibly","Write effective prompts","Automate routine tasks"])
    goals_other = st.text_input("Other goals (comma-separated)") if "Other (type below)" in goals_sel else ""
    pains_sel = st.multiselect("Pain Points", PAIN_OPTS, default=["Unclear prompt structure","Fear of data leaks"])
    pains_other = st.text_input("Other pain points (comma-separated)") if "Other (type below)" in pains_sel else ""

    goals_val = _merge(goals_sel, goals_other, 3)
    pains_val = _merge(pains_sel, pains_other, 5)

    n1, n2, n3 = st.columns([1,1,1])
    with n1:
        gen_notes = st.button("Generate Notes", use_container_width=True)
    with n2:
        ins_notes = st.button("Insert Notes into Chat", use_container_width=True, disabled=not bool(st.session_state.notes_text))
    with n3:
        clr_notes = st.button("Clear Notes", use_container_width=True, disabled=not bool(st.session_state.notes_text))

    if gen_notes:
        with tracing_context(project_name=LS_PROJECT, metadata={"type": "notes_turn", "scenario": scenario_name, "use_rag": st.session_state.use_rag, "model": model_id}):
            with trace("notes_turn", run_type="chain", inputs={"profile": {
                "role": role_val, "team": team_val, "level": level, "goals": goals_val, "pains": pains_val, "style": style, "time": time_per_day
            }}):
                if st.session_state.use_rag:
                    try:
                        index, side = build_rag_index(DOC_LINKS)
                    except Exception as e:
                        index, side = None, None
                        st.error(f"RAG index unavailable: {e}")

                    if index is not None:
                        profile_query = (
                            f"{scenario_name} study guide for a {level} learner (role: {role_val}, team: {team_val}); "
                            f"goals: {goals_val}; pain points: {pains_val}; style: {style}; time/day: {time_per_day}."
                        )
                        with st.spinner("Retrieving evidence for your study notes…"):
                            retrieved = retrieve(profile_query, index, side, top_k=TOP_K, k_candidates=K_CANDIDATES)
                        if not retrieved:
                            st.warning("No evidence retrieved; cannot create grounded notes.")
                        else:
                            ctx, srcs, _ = build_context_and_citations(retrieved)
                            messages = [
                                {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
                                {"role": "system", "content": f"CONTEXT:\n{ctx}\n\nSOURCES:\n{srcs}\n\n{rag_rules()}"},
                                {"role": "user", "content": "Using ONLY the CONTEXT, produce a concise, personalized study guide with inline [n] citations and a final 'Sources' list."},
                            ]
                            with st.spinner("Drafting your personalized study guide…"):
                                try:
                                    st.session_state.notes_text = call_hf_chat(model_id, messages, HF_TOKEN)
                                except Exception as e:
                                    st.session_state.notes_text = f"⚠️ Error while generating notes: {e}"
                else:
                    # Non-RAG fallback (kept minimal)
                    messages = [
                        {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
                        {"role": "user", "content":
                         f"Create a concise study guide for '{scenario_name}' for a {level} learner (role {role_val}, team {team_val}). "
                         f"Goals: {goals_val}. Pains: {pains_val}. Style: {style}. Time/day: {time_per_day}. "
                         f"Include key concepts, practical patterns, 3–5 micro-exercises with hints, a mini checklist, and a 5-day plan."}
                    ]
                    with st.spinner("Drafting your study guide…"):
                        try:
                            st.session_state.notes_text = call_hf_chat(model_id, messages, HF_TOKEN)
                        except Exception as e:
                            st.session_state.notes_text = f"⚠️ Error while generating notes: {e}"

    if ins_notes and st.session_state.notes_text:
        st.session_state.messages.append({"role": "system", "content": f"Reference notes:\n\n{st.session_state.notes_text}"})
        st.success("Notes inserted into chat context.")
    if clr_notes and st.session_state.notes_text:
        st.session_state.notes_text = ""
        st.info("Notes cleared.")
    if st.session_state.notes_text:
        st.markdown("#### 📚 Your Study Guide")
        st.write(st.session_state.notes_text)

# ===========================
# RAG Controls
# ===========================
st.markdown("---")
st.subheader("🔎 RAG (Retrieval-Augmented Generation)")
rc1, rc2, rc3 = st.columns([1,1,2])
with rc1:
    st.session_state.use_rag = st.checkbox("Use RAG (Notes & Chat)", value=st.session_state.use_rag)
with rc2:
    if st.button("Refresh RAG Corpus"):
        refresh_rag_cache()
        st.success("RAG caches cleared. Index will rebuild on next request.")
with rc3:
    st.caption("Uses 5 accessible sources; in-memory index; top-k=7 with reranking.")

def get_rag_index():
    try:
        return build_rag_index(DOC_LINKS)
    except Exception as e:
        st.error(f"RAG index build failed: {e}")
        return None, None

# ===========================
# Chatbot
# ===========================
st.markdown("---")
st.subheader("💬 Tutor Chat")
cc1, cc2 = st.columns([1,4])
with cc1:
    if st.button("Reset Chat", use_container_width=True):
        _seed_chat()
        st.success("Chat reset.")

# render history
for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

user_prompt = st.chat_input("Ask anything about this Gen-AI learning scenario…")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with tracing_context(project_name=LS_PROJECT, metadata={"type": "chat_turn", "scenario": scenario_name, "use_rag": st.session_state.use_rag, "model": model_id}):
        with trace("chat_turn", run_type="chain", inputs={"question": user_prompt}) as root:
            messages_for_call = list(st.session_state.messages)
            evidence_to_show = []
            if st.session_state.use_rag:
                index, side = get_rag_index()
                if index is not None:
                    with st.spinner("Retrieving evidence…"):
                        retrieved = retrieve(user_prompt, index, side, top_k=TOP_K, k_candidates=K_CANDIDATES)
                    if retrieved:
                        ctx, srcs, evidence_to_show = build_context_and_citations(retrieved)
                        messages_for_call = [
                            messages_for_call[0],  # scenario system
                            {"role":"system","content": f"CONTEXT:\n{ctx}\n\nSOURCES:\n{srcs}\n\n{rag_rules()}"},
                            *messages_for_call[1:]
                        ]

            with st.chat_message("assistant"):
                try:
                    reply = call_hf_chat(model_id, messages_for_call, HF_TOKEN)
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
                st.markdown(reply)

                if st.session_state.use_rag and evidence_to_show:
                    with st.expander("🔗 Evidence (top 7)"):
                        for i, c in enumerate(evidence_to_show, start=1):
                            preview = c["text"][:280] + ("…" if len(c["text"]) > 280 else "")
                            st.markdown(f"**[{i}] [{c['title']}]({c['url']})**")
                            st.write(preview)

                # Thumbs feedback → LangSmith
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

            st.session_state.messages.append({"role": "assistant", "content": reply})
            try:
                rid = str(get_current_run_tree().id)
            except Exception:
                rid = ""
            st.session_state.turn_logs.append({
                "run_id": rid,
                "question": user_prompt,
                "answer": reply,
                "contexts": [c["text"] for c in (evidence_to_show or [])]
            })

# ===========================
# Observe & Evaluate (RAGAS)
# ===========================
st.markdown("---")
with st.expander("🔬 Observe & Evaluate (RAGAS over recent chats)"):
    st.write("Runs **faithfulness**, **answer relevancy**, and **context precision** on the last N chats of this session. Optionally logs aggregate metrics to LangSmith on the latest run.")
    N = st.slider("How many recent chats to evaluate?", min_value=1, max_value=50,
                  value=min(10, len(st.session_state.turn_logs)) if st.session_state.turn_logs else 5)
    if st.button("Run RAGAS now"):
        turns = st.session_state.turn_logs[-N:] if st.session_state.turn_logs else []
        if not turns:
            st.warning("No chats to evaluate yet.")
        else:
            data = []
            for t in turns:
                if t["question"] and t["answer"] and t["contexts"]:
                    data.append({
                        "user_input": t["question"],
                        "retrieved_contexts": t["contexts"],
                        "response": t["answer"],
                    })
            if not data:
                st.warning("No evaluable turns (need RAG contexts).")
            else:
                ds = EvaluationDataset.from_list(data)
                # Judge LLM via LangChain → HF Inference; wrap for RAGAS
                try:
                    endpoint = HuggingFaceEndpoint(
                        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",   # choose any chat-capable model you have access to
                        task="text-generation",
                        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                        max_new_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )
                    lc_chat = ChatHuggingFace(llm=endpoint)  # IMPORTANT: use keyword 'client'
                    judge = LangchainLLMWrapper(lc_chat)
                except Exception as e:
                    st.error(f"HF judge failed: {e}")
                    judge = None
                if judge is None:
                    st.stop() 
                    scores = evaluate(
                        dataset=ds,
                        metrics=[Faithfulness(), AnswerRelevancy()],
                        llm=judge,
                        show_progress=True,
                        )
                    st.subheader("📈 RAGAS Results")
                    try:
                        st.write(scores)
                    except Exception:
                        st.json(scores)

                    # Optional: push placeholder metrics to LangSmith (aggregate logging)
                    if st.button("Log aggregate metric placeholders to LangSmith (latest run)"):
                        try:
                            latest_run_id = turns[-1]["run_id"] or None
                            if not latest_run_id:
                                st.info("No run_id to attach feedback.")
                            else:
                                # You can parse 'scores' to extract real aggregates; here we write placeholders safely.
                                for k in ["faithfulness", "answer_relevancy", "context_precision"]:
                                    ls_client.create_feedback(run_id=latest_run_id, key=f"ragas_{k}", score=None)
                                st.success("Logged placeholder RAGAS keys to LangSmith (customize parsing as needed).")
                        except Exception as e:
                            st.info(f"Feedback logging failed: {e}")

# ===========================
# Footer
# ===========================
st.markdown("---")
st.caption("GenAI-Tutor is educational. Verify critical info. Follow your organization’s policies.")
