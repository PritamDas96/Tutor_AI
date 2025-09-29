# GenAI-Tutor – with RAG Add-on (Hugging Face Chat + FAISS + bge)
# ----------------------------------------------------------------
# - Sidebar: ONLY two dropdowns (Learning Scenario, HF Model)
# - Main: Scenario Overview → Notes (expander) → Chat (with optional RAG)
# - RAG: link-only ingestion → clean → chunk (~600 tokens, 80 overlap) → embed (bge-small)
#        → FAISS (cosine via IP on normalized vecs) → rerank (bge-reranker) → top_k=7
#
# Deploy on Streamlit Cloud:
# 1) requirements.txt -> see end of file
# 2) Secrets -> HF_TOKEN="hf_***"
# 3) Push & run

import os
import io
import time
import uuid
import math
import hashlib
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from bs4 import BeautifulSoup
from pypdf import PdfReader

# ----------------------------
# App Config & Title
# ----------------------------
st.set_page_config(page_title="GenAI-Tutor", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor – Intelligent Conversational Learning Assistant</h1>", unsafe_allow_html=True)

# ----------------------------
# Open-Source Chat Models (HF)
# ----------------------------
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",     # Meta license — accept on HF
    "mistralai/Mistral-7B-Instruct-v0.2",      # Apache-2.0
    "mistralai/Mixtral-8x7B-Instruct-v0.1",    # Apache-2.0 (MoE)
    "google/gemma-2-9b-it",                    # Gemma license — accept on HF
    "Qwen/Qwen2.5-7B-Instruct",                # Qwen 2.5 license
]

# ----------------------------
# Gen-AI Learning Scenarios
# ----------------------------
SCENARIOS: Dict[str, Dict[str, str]] = {
    "Prompt Engineering Basics": {
        "overview": """- Core prompting concepts (role, task, context, constraints)
- Patterns: Few-shot, Chain-of-Thought (high level), ReAct (high level), Style/Format guides
- Practical templates for summaries, emails, brainstorming, checklists
- Tips to reduce hallucinations (be specific, ask for sources, step-by-step)""",
        "system": """You are GenAI-Tutor, an expert coach on prompt engineering for employees.
Explain concepts in plain English, be concise and actionable. Provide examples and mini-exercises.
When uncertain, ask clarifying questions. Avoid unsafe instructions and protect sensitive data."""
    },
    "Responsible & Secure GenAI at Work": {
        "overview": """- Safe inputs (no confidential/PII), data classification, minimal data principle
- Policy-aligned usage, review & approval paths
- Phishing awareness & social engineering risks
- Quick checklists and red-flag examples""",
        "system": """You are GenAI-Tutor focused on responsible, secure GenAI usage at work.
Teach best practices for data handling, compliance awareness, and risk spotting.
Be practical, checklist-driven, and give brief real-world examples."""
    },
    "Automating Everyday Tasks with GenAI": {
        "overview": """- Drafting: emails, meeting notes, SOPs, briefs
- Idea generation & prioritization frameworks
- Converting raw notes → structured outputs (tables, action items)
- Time-saving workflows and quick macros/prompts""",
        "system": """You are GenAI-Tutor specialized in everyday task automation.
Offer templates and mini-workflows for drafting, organizing, and prioritizing.
Optimize for speed and clarity. Encourage iteration and verification."""
    },
    "Writing & Communication with GenAI": {
        "overview": """- Style targeting (tone, audience, reading level)
- Rewrite/expand/condense with structure and clarity
- Persuasive & empathetic communication patterns
- Review checklists for grammar, bias, inclusivity""",
        "system": """You are GenAI-Tutor for business writing with Gen-AI.
Provide tone-adapted examples, structure-first rewrites, and succinct review checklists.
Focus on clarity, inclusivity, and audience fit."""
    },
    "Data Summarization & Analysis with GenAI": {
        "overview": """- Summarize long text into bullets, key insights, action items
- Compare/contrast views, pros/cons matrices
- Extract entities, dates, owners, deadlines
- Guard against misreads; ask for missing context""",
        "system": """You are GenAI-Tutor for summarization & light analysis.
Teach concise summarization patterns, extraction prompts, and validation steps.
Highlight assumptions and ask for missing inputs when needed."""
    },
    "Evaluation & Guardrails Basics": {
        "overview": """- Simple quality rubrics (correctness, completeness, clarity)
- Self-critique prompts; instruction fidelity checks
- Basic guardrails: refuse unsafe requests, ask for clarifications
- Lightweight eval loops for iterative improvement""",
        "system": """You are GenAI-Tutor for LLM evaluation & guardrails.
Provide small rubrics, self-check prompts, and refusal/clarification strategies.
Keep it pragmatic and safe-by-default."""
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# ----------------------------
# Sidebar (ONLY two dropdowns)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    st.caption("HF token is loaded from Secrets / env.")
st.caption(f"Model in use: **{model_id}**  •  Scenario: **{scenario_name}**")

if not hf_token:
    st.error("Missing HF token. Add `HF_TOKEN` in Streamlit Secrets or environment.")
    st.stop()

# ----------------------------
# Session State
# ----------------------------
if "scenario_prev" not in st.session_state:
    st.session_state.scenario_prev = scenario_name

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

def _seed_chat():
    st.session_state.messages = [
        {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
        {"role": "assistant", "content": "Hello! I’m GenAI-Tutor. What would you like to learn today?"}
    ]

if not st.session_state.messages or st.session_state.scenario_prev != scenario_name:
    _seed_chat()
    st.session_state.scenario_prev = scenario_name

# ----------------------------
# HF Chat Completion
# ----------------------------
def call_hf_chat(model: str,
                 messages: List[Dict[str, str]],
                 token: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9) -> str:
    """Hugging Face chat-completion with provider fallback."""
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
#                        RAG  SECTION
# ============================================================

TOP_K = 7               # final retrieved chunks to inject
K_CANDIDATES = 30       # initial ANN candidates before rerank
TARGET_WORDS = 450      # ~600 tokens rough
OVERLAP_WORDS = 80

# ---- Your curated links (7 + 4 RAG papers) ----
DOC_LINKS = [
    # 7 GenAI-in-education anchors
    {"title": "Ethical and Regulatory Challenges of Generative AI in Education (2025)",
     "url": "https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1565938/full"},
    {"title": "Learn Your Way: Reimagining Textbooks with Generative AI — Google Research (2025)",
     "url": "https://research.google/blog/learn-your-way-reimagining-textbooks-with-generative-ai/"},
    {"title": "Student Generative AI Survey 2025 – HEPI",
     "url": "https://www.hepi.ac.uk/reports/student-generative-ai-survey-2025/"},
    {"title": "Generative AI in Higher Education: A Global Perspective (2025)",
     "url": "https://www.sciencedirect.com/science/article/pii/S2666920X24001516"},
    {"title": "Educational Impacts of Generative AI on Learning and Performance (2025)",
     "url": "https://www.nature.com/articles/s41598-025-06930-w"},
    {"title": "Google AI Blog – Adaptive Learning & GenAI (collection)",
     "url": "https://ai.googleblog.com/"},
    {"title": "Frontiers – Artificial Intelligence in Education (collection)",
     "url": "https://www.frontiersin.org/journals/education/sections/artificial-intelligence-in-education"},

    # 4 RAG methodology papers
    {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)",
     "url": "https://arxiv.org/abs/2005.11401"},
    {"title": "Enhancing Retrieval-Augmented Generation: A Study of Best Practices (Li et al., 2025)",
     "url": "https://arxiv.org/abs/2501.01234"},  # placeholder as supplied
    {"title": "Retrieval-Augmented Language Model Pre-Training (Lewis et al.)",
     "url": "https://arxiv.org/abs/2112.09118"},
    {"title": "RAG: A Simple Baseline for Generative QA (Min et al., 2021)",
     "url": "https://arxiv.org/abs/2105.11418"},
]

# ----------------------------
# Utilities: fetch & clean
# ----------------------------
@st.cache_data(show_spinner=False)
def _download(url: str, timeout: int = 30) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    content_type = r.headers.get("Content-Type", "")
    return r.content, content_type.lower()

def _clean_html(html_bytes: bytes) -> str:
    text = ""
    try:
        # light fallback extractor using BeautifulSoup
        soup = BeautifulSoup(html_bytes, "html.parser")
        # Remove scripts/styles/nav
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            t.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = html_bytes.decode("utf-8", errors="ignore")
    # normalize whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def _clean_pdf(pdf_bytes: bytes) -> str:
    text = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in reader.pages:
        try:
            text.append(page.extract_text() or "")
        except Exception:
            text.append("")
    joined = "\n".join(text)
    lines = [ln.strip() for ln in joined.splitlines()]
    return "\n".join([ln for ln in lines if ln])

def fetch_and_clean(url: str) -> str:
    blob, ctype = _download(url)
    if ".pdf" in url.lower() or "application/pdf" in ctype:
        return _clean_pdf(blob)
    return _clean_html(blob)

# ----------------------------
# Chunking (~600 tokens ≈ ~450 words) with 80-word overlap
# ----------------------------
def _to_words(text: str) -> List[str]:
    # split on whitespace, keep non-empty
    return [w for w in text.replace("\u00a0", " ").split() if w]

def chunk_text(text: str, url: str, title: str,
               target_words: int = TARGET_WORDS,
               overlap_words: int = OVERLAP_WORDS) -> List[Dict[str, Any]]:
    if not text:
        return []
    words = _to_words(text)
    chunks = []
    start = 0
    k = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        piece = " ".join(words[start:end])
        chunk_id = f"{hashlib.sha1(url.encode()).hexdigest()}#{k:04d}"
        chunks.append({
            "chunk_id": chunk_id,
            "title": title,
            "url": url,
            "text": piece
        })
        if end == len(words):
            break
        start = max(0, end - overlap_words)
        k += 1
    return chunks

# ----------------------------
# Embeddings & Reranker (cached)
# ----------------------------
@st.cache_resource(show_spinner=True)
def load_embedder() -> SentenceTransformer:
    # 384-d, good speed/quality
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=True)
def load_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-v2-m3")

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    # cosine via normalized inner product
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")

# ----------------------------
# FAISS index (cosine via IP on normalized vecs)
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    import faiss  # lazy import
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    return index

# ----------------------------
# Build (or refresh) the RAG corpus & index
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, str]]):
    embedder = load_embedder()
    all_chunks: List[Dict[str, Any]] = []
    for doc in doc_links:
        try:
            raw = fetch_and_clean(doc["url"])
            chunks = chunk_text(raw, doc["url"], doc["title"])
            all_chunks.extend(chunks)
        except Exception as e:
            # Skip bad links but keep going
            st.warning(f"Failed to ingest {doc['url']}: {e}")

    if not all_chunks:
        raise RuntimeError("No chunks ingested. Check your links or network.")

    vectors = embed_texts([c["text"] for c in all_chunks], embedder)
    index = build_faiss(vectors)

    # side data needed by search
    side = {
        "chunks": all_chunks,
        "vectors_shape": vectors.shape
    }
    return index, side

def refresh_rag_cache():
    # Clears only RAG caches; leaves model/chat caches intact.
    st.cache_resource.clear()
    st.cache_data.clear()

# ----------------------------
# Retrieval (ANN → rerank → top_k)
# ----------------------------
def retrieve(query: str,
             index,
             side: Dict[str, Any],
             top_k: int = TOP_K,
             k_candidates: int = K_CANDIDATES) -> List[Dict[str, Any]]:
    import faiss
    embedder = load_embedder()
    reranker = load_reranker()

    qv = embed_texts([query], embedder)  # 1 x d (normalized)
    scores, idx = index.search(qv, k_candidates)  # inner product
    cand_ids = idx[0].tolist()
    cand_scores = scores[0].tolist()

    candidates = []
    for pos, (ci, s) in enumerate(zip(cand_ids, cand_scores)):
        if ci < 0:
            continue
        c = side["chunks"][ci]
        candidates.append({
            "rank_ann": pos + 1,
            "score_ann": float(s),
            **c
        })

    # Cross-encoder rerank
    pairs = [(query, c["text"]) for c in candidates]
    if pairs:
        rerank_scores = reranker.predict(pairs, batch_size=64).tolist()
        for c, rs in zip(candidates, rerank_scores):
            c["score_rerank"] = float(rs)
        candidates.sort(key=lambda x: x["score_rerank"], reverse=True)
        return candidates[:top_k]
    else:
        return []

# ----------------------------
# Build grounded context & citations
# ----------------------------
def build_context_and_citations(retrieved: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict[str, Any]]]:
    # Map urls to numeric refs
    url_to_ref: Dict[str, int] = {}
    refs: List[str] = []
    context_blocks: List[str] = []

    for c in retrieved:
        url = c["url"]
        if url not in url_to_ref:
            url_to_ref[url] = len(url_to_ref) + 1
            refs.append(f"[{url_to_ref[url]}] {url} — {c['title']}")
        ref_num = url_to_ref[url]
        snippet = c["text"].strip()
        # keep a short snippet (first ~800 characters)
        snippet = (snippet[:800] + "…") if len(snippet) > 800 else snippet
        context_blocks.append(f"[{ref_num}] {c['title']}\n{snippet}\n")

    context_text = "\n\n".join(context_blocks)
    sources_list = "\n".join(refs)
    return context_text, sources_list, retrieved

def rag_guardrails_instructions() -> str:
    return (
        "Use ONLY the provided CONTEXT to answer. "
        "Cite sources inline like [1], [2] after each claim that uses evidence. "
        "If the context does not contain enough information, say so and suggest which source to consult. "
        "Do NOT invent URLs. Provide a 'Sources' section listing the [n] → URL mapping at the end."
    )

# ============================================================
#                    UI – Overview & Notes
# ============================================================

st.subheader("📌 Scenario Overview")
st.markdown(f"**{scenario_name}**  \n{SCENARIOS[scenario_name]['overview']}")

st.markdown("---")
with st.expander("📝 Personalized Study Notes", expanded=False):

    # Dropdowns for Role/Team + Other fields, Goals/Pain multiselects with Other
    ROLE_OPTIONS = [
        "General", "Manager", "Analyst", "Engineer/Developer", "HR/People",
        "Sales", "Marketing", "Operations", "Finance", "Customer Support",
        "Legal/Compliance", "Data/Analytics", "Other"
    ]
    TEAM_OPTIONS = [
        "General", "HR", "Finance", "Marketing", "Sales", "IT/Engineering",
        "Operations", "Legal/Compliance", "Customer Support", "Data/Analytics", "Other"
    ]
    GOAL_OPTIONS = [
        "Use Gen-AI safely & responsibly",
        "Write effective prompts",
        "Automate routine tasks",
        "Improve business writing",
        "Summarize long content",
        "Analyze/compare information",
        "Build evaluation & guardrails",
        "Other (type below)",
    ]
    PAIN_OPTIONS = [
        "Unclear prompt structure",
        "Fear of data leaks",
        "Hallucinations/accuracy issues",
        "Hard to control tone/style",
        "Information overload",
        "Tool overwhelm / where to start",
        "Other (type below)",
    ]

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        role_choice = st.selectbox("Role", ROLE_OPTIONS, index=0)
        role_other = st.text_input("Specify Role", value="") if role_choice == "Other" else ""
        level = st.selectbox("Current Level", ["Beginner", "Intermediate", "Advanced"], index=0)
    with col2:
        team_choice = st.selectbox("Team / Domain", TEAM_OPTIONS, index=0)
        team_other = st.text_input("Specify Team/Domain", value="") if team_choice == "Other" else ""
        time_per_day = st.text_input("Time Available / Day", value="15 minutes")
    with col3:
        style = st.selectbox("Preferred Style", ["Concise & example-driven", "Step-by-step", "Visual & analogies"], index=0)
        st.write("")

    goals_selected = st.multiselect("Your Top 3 Goals", options=GOAL_OPTIONS,
                                    default=["Use Gen-AI safely & responsibly", "Write effective prompts", "Automate routine tasks"])
    goals_other_text = st.text_input("Other goals (comma-separated)") if "Other (type below)" in goals_selected else ""

    pain_selected = st.multiselect("Pain Points / Confusions", options=PAIN_OPTIONS,
                                   default=["Unclear prompt structure", "Fear of data leaks"])
    pain_other_text = st.text_input("Other pain points (comma-separated)") if "Other (type below)" in pain_selected else ""

    def _finalize(choice: str, other: str) -> str:
        return other.strip() if (choice == "Other" and other.strip()) else choice

    role_val = _finalize(role_choice, role_other) or "General"
    team_val = _finalize(team_choice, team_other) or "General"

    def _merge_multiselect(base_list: List[str], other_text: str, max_keep: int = 3) -> str:
        fixed = [x for x in base_list if x != "Other (type below)"][:max_keep]
        more = [x.strip() for x in (other_text or "").split(",") if x.strip()]
        # dedupe preserve order
        seen, out = set(), []
        for x in fixed + more:
            if x not in seen:
                out.append(x); seen.add(x)
        return ", ".join(out) if out else "(not provided)"

    goals_val = _merge_multiselect(goals_selected, goals_other_text, max_keep=3)
    pain_val = _merge_multiselect(pain_selected, pain_other_text, max_keep=5)

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        gen_notes = st.button("Generate Notes", use_container_width=True)
    with colB:
        insert_notes = st.button("Insert Notes into Chat", use_container_width=True, disabled=not bool(st.session_state.notes_text))
    with colC:
        clear_notes = st.button("Clear Notes", use_container_width=True, disabled=not bool(st.session_state.notes_text))

    if gen_notes:
        curated = [
            # You can optionally pass curated links to the notes prompt; reusing earlier logic is fine.
            "- Anthropic Prompt Engineering Guide: https://docs.anthropic.com/claude/docs/prompt-engineering",
            "- OpenAI Cookbook: https://cookbook.openai.com/",
            "- Microsoft Prompt Engineering: https://learn.microsoft.com/azure/ai-services/openai/concepts/prompt-engineering",
            "- Google Prompting with Gemini: https://ai.google.dev/gemini-api/docs/prompting",
        ]
        user_req = f"""
Create a concise, personalized study guide on **{scenario_name}** for the profile below.
Keep it actionable with examples, mini-exercises, and quick checks. Prefer bullets/tables.

Profile:
- Role: {role_val}
- Team/Domain: {team_val}
- Level: {level}
- Goals (top 3): {goals_val}
- Pain Points: {pain_val}
- Preferred Style: {style}
- Time per day: {time_per_day}

Include:
1) Key Concepts (1–2 lines each)
2) Practical Patterns / Templates (aligned to the scenario)
3) 3–5 Micro-exercises (with solutions or hints)
4) Mini-Checklist (Do / Don’t)
5) 5-day learning plan (15–20 min/day)

Important:
- Add a **Sources** section at the end with **clickable markdown links**.
- Use at least 3 of these **authoritative resources** (no invented URLs):
{chr(10).join(curated)}
"""
        notes_messages = [
            {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
            {"role": "user", "content": user_req},
        ]
        with st.spinner("Drafting your personalized study guide…"):
            try:
                st.session_state.notes_text = call_hf_chat(model_id, notes_messages, hf_token)
            except Exception as e:
                st.session_state.notes_text = f"⚠️ Error while generating notes: {e}"

    if insert_notes and st.session_state.notes_text:
        context_blob = f"Reference study notes for future answers (scenario: {scenario_name}):\n\n{st.session_state.notes_text}"
        st.session_state.messages.append({"role": "system", "content": context_blob})
        st.success("Notes inserted into chat context.")

    if clear_notes and st.session_state.notes_text:
        st.session_state.notes_text = ""
        st.info("Notes cleared.")

    if st.session_state.notes_text:
        st.markdown("#### 📚 Your Study Guide")
        st.write(st.session_state.notes_text)

# ============================================================
#                    UI – RAG Controls
# ============================================================

st.markdown("---")
st.subheader("🔎 RAG (Retrieval-Augmented Generation)")

rag_col1, rag_col2, rag_col3 = st.columns([1, 1, 2])
with rag_col1:
    use_rag = st.checkbox("Use RAG for Chat", value=True, help="Ground answers in retrieved evidence with citations.")
with rag_col2:
    do_refresh = st.button("Refresh RAG Corpus", help="Rebuild index from the online links (clears caches).")
with rag_col3:
    st.caption("Sources: curated online research links (no file storage).")

if do_refresh:
    refresh_rag_cache()
    st.success("RAG caches cleared. The index will rebuild on next use.")

# Lazy build of RAG index (only when enabled or evidence needed)
def get_rag_index():
    try:
        index, side = build_rag_index(DOC_LINKS)
        return index, side
    except Exception as e:
        st.error(f"RAG index build failed: {e}")
        return None, None

# ============================================================
#                         CHATBOT
# ============================================================

st.markdown("---")
st.subheader("💬 Tutor Chat")

# Reset chat
cc1, cc2 = st.columns([1, 4])
with cc1:
    if st.button("Reset Chat", use_container_width=True):
        _seed_chat()
        st.success("Chat reset.")

# Render history
for m in st.session_state.messages:
    if m["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(m["content"])
    elif m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])

# Chat input
user_prompt = st.chat_input("Ask anything about this Gen-AI learning scenario…")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Build messages for this turn
    messages_for_call = list(st.session_state.messages)

    evidence_to_show = []
    if use_rag:
        index, side = get_rag_index()
        if index is not None:
            with st.spinner("Retrieving evidence…"):
                retrieved = retrieve(user_prompt, index, side, top_k=TOP_K, k_candidates=K_CANDIDATES)
            if retrieved:
                context_text, sources_list, evidence_to_show = build_context_and_citations(retrieved)
                # Insert a RAG system message ONLY for this call
                rag_rules = rag_guardrails_instructions()
                rag_system = {
                    "role": "system",
                    "content": f"CONTEXT (evidence snippets):\n\n{context_text}\n\nSOURCE LIST:\n{sources_list}\n\n{rag_rules}"
                }
                # Place right after the scenario system
                # messages_for_call[0] is scenario system
                messages_for_call = [messages_for_call[0], rag_system] + messages_for_call[1:]
            else:
                st.info("No evidence retrieved; answering without RAG context.")

    with st.chat_message("assistant"):
        try:
            reply = call_hf_chat(model_id, messages_for_call, hf_token)
        except Exception as e:
            reply = f"⚠️ Error: {e}"
        st.markdown(reply)

        # Evidence viewer
        if use_rag and evidence_to_show:
            with st.expander("🔗 Evidence (top 7)"):
                for i, c in enumerate(evidence_to_show, start=1):
                    preview = c["text"][:280] + ("…" if len(c["text"]) > 280 else "")
                    st.markdown(f"**[{i}] [{c['title']}]({c['url']})**")
                    st.write(preview)

    st.session_state.messages.append({"role": "assistant", "content": reply})

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "GenAI-Tutor provides educational assistance. Verify critical info. "
    "Follow your organization’s security and compliance policies."
)

