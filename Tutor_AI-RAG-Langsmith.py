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
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
 

# -------- LangChain (judge LLM + embeddings for RAGAS) --------lc_emb = LcHfEmbeddings(model_name="BAAI/bge-small-en-v1.5")
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings as LcHfEmbeddings  # <-- embeddings impl

# ===========================
# App Config & Secrets
# ===========================
st.set_page_config(page_title="GenAI-Tutor | RAG + Observability", layout="wide", initial_sidebar_state="expanded")
from ui import inject_css, hero, status_bar, footer
inject_css()
hero("Version 3 : RAG + Observability Edition")

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN in Streamlit Secrets.")
    st.stop()
# Ensure downstream libs see the token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", HF_TOKEN)

os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
LS_PROJECT = st.secrets.get("LANGSMITH_PROJECT", os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-RAG"))

# Only enable LangSmith tracing when an API key is actually present; otherwise the
# explicit tracing_context()/trace() blocks force uploads that 401-spam the logs.
_LS_ENABLED = bool(os.environ.get("LANGSMITH_API_KEY", "").strip())
if _LS_ENABLED:
    os.environ["LANGSMITH_TRACING"] = str(st.secrets.get("LANGSMITH_TRACING", True)).lower()
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    import types
    from contextlib import contextmanager
    @contextmanager
    def _ls_noop(*args, **kwargs):
        yield types.SimpleNamespace(outputs=None, inputs=None,
                                    id="00000000-0000-0000-0000-000000000000")
    tracing_context = _ls_noop  # shadow the imported names with no-ops
    trace = _ls_noop
ls_client = Client()

# ===========================
# HF Chat Models (open-source)
# ===========================
HF_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "deepseek-ai/DeepSeek-V3-0324",
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
    st.header("Configuration")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    st.caption("HF token & LangSmith settings come from Secrets.")
    with st.expander("About this version"):
        st.write(
            "Version 3 (RAG + Observability). The RAG tutor plus production tooling: "
            "LangSmith tracing of every turn, thumbs up/down feedback, and a RAGAS "
            "panel scoring faithfulness and answer relevancy."
        )

# Status bar under the header
_rag_on = st.session_state.get("use_rag", True)
status_bar([
    (f"Model: <b>{model_id.split('/')[-1]}</b>", ""),
    (f"Scenario: <b>{scenario_name}</b>", ""),
    (("RAG: On", "mode") if _rag_on else ("RAG: Off", "")),
    (("LangSmith: On", "ok") if _LS_ENABLED else ("LangSmith: Off", "")),
])

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
#                    RAG CORE (shared module: rag_core.py)
# ============================================================
from rag_core import (
    TOP_K, K_CANDIDATES, DOC_LINKS,
    build_rag_index, refresh_rag_cache,
    build_context_and_citations, rag_rules,
    retrieve as _core_retrieve,
)

@traceable(run_type="retriever", name="retrieve", metadata={"top_k": TOP_K})
def retrieve(query, index, side, top_k=TOP_K, k_candidates=K_CANDIDATES):
    return _core_retrieve(query, index, side, top_k=top_k, k_candidates=k_candidates)

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
                            retrieved, _ = retrieve(profile_query, index, side, top_k=TOP_K, k_candidates=K_CANDIDATES)
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
    st.caption("Uses a curated corpus; in-memory index; top-k=7 with reranking.")

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
                        retrieved, _ = retrieve(user_prompt, index, side, top_k=TOP_K, k_candidates=K_CANDIDATES)
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
    st.write("Runs **faithfulness** and **answer relevancy** on the last N chats of this session. Optionally logs aggregate metrics to LangSmith on the latest run.")
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

                # --- HF Judge (LangChain) ---
                try:
                    endpoint = HuggingFaceEndpoint(
                        repo_id="meta-llama/Llama-3.1-8B-Instruct",
                        task="conversational",
                        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                        max_new_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )
                    lc_chat = ChatHuggingFace(llm=endpoint)   # pass endpoint as llm=
                    judge = LangchainLLMWrapper(lc_chat)
                except Exception as e:
                    st.error(f"HF judge failed: {e}")
                    judge = None

                # --- HF Embeddings for RAGAS (prevents OpenAI fallback) ---
                
                @st.cache_resource(show_spinner=False)
                def get_ragas_embeddings():
                    return LcHfEmbeddings(model_name="BAAI/bge-small-en-v1.5")
                hf_emb = get_ragas_embeddings()

                if judge is None:
                    st.error("HF judge did not initialize. Check HF token / model access.")
                    st.stop()
                else:
                    with st.spinner("Scoring with RAGAS…"):
                        scores = evaluate(
                            dataset=ds,
                            metrics=[Faithfulness(), AnswerRelevancy()],
                            llm=judge,            # HF judge (no OpenAI)
                            embeddings=hf_emb,    # HF embeddings (no OpenAI)
                            show_progress=True,
                        )
                    # Persist results so they survive the rerun triggered by the log button.
                    st.session_state["ragas_scores"] = scores
                    st.session_state["ragas_latest_run_id"] = turns[-1]["run_id"] if turns else ""

    # Render last RAGAS results + logging button at TOP LEVEL (not nested inside the
    # "Run RAGAS now" button block — a nested button never fires in Streamlit).
    if st.session_state.get("ragas_scores") is not None:
        st.subheader("📈 RAGAS Results")
        try:
            st.write(st.session_state["ragas_scores"])
        except Exception:
            st.json(st.session_state["ragas_scores"])

        if st.button("Log RAGAS metrics to LangSmith (latest run)"):
            try:
                rid = st.session_state.get("ragas_latest_run_id") or None
                if not rid:
                    st.info("No run_id to attach feedback.")
                else:
                    scores_obj = st.session_state["ragas_scores"]
                    logged = False
                    # Prefer real aggregate means if the result exposes a dataframe.
                    try:
                        df = scores_obj.to_pandas()
                        for k in ["faithfulness", "answer_relevancy"]:
                            if k in df.columns:
                                ls_client.create_feedback(run_id=rid, key=f"ragas_{k}", score=float(df[k].mean()))
                                logged = True
                    except Exception:
                        pass
                    if not logged:
                        for k in ["faithfulness", "answer_relevancy"]:
                            ls_client.create_feedback(run_id=rid, key=f"ragas_{k}", score=None)
                    st.success("Logged RAGAS metrics to LangSmith.")
            except Exception as e:
                st.info(f"Feedback logging failed: {e}")

# ===========================
# Footer
# ===========================
footer("GenAI-Tutor is educational. Verify critical information and follow your organization's policies.")
