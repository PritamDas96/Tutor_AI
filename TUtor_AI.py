# GenAI-Tutor – Intelligent Conversational Learning Assistant (Hugging Face Chat)
# ------------------------------------------------------------------------------
# Blueprint-aligned UI:
# - Sidebar: ONLY two dropdowns (Learning Scenario, HF Model)
# - Main: Scenario Overview (card) → Expander: Personalized Study Notes → Chatbot below
# - Open-source LLMs via huggingface_hub.InferenceClient.chat_completion
#
# Deploy on Streamlit Cloud:
# 1) requirements.txt -> streamlit, huggingface_hub>=0.24
# 2) Secrets -> HF_TOKEN="hf_***"
# 3) Push & run

import os
import time
import uuid
from typing import List, Dict, Any, Optional

import streamlit as st
from huggingface_hub import InferenceClient

# ----------------------------
# App Config & Header
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

GEN_DEFAULTS = dict(max_new_tokens=512, temperature=0.7, top_p=0.9)

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
    # Token strictly comes from Secrets or env (no extra UI)
    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
    st.caption("HF token is loaded from Secrets / env.")

# Model badge under title
st.caption(f"Model in use: **{model_id}**  •  Scenario: **{scenario_name}**")

# Warn if no token
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

# Seed or reseed on first load / scenario change
if not st.session_state.messages or st.session_state.scenario_prev != scenario_name:
    _seed_chat()
    st.session_state.scenario_prev = scenario_name

# ----------------------------
# HF Call (chat-completion) with robust fallback provider
# ----------------------------
def call_hf_chat(model: str,
                 messages: List[Dict[str, str]],
                 token: str,
                 max_new_tokens: int = GEN_DEFAULTS["max_new_tokens"],
                 temperature: float = GEN_DEFAULTS["temperature"],
                 top_p: float = GEN_DEFAULTS["top_p"]) -> str:
    """
    Try default routing first; if provider-task issues occur, fall back to provider='hf-inference'.
    """
    # Attempt 1: auto provider
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

# ----------------------------
# Scenario Overview Card
# ----------------------------
st.subheader("📌 Scenario Overview")
with st.container():
    st.markdown(
        f"""
**{scenario_name}**  
{SCENARIOS[scenario_name]["overview"]}
""".strip()
    )

# ----------------------------
# Personalized Study Notes (Expander)
# ----------------------------
st.markdown("---")
with st.expander("📝 Personalized Study Notes", expanded=False):
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        role = st.text_input("Your Role", value="People Operations Specialist")
        level = st.selectbox("Current Level", ["Beginner", "Intermediate", "Advanced"], index=0)
    with col2:
        time_per_day = st.text_input("Time Available / Day", value="15 minutes")
        style = st.selectbox("Preferred Style", ["Concise & example-driven", "Step-by-step", "Visual & analogies"], index=0)
    with col3:
        domain = st.text_input("Team/Domain", value="General")
        _spacer = st.empty()

    goals = st.text_area("Your Top 3 Goals (comma-separated)", value="Use AI safely; Write effective prompts; Automate routine tasks")
    pain = st.text_area("Pain Points / Confusions (optional)", value="Unsure how to structure prompts; Concerned about data security")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        gen_notes = st.button("Generate Notes", use_container_width=True)
    with colB:
        insert_notes = st.button("Insert Notes into Chat", use_container_width=True, disabled=not bool(st.session_state.notes_text))
    with colC:
        clear_notes = st.button("Clear Notes", use_container_width=True, disabled=not bool(st.session_state.notes_text))

    if gen_notes:
        profile = {
            "role": role, "level": level, "time": time_per_day, "style": style,
            "domain": domain, "goals": goals, "pain": pain
        }
        # Build a small chat for notes generation (scenario-specific system + user request)
        notes_messages = [
            {"role": "system", "content": SCENARIOS[scenario_name]["system"]},
            {"role": "user", "content": f"""
Create a concise, personalized study guide on **{scenario_name}** for the profile below.
Keep it actionable with examples, mini-exercises, and quick checks. Prefer bullets/tables.

Profile:
- Role: {profile['role']}
- Team/Domain: {profile['domain']}
- Level: {profile['level']}
- Goals: {profile['goals']}
- Pain Points: {profile['pain']}
- Preferred Style: {profile['style']}
- Time per day: {profile['time']}

Include:
1) Key Concepts (1–2 lines each)
2) Practical Patterns / Templates
3) 3–5 Micro-exercises (with solutions or hints)
4) Mini-Checklist (Do / Don’t)
5) 5-day learning plan (15–20 min/day)
"""}
        ]
        with st.spinner("Drafting your personalized study guide…"):
            try:
                notes_text = call_hf_chat(model_id, notes_messages, hf_token)
            except Exception as e:
                notes_text = f"⚠️ Error while generating notes: {e}"
        st.session_state.notes_text = notes_text

    if insert_notes and st.session_state.notes_text:
        # Insert notes into chat as additional system context
        context_blob = f"Reference study notes for future answers (scenario: {scenario_name}):\n\n{st.session_state.notes_text}"
        st.session_state.messages.append({"role": "system", "content": context_blob})
        st.success("Notes inserted into chat context.")

    if clear_notes and st.session_state.notes_text:
        st.session_state.notes_text = ""
        st.info("Notes cleared.")

    # Render notes (if any) and download button
    if st.session_state.notes_text:
        st.markdown("#### 📚 Your Study Guide")
        st.write(st.session_state.notes_text)
        st.download_button(
            label="Download as .md",
            data=st.session_state.notes_text.encode("utf-8"),
            file_name=f"genai_study_guide_{scenario_name.replace(' ','_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.caption("Fill the profile and click **Generate Notes** to create your personalized study guide.")

# ----------------------------
# Chatbot (below the notes)
# ----------------------------
st.markdown("---")
st.subheader("💬 Tutor Chat")

# Controls row: Reset Chat
cc1, cc2 = st.columns([1, 4])
with cc1:
    if st.button("Reset Chat", use_container_width=True):
        _seed_chat()
        st.success("Chat reset.")

# Show chat history
for m in st.session_state.messages:
    if m["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(m["content"])
    elif m["role"] == "user":
        with st.chat_message("user"):
            st.markdown(m["content"])

# Chat input (send below notes)
user_prompt = st.chat_input("Ask anything about this Gen-AI learning scenario…")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("assistant"):
        try:
            reply = call_hf_chat(model_id, st.session_state.messages, hf_token)
        except Exception as e:
            reply = f"⚠️ Error: {e}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "GenAI-Tutor provides educational assistance. Verify critical info. "
    "Follow your organization’s security and compliance policies."
)
