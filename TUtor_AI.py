# GenAI-Tutor – Intelligent Conversational Learning Assistant (Hugging Face Chat)
# ------------------------------------------------------------------------------
# UI to-spec:
# - Sidebar: ONLY 2 dropdowns (Learning Scenario, HF Model)
# - Main: Scenario Overview → Expander: Personalized Study Notes (with dropdowns + "Other") → Chat below
# - Notes: include citations/links to authoritative resources (curated per scenario)
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
# Gen-AI Learning Scenarios (content + system prompt)
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
# Curated Authoritative Resources (used in notes + rendered below)
# ----------------------------
RESOURCES: Dict[str, List[Dict[str, str]]] = {
    "Prompt Engineering Basics": [
        {"title": "Anthropic Prompt Engineering Guide", "url": "https://docs.anthropic.com/claude/docs/prompt-engineering"},
        {"title": "OpenAI Cookbook", "url": "https://cookbook.openai.com/"},
        {"title": "Microsoft Prompt Engineering (Learn)", "url": "https://learn.microsoft.com/azure/ai-services/openai/concepts/prompt-engineering"},
        {"title": "Google Prompting with Gemini", "url": "https://ai.google.dev/gemini-api/docs/prompting"},
    ],
    "Responsible & Secure GenAI at Work": [
        {"title": "NIST AI Risk Management Framework", "url": "https://www.nist.gov/itl/ai-risk-management-framework"},
        {"title": "OWASP Top 10 for LLM Applications", "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/"},
        {"title": "UK ICO – Guidance on AI & Data Protection", "url": "https://ico.org.uk/for-organisations/ai/"},
    ],
    "Automating Everyday Tasks with GenAI": [
        {"title": "OpenAI Cookbook – Patterns & Examples", "url": "https://cookbook.openai.com/"},
        {"title": "LangChain Docs – Prompt Templates", "url": "https://python.langchain.com/docs/concepts/prompt_templates"},
        {"title": "Google Prompting with Gemini", "url": "https://ai.google.dev/gemini-api/docs/prompting"},
    ],
    "Writing & Communication with GenAI": [
        {"title": "PlainLanguage.gov – Federal Plain Language Guidelines", "url": "https://www.plainlanguage.gov/"},
        {"title": "Nielsen Norman Group – Writing for the Web", "url": "https://www.nngroup.com/topic/writing-web/"},
    ],
    "Data Summarization & Analysis with GenAI": [
        {"title": "Nielsen Norman Group – Summarization Guidance", "url": "https://www.nngroup.com/articles/summarization/"},
        {"title": "Harvard Guide to Summarizing", "url": "https://writingcenter.fas.harvard.edu/pages/strategies-essay-writing"},
    ],
    "Evaluation & Guardrails Basics": [
        {"title": "NIST AI RMF (Risk Management)", "url": "https://www.nist.gov/itl/ai-risk-management-framework"},
        {"title": "OWASP Top 10 for LLM Applications", "url": "https://owasp.org/www-project-top-10-for-large-language-model-applications/"},
    ],
}

# ----------------------------
# Sidebar (ONLY two dropdowns)
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    # Token strictly from Secrets or env
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
# HF Call (chat-completion) with provider fallback
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
# Personalized Study Notes (Expander with dropdowns + "Other")
# ----------------------------
st.markdown("---")
with st.expander("📝 Personalized Study Notes", expanded=False):

    # ---- Dropdown options ----
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

    # ---- Compact layout ----
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        role_choice = st.selectbox("Role", ROLE_OPTIONS, index=0)
        role_other = ""
        if role_choice == "Other":
            role_other = st.text_input("Specify Role")
        level = st.selectbox("Current Level", ["Beginner", "Intermediate", "Advanced"], index=0)

    with col2:
        team_choice = st.selectbox("Team / Domain", TEAM_OPTIONS, index=0)
        team_other = ""
        if team_choice == "Other":
            team_other = st.text_input("Specify Team/Domain")
        time_per_day = st.text_input("Time Available / Day", value="15 minutes")

    with col3:
        style = st.selectbox("Preferred Style", ["Concise & example-driven", "Step-by-step", "Visual & analogies"], index=0)
        st.write("")  # spacer

    # Goals (multiselect with "Other")
    goals_selected = st.multiselect("Your Top 3 Goals", options=GOAL_OPTIONS,
                                    default=["Use Gen-AI safely & responsibly", "Write effective prompts", "Automate routine tasks"])
    goals_other_text = ""
    if "Other (type below)" in goals_selected:
        goals_other_text = st.text_input("Other goals (comma-separated)")

    # Pain points (multiselect with "Other")
    pain_selected = st.multiselect("Pain Points / Confusions", options=PAIN_OPTIONS,
                                   default=["Unclear prompt structure", "Fear of data leaks"])
    pain_other_text = ""
    if "Other (type below)" in pain_selected:
        pain_other_text = st.text_input("Other pain points (comma-separated)")

    # Enforce up to 3 goals
    if len([g for g in goals_selected if g != "Other (type below)"]) > 3:
        st.warning("Please select at most 3 preset goals (excluding 'Other'). Extra selections will be ignored.")

    # Build final strings
    def _finalize(value_choice: str, other: str) -> str:
        return other.strip() if value_choice == "Other" and other.strip() else value_choice

    role_val = _finalize(role_choice, role_other) or "General"
    team_val = _finalize(team_choice, team_other) or "General"

    def _merge_multiselect(base_list: List[str], other_text: str, max_keep: int = 3) -> str:
        fixed = [x for x in base_list if x != "Other (type below)"]
        fixed = fixed[:max_keep]  # cap
        other_items = [x.strip() for x in other_text.split(",") if x.strip()] if other_text else []
        merged = fixed + other_items
        # de-duplicate while preserving order
        seen, final = set(), []
        for x in merged:
            if x not in seen:
                final.append(x); seen.add(x)
        return ", ".join(final) if final else "(not provided)"

    goals_val = _merge_multiselect(goals_selected, goals_other_text, max_keep=3)
    pain_val = _merge_multiselect(pain_selected, pain_other_text, max_keep=5)

    # Action buttons
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        gen_notes = st.button("Generate Notes", use_container_width=True)
    with colB:
        insert_notes = st.button("Insert Notes into Chat", use_container_width=True, disabled=not bool(st.session_state.notes_text))
    with colC:
        clear_notes = st.button("Clear Notes", use_container_width=True, disabled=not bool(st.session_state.notes_text))

    # Notes generation
    if gen_notes:
        # Build system+user messages for notes; include authoritative resources IN the prompt
        scenario_system = SCENARIOS[scenario_name]["system"]

        # Prepare a list of curated links for the chosen scenario
        curated = RESOURCES.get(scenario_name, [])
        curated_bullets = "\n".join([f"- [{r['title']}]({r['url']})" for r in curated]) if curated else ""

        user_request = f"""
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
- Use **at least 3** of these **authoritative resources** exactly as listed (no invented URLs):
{curated_bullets if curated_bullets else "- (No curated links available)"}
"""

        notes_messages = [
            {"role": "system", "content": scenario_system},
            {"role": "user", "content": user_request},
        ]
        with st.spinner("Drafting your personalized study guide…"):
            try:
                notes_text = call_hf_chat(model_id, notes_messages, hf_token)
            except Exception as e:
                notes_text = f"⚠️ Error while generating notes: {e}"
        st.session_state.notes_text = notes_text

    if insert_notes and st.session_state.notes_text:
        # Insert notes into chat as extra system context
        context_blob = f"Reference study notes for future answers (scenario: {scenario_name}):\n\n{st.session_state.notes_text}"
        st.session_state.messages.append({"role": "system", "content": context_blob})
        st.success("Notes inserted into chat context.")

    if clear_notes and st.session_state.notes_text:
        st.session_state.notes_text = ""
        st.info("Notes cleared.")

    # Render notes and a visible resource block (guaranteed accurate links)
    if st.session_state.notes_text:
        st.markdown("#### 📚 Your Study Guide")
        st.write(st.session_state.notes_text)

        # Always render our curated resources below (even if model missed some)
        curated = RESOURCES.get(scenario_name, [])
        if curated:
            st.markdown("#### 🔗 Authoritative Resources")
            for r in curated:
                st.markdown(f"- [{r['title']}]({r['url']})")

        st.download_button(
            label="Download as .md",
            data=st.session_state.notes_text.encode("utf-8"),
            file_name=f"genai_study_guide_{scenario_name.replace(' ','_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.caption("Select your role/team, pick goals and pain points, then click **Generate Notes**.")

# ----------------------------
# Chatbot (below the notes)
# ----------------------------
st.markdown("---")
st.subheader("💬 Tutor Chat")

# Controls: Reset Chat
cc1, cc2 = st.columns([1, 4])
with cc1:
    if st.button("Reset Chat", use_container_width=True):
        _seed_chat()
        st.success("Chat reset.")

# Render chat history
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
