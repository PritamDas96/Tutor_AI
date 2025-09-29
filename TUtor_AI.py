# TutorAI – Intelligent Conversational Learning Assistant (Open-Source + Hugging Face Chat)
# ----------------------------------------------------------------------------------------
# - Uses huggingface_hub.InferenceClient.chat_completion (conversational task)
# - Works with open-source instruction-tuned chat models (Llama, Mixtral, Gemma, Qwen)
# - Streamlit UI: chat + personalized study notes
# - LLM-agnostic: pick any HF chat model from the sidebar


import os
import time
import uuid
from typing import List, Dict, Any, Optional

import streamlit as st
from huggingface_hub import InferenceClient

# ----------------------------
# App Config & Title
# ----------------------------
st.set_page_config(page_title="TutorAI – Open-Source Tutor", layout="wide")
st.title("🎓 TutorAI – Intelligent Conversational Learning Assistant")

# ----------------------------
# Open-Source Chat Models (HF)
# ----------------------------
DEFAULT_MODELS = [
    # All of these are instruction-tuned, chat-optimized models
    "meta-llama/Meta-Llama-3-8B-Instruct",     # Meta license — accept on HF
    "mistralai/Mistral-7B-Instruct-v0.2",      # Apache-2.0
    "mistralai/Mixtral-8x7B-Instruct-v0.1",    # Apache-2.0 (MoE)
    "google/gemma-2-9b-it",                    # Gemma license — accept on HF
    "Qwen/Qwen2.5-7B-Instruct",                # Qwen 2.5 license
]

# Reasonable defaults
GEN_DEFAULTS = dict(
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
)

# ----------------------------
# Scenarios to demonstrate capability
# ----------------------------
SCENARIOS = {
    "Generative AI & Prompting for Productivity (Employees)": """You are TutorAI, a helpful corporate learning tutor.
Audience: Non-technical employees learning to safely and effectively use Generative AI at work.
Goals:
- Explain key concepts (tokens, prompts, temperature) in plain English.
- Teach safe/secure usage, data handling, and compliance awareness.
- Provide practical prompting patterns and examples for everyday tasks (summaries, emails, brainstorming, checklists).
Constraints:
- Be concise, friendly, and actionable.
- Avoid policy violations. When unsure, ask clarifying questions.
""",
    "Data Privacy & Security Basics (Employees)": """You are TutorAI, a cybersecurity awareness tutor.
Audience: All employees.
Goals:
- Teach good security hygiene: strong passwords, MFA, phishing recognition, data classification.
- Provide examples and short exercises.
- Emphasize real-world risks and best practices.
Style:
- Clear, non-technical language with checklists and do/don'ts.
""",
    "SQL Basics for Analysts (Beginners)": """You are TutorAI, a SQL tutor for beginners.
Audience: New data analysts learning SQL fundamentals.
Goals:
- Cover SELECT, WHERE, ORDER BY, GROUP BY, JOIN basics.
- Provide short examples with sample tables.
- Offer small practice exercises and solutions step-by-step.
Style:
- Encourage exploration, give helpful hints before full answers.
""",
}

# ----------------------------
# Sidebar Controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    selected_scenario = st.selectbox("Learning Scenario", list(SCENARIOS.keys()), index=0)

    model_id = st.selectbox(
        "Hugging Face Model (chat)",
        DEFAULT_MODELS,
        index=0,
        help="Open-source, instruction-tuned chat models. Accept licenses on HF if required."
    )

    temperature = st.slider("Temperature", 0.0, 1.5, GEN_DEFAULTS["temperature"], 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, GEN_DEFAULTS["top_p"], 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 2048, GEN_DEFAULTS["max_new_tokens"], 32)

    st.markdown("---")
    st.subheader("🔐 Hugging Face Token")
    pasted = st.text_input("HF token (starts with hf_…)", type="password")
    hf_token = st.secrets.get("HF_TOKEN") or pasted or os.environ.get("HF_TOKEN", "")
    if not hf_token:
        st.warning("Add your HF token here or in Streamlit Secrets as HF_TOKEN.")

    st.markdown("---")
    provider = st.selectbox(
        "Provider (advanced)",
        ["Auto", "hf-inference"],
        help="If you see provider-task mismatch errors, try 'hf-inference'."
    )
    provider = None if provider == "Auto" else provider

    st.caption("Tip: If a model throttles or errors, switch model or lower max_new_tokens.")

# ----------------------------
# Session State (Chat Memory)
# ----------------------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = [
        {"role": "system", "content": SCENARIOS[selected_scenario]},
        {"role": "assistant", "content": "Hello! I’m TutorAI. What would you like to learn today?"}
    ]

# If scenario changes mid-convo, update system msg
if st.session_state.messages and st.session_state.messages[0]["content"] != SCENARIOS[selected_scenario]:
    st.session_state.messages[0] = {"role": "system", "content": SCENARIOS[selected_scenario]}

# ----------------------------
# Helpers
# ----------------------------
def build_chat_messages_from_state() -> List[Dict[str, str]]:
    msgs = []
    for m in st.session_state.messages:
        if m["role"] in ("system", "user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs

def call_hf_chat(
    model: str,
    messages: List[Dict[str, str]],
    token: str,
    gen_cfg: Dict[str, Any],
    provider: Optional[str] = None,
) -> str:
    """
    Call Hugging Face Chat Completion API (conversational task).
    This avoids 'not supported for task text-generation' errors.
    """
    if not token:
        raise RuntimeError("Missing HF token.")
    client = InferenceClient(model=model, token=token, provider=provider)
    resp = client.chat_completion(
        messages=messages,
        max_tokens=int(gen_cfg.get("max_new_tokens", 512)),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.9)),
        # stream=False by default
    )
    choice = resp.choices[0]
    msg = getattr(choice, "message", None) or choice["message"]
    content = getattr(msg, "content", None) or msg["content"]
    return (content or "").strip()

def tutor_reply_after_user_turn() -> str:
    """
    Use current session-state messages (including latest user turn)
    to generate the assistant reply. Try selected model, then fallbacks.
    """
    messages = build_chat_messages_from_state()
    gen_cfg = dict(max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)

    # Try selected model, then others as fallback
    fallbacks = [model_id] + [m for m in DEFAULT_MODELS if m != model_id]
    errors = []
    for mid in fallbacks:
        try:
            return call_hf_chat(mid, messages, hf_token, gen_cfg, provider=provider)
        except Exception as e:
            errors.append(f"{mid}: {e}")
            time.sleep(0.3)
    raise RuntimeError("All model calls failed:\n- " + "\n- ".join(errors))

def build_notes_prompt(profile: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Build a mini-conversation to instruct the model to write a personalized study guide.
    We keep it conversational (system + user) so it fits chat-completions cleanly.
    """
    system_block = SCENARIOS[selected_scenario]
    user_block = f"""
Create a concise, personalized study guide for me based on this profile.
Keep it highly actionable, with examples, small exercises, and quick checks-for-understanding.
Prefer bullet points, tables, and short sections I can read in 5–10 minutes.

My Profile:
- Role: {profile.get('role','Employee')}
- Team/Domain: {profile.get('domain','General')}
- Current Level: {profile.get('level','Beginner')}
- Goals (top 3): {profile.get('goals','(not provided)')}
- Pain Points / Confusions: {profile.get('pain','(not provided)')}
- Preferred Learning Style: {profile.get('style','Concise & example-driven')}
- Time Available Per Day: {profile.get('time','15 minutes')}

Include:
1) Key Concepts (plain English, 1–2 lines each)
2) Practical Patterns / Templates (aligned to the chosen scenario)
3) 3–5 Micro-exercises (with solutions or hints)
4) Mini-Checklist (Do / Don’t)
5) A 5-day learning plan (15–20 min/day)
"""
    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block},
    ]

# ----------------------------
# Layout: Chat + Notes
# ----------------------------
chat_col, notes_col = st.columns([2, 1], gap="large")

with chat_col:
    st.subheader("💬 Conversational Tutor")

    # Display history
    for m in st.session_state.messages:
        if m["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(m["content"])
        elif m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])

    # User input
    user_prompt = st.chat_input("Ask a question, or say what you want to learn…")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("assistant"):
            try:
                reply = tutor_reply_after_user_turn()
            except Exception as e:
                reply = f"⚠️ Error: {e}"
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

with notes_col:
    st.subheader("📝 Generate Personalized Study Notes")
    with st.form("notes_form"):
        role = st.text_input("Your Role", value="People Operations Specialist")
        domain = st.text_input("Team/Domain", value="General")
        level = st.selectbox("Current Level", ["Beginner", "Intermediate", "Advanced"], index=0)
        goals = st.text_area("Your Top 3 Goals", value="Understand safe AI use; Write effective prompts; Automate routine tasks")
        pain = st.text_area("Pain Points / Confusions", value="Unsure how to structure prompts; Concerned about data security")
        style = st.selectbox("Preferred Learning Style", ["Concise & example-driven", "Step-by-step & detailed", "Visual & analogies"], index=0)
        time_per_day = st.text_input("Time Available / Day", value="15 minutes")
        submitted = st.form_submit_button("Generate Notes")

    if submitted:
        profile = {
            "role": role,
            "domain": domain,
            "level": level,
            "goals": goals,
            "pain": pain,
            "style": style,
            "time": time_per_day,
        }
        # Build a clean 2-turn chat just for notes
        notes_messages = build_notes_prompt(profile)
        with st.spinner("Drafting your personalized study guide…"):
            try:
                notes_text = call_hf_chat(model_id, notes_messages, hf_token, GEN_DEFAULTS, provider=provider)
            except Exception as e:
                notes_text = f"⚠️ Error while generating notes: {e}"
        st.markdown("---")
        st.markdown("#### 📚 Your Study Guide")
        st.write(notes_text)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "TutorAI is an educational assistant. It may produce mistakes—verify critical info. "
    "For sensitive or confidential topics, follow your organization’s policies."
)
