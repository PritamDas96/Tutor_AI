# TutorAI – Intelligent Conversational Learning Assistant (Hugging Face / Open-source / LLM-agnostic)
# ---------------------------------------------------------------------
# Features:
# - Uses open-source LLMs from Hugging Face via Inference API (no vendor lock-in)
# - Simple chatbot flow + “Generate Personalized Study Notes”
# - Scenario-based system prompt (choose one scenario to demo)
# - Keeps chat history, adjustable temperature/top_p/max_new_tokens
# - Graceful error handling and model fallback
#
# Setup:
# 1) pip install streamlit huggingface_hub>=0.23
# 2) Put your HF token in Streamlit secrets as HF_TOKEN, or paste in the sidebar.
#    (Never hardcode your secret in code committed to git.)
#
# Run: streamlit run app.py

import os
import time
import uuid
import streamlit as st
from typing import List, Dict, Any
from huggingface_hub import InferenceClient

# ----------------------------
# UI + App Config
# ----------------------------
st.set_page_config(page_title="TutorAI – Open-Source Tutor", layout="wide")
st.title("🎓 TutorAI – Intelligent Conversational Learning Assistant")

# ----------------------------
# Model & Runtime Defaults
# ----------------------------
DEFAULT_MODELS = [
    # All are instruction-tuned, chat-friendly open-source models hosted on HF
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

GEN_DEFAULTS = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}

# ----------------------------
# Scenarios (choose one to demo)
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
- Emphasize real-world risks and company-friendly best practices.
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
    selected_scenario = st.selectbox("Learning Scenario", list(SCENARIOS.keys()))
    model_id = st.selectbox("HF Model", DEFAULT_MODELS, index=0,
                            help="All options are open-source models hosted on Hugging Face.")
    temperature = st.slider("Temperature", 0.0, 1.5, GEN_DEFAULTS["temperature"], 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, GEN_DEFAULTS["top_p"], 0.05)
    max_new_tokens = st.slider("Max new tokens", 64, 2048, GEN_DEFAULTS["max_new_tokens"], 32)
    repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, GEN_DEFAULTS["repetition_penalty"], 0.01)

    st.markdown("---")
    st.subheader("🔐 Hugging Face Token")
    pasted_token = st.text_input(
        "Paste HF token (starts with 'hf_...')",
        type="password",
        help="You can also set st.secrets['HF_TOKEN'] for safer storage."
    )
    hf_token = st.secrets.get("HF_TOKEN") or pasted_token or os.environ.get("HF_TOKEN", "")

    if not hf_token:
        st.warning("No Hugging Face token found. Paste it above or set st.secrets['HF_TOKEN'].")

    st.markdown("---")
    st.caption("Tip: If a model is rate-limited or slow, switch models or reduce max_new_tokens.")

# ----------------------------
# Session State (Chat Memory)
# ----------------------------
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    # Seed the conversation with a system + a short assistant welcome
    st.session_state.messages: List[Dict[str, str]] = [
        {"role": "system", "content": SCENARIOS[selected_scenario]},
        {"role": "assistant", "content": "Hello! I’m TutorAI. What would you like to learn today?"}
    ]

# If the user switches scenarios midstream, update the system prompt (keep prior user turns separate)
if st.session_state.messages and st.session_state.messages[0]["content"] != SCENARIOS[selected_scenario]:
    st.session_state.messages[0] = {"role": "system", "content": SCENARIOS[selected_scenario]}

# ----------------------------
# Utilities
# ----------------------------
def build_prompt_from_history(history: List[Dict[str, str]], user_input: str) -> str:
    """
    Build a simple, model-agnostic prompt by concatenating roles and text.
    This works reliably across many instruction-tuned models without bespoke chat templates.
    """
    lines = []
    for m in history:
        if m["role"] == "system":
            lines.append(f"System:\n{m['content'].strip()}\n")
        elif m["role"] == "user":
            lines.append(f"User:\n{m['content'].strip()}\n")
        elif m["role"] == "assistant":
            lines.append(f"Assistant:\n{m['content'].strip()}\n")
    lines.append(f"User:\n{user_input.strip()}\n")
    lines.append("Assistant:")
    return "\n".join(lines)

def call_hf_inference(
    model: str,
    prompt: str,
    token: str,
    gen_cfg: Dict[str, Any],
) -> str:
    """
    Call Hugging Face Inference API using huggingface_hub.InferenceClient.
    We use .text_generation for broad model compatibility.
    """
    client = InferenceClient(model=model, token=token)

    # Some models require an explicit stop sequence to avoid over-generation of role tags.
    stop_seq = ["\nUser:", "\nSystem:", "\nAssistant:\nUser:"]

    try:
        result = client.text_generation(
            prompt=prompt,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
            temperature=float(gen_cfg.get("temperature", 0.7)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
            stop_sequences=stop_seq,
            # return_full_text=False  # InferenceClient ignores this; we build prompts ourselves
        )
        return result.strip()
    # Graceful failure with useful context
    except Exception as e:
        raise RuntimeError(f"Hugging Face Inference error for model '{model}': {e}")

def tutor_reply(user_text: str) -> str:
    prompt = build_prompt_from_history(
        [m for m in st.session_state.messages if m["role"] in ("system", "user", "assistant")],
        user_text,
    )
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Try selected model, then fall back through a small list automatically
    fallbacks = [model_id] + [m for m in DEFAULT_MODELS if m != model_id]

    errors = []
    for mid in fallbacks:
        try:
            return call_hf_inference(mid, prompt, hf_token, gen_cfg)
        except Exception as err:
            errors.append(str(err))
            time.sleep(0.3)
    # If all models fail, surface the aggregated errors
    raise RuntimeError("All model calls failed:\n- " + "\n- ".join(errors))

def build_notes_prompt(profile: Dict[str, str]) -> str:
    """
    Builds a structured prompt for personalized study notes, aligned to the selected scenario.
    """
    scenario_block = SCENARIOS[selected_scenario]
    return f"""
System:
{scenario_block}

User:
Create a concise, personalized study guide for me based on the profile below. 
Keep it highly actionable, with examples, small exercises, and quick checks-for-understanding. 
Prefer bullet points, tables, and short sections I can read in 5-10 minutes.

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
2) Practical Patterns / Templates (e.g., prompting patterns or SQL patterns depending on scenario)
3) 3–5 Micro-exercises (with solutions or hints)
4) Mini-Checklist (Do/Don’t)
5) A 5-day learning plan (15–20 min/day)

Assistant:
"""

# ----------------------------
# Layout: Chat + Notes
# ----------------------------
chat_col, notes_col = st.columns([2, 1], gap="large")

with chat_col:
    st.subheader("💬 Conversational Tutor")
    # Show history
    for m in st.session_state.messages:
        if m["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(m["content"])
        elif m["role"] == "user":
            with st.chat_message("user"):
                st.markdown(m["content"])

    # Input box
    user_prompt = st.chat_input("Ask a question, or say what you want to learn…")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = tutor_reply(user_prompt)
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
            prompt_notes = build_notes_prompt(profile)
            with st.spinner("Drafting your personalized study guide…"):
                try:
                    notes_text = tutor_reply(prompt_notes)  # reuse same model flow
                except Exception as e:
                    notes_text = f"⚠️ Error while generating notes: {e}"
            st.markdown("---")
            st.markdown("#### 📚 Your Study Guide")
            st.write(notes_text)

# ----------------------------
# Footer & Disclaimers
# ----------------------------
st.markdown("---")
st.caption(
    "TutorAI is an educational assistant. It may produce mistakes—verify critical information. "
    "For sensitive or confidential topics, follow your organization’s policies."
)
