import os
import re
import json
import time
import requests
from typing import Any, Dict, List

import streamlit as st
from bs4 import BeautifulSoup

# ---- LangChain / LangGraph / Tools ----
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langgraph.prebuilt import create_react_agent

# ------------------------------
# Streamlit Page + Secrets
# ------------------------------
st.set_page_config(page_title="LangGraph Agent (HF + LangSmith)", layout="wide")
st.markdown("<h1>🔎 LangGraph Agent • Web Search via LangChain Tools</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
LS_KEY   = st.secrets.get("LANGSMITH_API_KEY") or os.environ.get("LANGSMITH_API_KEY", "")
LS_PROJ  = st.secrets.get("LANGSMITH_PROJECT") or os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-Agentic")

if not HF_TOKEN:
    st.error("Missing HF_TOKEN in Streamlit Secrets or env.")
    st.stop()

# Set env for HF + LangSmith tracing
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
if LS_KEY:
    os.environ["LANGSMITH_API_KEY"] = LS_KEY
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LS_PROJ)

# -----------------------------------
# Robust Utilities (read_url cleaner)
# -----------------------------------
def clean_html(html: bytes) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            t.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = html.decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def http_read(url: str, timeout: int = 20) -> Dict[str, Any]:
    if not url.startswith("http"):
        return {"error": "URL must start with http(s)://"}
    try:
        r = requests.get(url, headers={"User-Agent": "LangGraphAgent/1.0"}, timeout=timeout)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type", "")).lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return {"error": "PDF not supported in this minimal demo."}
        text = clean_html(r.content)
        if len(text) < 80:
            return {"error": "No meaningful content extracted."}
        return {"title": url.split("/")[-1][:100], "url": url, "text": text[:8000]}
    except requests.RequestException as e:
        return {"error": f"Fetch failed: {e}"}

# ----------------------------------------------------
# Query Rewrite Tool (normalization + de-noising)
# ----------------------------------------------------
def rewrite_query(q: str) -> str:
    q0 = q.strip()
    # normalize brand/model tokens
    q1 = re.sub(r"\bopen\s*ai\b", "openai", q0, flags=re.I)
    q1 = re.sub(r"\bgpt\s*[- ]?\s*5\b", "gpt-5", q1, flags=re.I)
    q1 = re.sub(r"\bgpt\s*[- ]?\s*4\.?o?\b", "gpt-4o", q1, flags=re.I)

    # remove noise words once
    q1 = re.sub(r"\b(latest|new|pricing model|official)\b", "", q1, flags=re.I)
    # collapse multiple spaces
    q1 = re.sub(r"\s{2,}", " ", q1).strip()
    return q1 or q0

# ----------------------------------------------------
# Web Search Tool (DuckDuckGo via LangChain wrapper)
# - single-use instance per call
# - retry/backoff once
# - dedupe repeated queries
# ----------------------------------------------------
def make_web_search_tool(k: int = 8) -> Tool:
    def _search(q: str) -> str:
        # de-dupe guard across the session
        st.session_state.setdefault("_seen_queries", set())
        if q in st.session_state["_seen_queries"]:
            # auto rewrite once if repeated
            q2 = rewrite_query(q)
            if q2 != q:
                q = q2
            else:
                # add a lightweight broadening: drop years/noisy tokens
                q = re.sub(r"\b(202[0-9])\b", "", q).strip()
        st.session_state["_seen_queries"].add(q)

        def _do(kwords: str) -> List[Dict[str, str]]:
            ddg = DuckDuckGoSearchAPIWrapper(region="us-en", time="y", max_results=k)
            rs = ddg.results(kwords, max_results=k)
            cleaned = []
            for r in rs:
                cleaned.append({
                    "title": (r.get("title") or "")[:140],
                    "link": r.get("link") or "",
                    "snippet": (r.get("snippet") or "")[:300]
                })
            return cleaned

        try:
            results = _do(q)
        except Exception as e:
            # retry once after a short sleep (rate limit / transient)
            time.sleep(1.0)
            results = _do(q)

        # If nothing, try broadened query once
        if not results and len(q.split()) >= 3:
            q_simple = re.sub(r"\b(202[0-9]|latest|pricing model)\b", "", q, flags=re.I)
            q_simple = " ".join(w for w in q_simple.split() if len(w) > 2).strip()
            if q_simple and q_simple.lower() != q.lower():
                results = _do(q_simple)

        # return a compact JSON-ish string to save tokens
        return json.dumps(results[:k], ensure_ascii=False)

    return Tool(
        name="web_search",
        description=(
            "Search the public web (DuckDuckGo). "
            "Use for up-to-date info: prices, docs, news, product features. "
            "Input a concise query."
        ),
        func=_search,
    )

# ----------------------------------------------------
# Read URL Tool
# ----------------------------------------------------
def make_read_url_tool() -> Tool:
    def _read(url: str) -> str:
        out = http_read(url)
        return json.dumps(out, ensure_ascii=False)
    return Tool(
        name="read_url",
        description="Fetch and extract main text from a URL (HTML only, no PDFs). Input must be a full URL.",
        func=_read,
    )

# ----------------------------------------------------
# Query Rewrite Tool (explicit)
# ----------------------------------------------------
def make_rewrite_tool() -> Tool:
    def _rewrite(q: str) -> str:
        return rewrite_query(q)
    return Tool(
        name="rewrite_query",
        description=(
            "Normalize and simplify a search query. "
            "Use this before web_search if the query looks noisy or brand/model names are misspelled."
        ),
        func=_rewrite,
    )

# ----------------------------------------------------
# Build LangGraph Agent (ReAct)
# ----------------------------------------------------
def build_agent(model_id: str) -> Any:
    # Hugging Face chat model via LangChain
    llm = ChatHuggingFace(
        repo_id=model_id,
        temperature=0.2,
        max_new_tokens=512,
    )
    tools = [
        make_rewrite_tool(),
        make_web_search_tool(k=8),
        make_read_url_tool(),
    ]
    agent = create_react_agent(llm, tools)
    return agent

# ----------------------------------------------------
# Invoke Agent
# ----------------------------------------------------
def run_agent(user_query: str, model_id: str, system_prompt: str) -> Dict[str, Any]:
    agent = build_agent(model_id=model_id)
    state = {
        "messages": [
            SystemMessage(system_prompt),
            HumanMessage(user_query),
        ]
    }
    final_state = agent.invoke(state)

    # Extract last AI message
    msgs: List = final_state["messages"]
    answer = ""
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            answer = m.content
            break

    return {"answer": answer, "raw": final_state}

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.header("⚙️ Settings")
model_id = st.sidebar.selectbox(
    "Hugging Face Model",
    [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "google/gemma-2-9b-it",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    index=0
)
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value="You are a precise, practical, and safe AI tutor. Decide when to invoke tools. Prefer web_search for time-sensitive facts.",
    height=120
)

st.sidebar.caption("LangSmith tracing uses your LANGSMITH_API_KEY if provided.")
st.sidebar.markdown("---")
st.sidebar.subheader("Search Health")
if st.sidebar.button("Run health check"):
    try:
        r = requests.get("https://duckduckgo.com", timeout=8)
        st.sidebar.write("DuckDuckGo reachability:", r.status_code)
    except Exception as e:
        st.sidebar.error(f"Egress/SSL issue: {e}")
    try:
        ddg = DuckDuckGoSearchAPIWrapper(region="us-en", time="y", max_results=3)
        rows = ddg.results("OpenAI API pricing", max_results=3)
        st.sidebar.write(f"DDG rows: {len(rows)}")
        if rows:
            st.sidebar.write(rows[0])
    except Exception as e:
        st.sidebar.error(f"DDG error: {e}")

# ------------------------------
# Chat UI
# ------------------------------
st.markdown("---")
st.subheader("💬 Chat")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hi! Ask me something that might need web search. I can also fetch a URL."}
    ]

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_query = st.chat_input("Type your question…")
if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with LangGraph agent…"):
            out = run_agent(
                user_query=user_query,
                model_id=model_id,
                system_prompt=system_prompt
            )
            st.markdown(out["answer"])
            # Debug / Trace panel (raw agent state)
            with st.expander("🔍 Agent Trace (raw LangGraph state)", expanded=False):
                st.json(out["raw"], expanded=False)

    st.session_state.chat.append({"role": "assistant", "content": out["answer"]})

st.markdown("---")
st.caption("Tip: If a query looks too specific, the agent may call `rewrite_query` before `web_search` and then `read_url`.")
