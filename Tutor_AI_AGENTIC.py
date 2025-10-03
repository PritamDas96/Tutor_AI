# LangGraph Agent • HF + LangSmith • Streamlit
# - No system prompt shown in UI (constant below)
# - File watcher disabled to avoid inotify errors
# - Uses langchain_huggingface.ChatHuggingFace with HuggingFaceEndpoint (no deprecation)
# - Web search via LangChain DuckDuckGo tool, plus read_url & rewrite_query tools

import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")  # disable file watcher early

import re
import json
import time
import requests
from typing import Any, Dict, List

import streamlit as st
st.set_page_config(page_title="LangGraph Agent (HF + LangSmith)", layout="wide")
# double safety in case the env var is ignored in some hosts:
try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass

from bs4 import BeautifulSoup

# LangChain / LangGraph
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langgraph.prebuilt import create_react_agent

# HF chat wrapper (correct, non-deprecated)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ------------------------------
# Secrets & Environment
# ------------------------------
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
LS_KEY   = st.secrets.get("LANGSMITH_API_KEY") or os.environ.get("LANGSMITH_API_KEY", "")
LS_PROJ  = st.secrets.get("LANGSMITH_PROJECT") or os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-Agentic")

if not HF_TOKEN:
    st.error("Missing HF_TOKEN in Streamlit Secrets or env.")
    st.stop()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
if LS_KEY:
    os.environ["LANGSMITH_API_KEY"] = LS_KEY
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LS_PROJ)

# ------------------------------
# System Prompt (not in UI)
# ------------------------------
SYSTEM_PROMPT = (
    "You are a precise, practical, and safe AI tutor. "
    "Use tools when needed. Prefer web_search for time-sensitive facts. "
    "If search returns nothing, rewrite the query, then try again. "
    "Avoid repeating the exact same failing query."
)

# -----------------------------------
# Utilities: clean & fetch HTML
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

# -----------------------------------
# Query Rewrite
# -----------------------------------
def rewrite_query(q: str) -> str:
    q0 = q.strip()
    q1 = re.sub(r"\bopen\s*ai\b", "openai", q0, flags=re.I)
    q1 = re.sub(r"\bgpt\s*[- ]?\s*5\b", "gpt-5", q1, flags=re.I)
    q1 = re.sub(r"\bgpt\s*[- ]?\s*4\.?o?\b", "gpt-4o", q1, flags=re.I)
    q1 = re.sub(r"\b(latest|new|pricing model|official)\b", "", q1, flags=re.I)
    q1 = re.sub(r"\s{2,}", " ", q1).strip()
    return q1 or q0

# -----------------------------------
# Tools
# -----------------------------------
def make_web_search_tool(k: int = 8) -> Tool:
    def _search(q: str) -> str:
        st.session_state.setdefault("_seen_queries", set())
        if q in st.session_state["_seen_queries"]:
            q2 = rewrite_query(q)
            if q2 != q:
                q = q2
            else:
                q = re.sub(r"\b(202[0-9])\b", "", q).strip()
        st.session_state["_seen_queries"].add(q)

        def _do(kwords: str) -> List[Dict[str, str]]:
            ddg = DuckDuckGoSearchAPIWrapper(region="us-en", time="y", max_results=k)
            rs = ddg.results(kwords, max_results=k)
            out = []
            for r in rs:
                out.append({
                    "title": (r.get("title") or "")[:140],
                    "link": r.get("link") or "",
                    "snippet": (r.get("snippet") or "")[:300]
                })
            return out

        try:
            results = _do(q)
        except Exception:
            time.sleep(1.0)
            results = _do(q)

        if not results and len(q.split()) >= 3:
            q_simple = re.sub(r"\b(202[0-9]|latest|pricing model)\b", "", q, flags=re.I)
            q_simple = " ".join(w for w in q_simple.split() if len(w) > 2).strip()
            if q_simple and q_simple.lower() != q.lower():
                results = _do(q_simple)

        return json.dumps(results[:k], ensure_ascii=False)

    return Tool(
        name="web_search",
        description="Search the public web (DuckDuckGo). Use for prices, docs, news. Input a concise query.",
        func=_search,
    )

def make_read_url_tool() -> Tool:
    def _read(url: str) -> str:
        out = http_read(url)
        return json.dumps(out, ensure_ascii=False)
    return Tool(
        name="read_url",
        description="Fetch and extract main text from a URL (HTML only, no PDFs). Input must be a full URL.",
        func=_read,
    )

def make_rewrite_tool() -> Tool:
    def _rewrite(q: str) -> str:
        return rewrite_query(q)
    return Tool(
        name="rewrite_query",
        description="Normalize/simplify a search query (fix brand/model names, remove noise).",
        func=_rewrite,
    )

# -----------------------------------
# Build Agent (LangGraph ReAct)
# -----------------------------------
def build_agent(model_id: str) -> Any:
    # Correct non-deprecated chat wrapper:
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        temperature=0.2,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    tools = [
        make_rewrite_tool(),
        make_web_search_tool(k=8),
        make_read_url_tool(),
    ]
    agent = create_react_agent(llm, tools)
    return agent

def run_agent(user_query: str, model_id: str) -> Dict[str, Any]:
    agent = build_agent(model_id=model_id)
    state = {
        "messages": [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(user_query),
        ]
    }
    final_state = agent.invoke(state)
    msgs: List = final_state["messages"]
    answer = ""
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            answer = m.content
            break
    return {"answer": answer, "raw": final_state}

# ------------------------------
# Sidebar (no system prompt)
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
st.sidebar.caption("LangSmith tracing is enabled if you set LANGSMITH_API_KEY.")

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
st.markdown("<hr>", unsafe_allow_html=True)
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
            out = run_agent(user_query=user_query, model_id=model_id)
            st.markdown(out["answer"])
            with st.expander("🔍 Agent Trace (raw LangGraph state)", expanded=False):
                st.json(out["raw"], expanded=False)

    st.session_state.chat.append({"role": "assistant", "content": out["answer"]})

st.markdown("---")
st.caption("Tools: rewrite_query → web_search → read_url. No system prompt shown in UI. File watcher disabled to avoid inotify errors.")
