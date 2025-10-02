# app.py
# GenAI-Tutor (Agentic, ReAct) — RAG + Tools + LangSmith (HF-only)
# Robust against HF models' imperfect tool-calls by using single-input tools + permissive parsing.

import os, io, json, re, hashlib, requests
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

# LangSmith
from langsmith import Client
from langsmith.run_helpers import tracing_context, trace, get_current_run_tree

# LangChain (ReAct agent + tools + prompt)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------------
# Streamlit & Secrets
# -------------------------------
st.set_page_config(page_title="GenAI-Tutor (Agentic ReAct)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Agentic ReAct (RAG + Tools + LangSmith)</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    st.error("Missing HF token. Add HF_TOKEN in Streamlit Secrets.")
    st.stop()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", os.environ.get("LANGSMITH_API_KEY", ""))
os.environ["LANGSMITH_TRACING"] = str(st.secrets.get("LANGSMITH_TRACING", True)).lower()
LS_PROJECT = st.secrets.get("LANGSMITH_PROJECT", os.environ.get("LANGSMITH_PROJECT", "GenAI-Tutor-Agentic"))
ls_client = Client()

# -------------------------------
# Models & Scenarios
# -------------------------------
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

SCENARIOS: Dict[str, Dict[str, str]] = {
    "Prompt Engineering Basics": {
        "overview": "Core prompting concepts, patterns, templates, hallucination reduction.",
        "system": "You are GenAI-Tutor, an expert coach on prompt engineering for employees. Be concise, practical, and safe."
    },
    "Responsible & Secure GenAI at Work": {
        "overview": "Safe inputs, approvals, phishing risks, policy alignment.",
        "system": "You are GenAI-Tutor for responsible GenAI usage at work. Teach checklist-driven guidance."
    },
    "Automating Everyday Tasks with GenAI": {
        "overview": "Drafting, ideation, structuring notes, time-saving workflows.",
        "system": "You are GenAI-Tutor for everyday task automation. Provide templates and quick workflows."
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# -------------------------------
# RAG corpus (public)
# -------------------------------
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

# -------------------------------
# RAG plumbing
# -------------------------------
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
        for t in soup(["script","style","noscript","header","footer","nav","form"]): t.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = html_bytes.decode("utf-8", errors="ignore")
    return "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())

def _clean_pdf(pdf_bytes: bytes) -> str:
    pages = []
    reader = PdfReader(io.BytesIO(pdf_bytes))
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(ln.strip() for ln in "\n".join(pages).splitlines() if ln.strip())

def fetch_and_clean(url: str) -> str:
    try:
        blob, ctype = _download(url)
        if ".pdf" in url.lower() or "application/pdf" in ctype:
            return _clean_pdf(blob)
        return _clean_html(blob)
    except Exception:
        return ""

def chunk_text(text: str, url: str, title: str) -> List[Dict[str, Any]]:
    if not text: return []
    words = text.split()
    chunks, start, k = [], 0, 0
    while start < len(words):
        end = min(start+WORDS_PER_CHUNK, len(words))
        piece = " ".join(words[start:end])
        chunk_id = f"{hashlib.sha1(url.encode()).hexdigest()}#{k:04d}"
        chunks.append({"chunk_id":chunk_id,"title":title,"url":url,"text":piece})
        if end == len(words): break
        start = max(0, end-OVERLAP_WORDS); k += 1
    return chunks

@st.cache_resource(show_spinner=True)
def load_embedder(): return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource(show_spinner=True)
def load_reranker(): return CrossEncoder("BAAI/bge-reranker-v2-m3")

def embed_texts(texts: List[str], model): 
    X = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    return X.astype("float32")

@st.cache_resource(show_spinner=True)
def build_faiss(vectors: np.ndarray):
    import faiss
    d = vectors.shape[1]; index = faiss.IndexFlatIP(d); index.add(vectors); return index

@st.cache_resource(show_spinner=True)
def build_rag_index(doc_links: List[Dict[str, Any]]):
    embedder = load_embedder()
    all_chunks = []
    for doc in doc_links:
        if not doc.get("enabled", True): continue
        raw = fetch_and_clean(doc["url"])
        if not raw: continue
        all_chunks.extend(chunk_text(raw, doc["url"], doc["title"]))
    if not all_chunks: raise RuntimeError("No chunks ingested from sources.")
    vectors = embed_texts([c["text"] for c in all_chunks], embedder)
    index = build_faiss(vectors)
    side = {"chunks": all_chunks, "vectors_shape": vectors.shape}
    return index, side

def retrieve(query: str, index, side: Dict[str, Any]) -> List[Dict[str, Any]]:
    import faiss
    embedder, reranker = load_embedder(), load_reranker()
    qv = embed_texts([query], embedder)
    scores, idxs = index.search(qv, K_CANDIDATES)
    candidates = []
    for rank, (ci, s) in enumerate(zip(idxs[0], scores[0]), start=1):
        if ci < 0: continue
        c = side["chunks"][ci]
        candidates.append({"rank_ann":rank,"score_ann":float(s),**c})
    if not candidates: return []
    pairs = [(query, c["text"]) for c in candidates]
    rers = reranker.predict(pairs, batch_size=64).tolist()
    for c, rs in zip(candidates, rers): c["score_rerank"] = float(rs)
    return sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)[:TOP_K]

_global_rag = {"index": None, "side": None}

# -------------------------------
# Tool helpers: permissive parsing
# -------------------------------
def extract_json_block(text: str) -> Dict[str, Any]:
    # try to find a JSON object in the text
    try:
        # match first {...} block
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}

def parse_kv(text: str) -> Dict[str, Any]:
    # parse "key=value" pairs like query=foo max_results=5
    out = {}
    for part in re.split(r"[,\n;]\s*|\s{2,}", text):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if v.isdigit(): v = int(v)
            out[k] = v
    return out

# -------------------------------
# Tools: single-string ReAct style
# -------------------------------
def tool_web_search(inp: str) -> str:
    """web_search(input: str) -> JSON string.
    Input can be natural language, or JSON with keys: query (str), max_results (int<=10).
    Returns JSON list of {'title','url','snippet'}.
    """
    params = {"query": inp, "max_results": 5}
    data = extract_json_block(inp) or parse_kv(inp)
    if "query" in data: params["query"] = data["query"]
    if "max_results" in data:
        try: params["max_results"] = max(1, min(10, int(data["max_results"])))
        except: pass
    results = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(params["query"], max_results=params["max_results"], safesearch="moderate"):
                title = r.get("title") or ""
                url = r.get("href") or r.get("url") or ""
                snippet = r.get("body") or ""
                if title and url:
                    results.append({"title": title, "url": url, "snippet": snippet})
    except Exception as e:
        return json.dumps([{"error": f"web_search failed: {e}"}], ensure_ascii=False)
    return json.dumps(results, ensure_ascii=False)

def tool_read_url(inp: str) -> str:
    """read_url(input: str) -> JSON string.
    Input can be the URL itself, or JSON with keys: url (str), max_chars (int).
    Returns {'title','url','text'}.
    """
    data = extract_json_block(inp) or parse_kv(inp)
    url = data.get("url") or inp.strip()
    max_chars = int(data.get("max_chars", 6000)) if str(data.get("max_chars","")).isdigit() else 6000
    try:
        resp = requests.get(url, headers={"User-Agent": "TutorAI/1.0"}, timeout=25)
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type","")).lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            text = _clean_pdf(resp.content)
        else:
            text = _clean_html(resp.content)
        return json.dumps({"title": url, "url": url, "text": (text or "")[:max_chars]}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"title": url, "url": url, "text": f"[Error: {e}]"}, ensure_ascii=False)

def tool_rag_retrieve(inp: str) -> str:
    """rag_retrieve(input: str) -> JSON string.
    Input can be natural language or JSON with keys: query (str), top_k (int).
    Returns JSON list of {'title','url','chunk','score'}.
    """
    if _global_rag["index"] is None or _global_rag["side"] is None:
        return json.dumps([{"error":"RAG index not initialized. Build it from the sidebar."}], ensure_ascii=False)
    data = extract_json_block(inp) or parse_kv(inp)
    q = data.get("query") or inp.strip()
    try:
        k = int(data.get("top_k", TOP_K))
        k = max(1, min(10, k))
    except:
        k = TOP_K
    results = retrieve(q, _global_rag["index"], _global_rag["side"])[:k]
    out = [{"title":c["title"],"url":c["url"],"chunk":c["text"][:900],"score":round(float(c.get("score_rerank",0.0)),4)} for c in results]
    return json.dumps(out, ensure_ascii=False)

TOOLS: List[Tool] = [
    Tool(name="web_search", func=tool_web_search, description="Search the web. Input: natural language or JSON with {query, max_results}. Output: JSON list."),
    Tool(name="read_url", func=tool_read_url, description="Read and clean a URL (HTML/PDF). Input: URL or JSON {url, max_chars}. Output: JSON object."),
    Tool(name="rag_retrieve", func=tool_rag_retrieve, description="Retrieve from curated Gen-AI corpus. Input: natural language or JSON {query, top_k}. Output: JSON list."),
]

# -------------------------------
# HF Fallback (no tools)
# -------------------------------
def hf_direct_reply(model_id: str, user_text: str, system_text: str = "") -> str:
    try:
        client = InferenceClient(model=model_id, token=HF_TOKEN)
        msgs = []
        if system_text: msgs.append({"role":"system","content":system_text})
        msgs.append({"role":"user","content":user_text})
        resp = client.chat_completion(messages=msgs, max_tokens=512, temperature=0.4, top_p=0.9)
        choice = resp.choices[0]
        msg = getattr(choice,"message",None) or choice["message"]
        content = getattr(msg,"content",None) or msg["content"]
        return (content or "").strip()
    except Exception as e:
        return f"⚠️ Fallback HF error: {e}"

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model (chat)", HF_MODELS, index=0)
    enable_agent = st.checkbox("Enable Tools (Agentic)", value=True)
    max_steps = st.slider("Max tool calls per turn", 1, 10, 6)
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
                st.error(f"RAG build failed: {e}")
    with c2:
        st.caption("Uses curated sources; top-k=7 with reranking.")
st.caption(f"Model: **{model_id}**  •  Scenario: **{scenario_name}**")

# -------------------------------
# Build ReAct Agent
# -------------------------------
def get_react_agent(model_id: str, use_tools: bool, max_iterations: int):
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
    )
    llm = ChatHuggingFace(llm=endpoint)
    tools = TOOLS if use_tools else [TOOLS[-1]]  # rag_retrieve only if tools disabled

    # ReAct prompt (clear formatting)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are GenAI-Tutor, a safe, practical Gen-AI coach.\n"
             "Use the ReAct format:\n"
             "Thought: reason about what to do next.\n"
             "Action: one of [{tool_names}] if you need it, else say Final Answer.\n"
             "Action Input: a single string (natural language or JSON) for the tool.\n"
             "Observation: the tool result.\n"
             "Repeat Thought/Action/Action Input/Observation as needed.\n"
             "When done, write Final Answer: <your answer with citations if used>.\n"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        # memory is handled outside for Streamlit history rendering
    )
    return executor

# -------------------------------
# Chat State
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"system","content":SCENARIOS[scenario_name]["system"]},
        {"role":"assistant","content":"Hi! I’m GenAI-Tutor. Ask me anything about Gen-AI."}
    ]
if "scenario_prev" not in st.session_state:
    st.session_state.scenario_prev = scenario_name

def _seed_chat():
    st.session_state.messages = [
        {"role":"system","content":SCENARIOS[scenario_name]["system"]},
        {"role":"assistant","content":"Hi! I’m GenAI-Tutor. Ask me anything about Gen-AI."}
    ]
if st.session_state.scenario_prev != scenario_name:
    _seed_chat()
    st.session_state.scenario_prev = scenario_name

# -------------------------------
# Render history
# -------------------------------
st.markdown("---")
st.subheader("💬 Tutor Chat (Agentic ReAct)")
for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

# -------------------------------
# Handle user prompt
# -------------------------------
user_q = st.chat_input("Ask a question. The agent may use WebSearch, ReadURL, or RAG…")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    agent = get_react_agent(model_id, enable_agent, max_steps)
    # Build chat_history for prompt (LangChain expects BaseMessages; we just send text in memoryless mode)
    chat_history = []  # we display history ourselves; ReAct agent keeps its own scratchpad

    with tracing_context(project_name=LS_PROJECT, metadata={"type":"agent_turn","scenario":scenario_name,"model":model_id}):
        with trace("agent_turn", run_type="chain", inputs={"question": user_q}):
            with st.chat_message("assistant"):
                with st.spinner("Thinking with tools…"):
                    result = {}
                    error_text = ""
                    try:
                        result = agent.invoke(
                            {"input": user_q, "chat_history": chat_history},
                            config={"metadata":{"scenario": scenario_name, "model": model_id},
                                    "tags":["TutorAI","Agentic","ReAct"], "run_name":"TutorAI-ReAct-Turn"},
                        )
                        reply = result.get("output","")
                    except Exception as e:
                        reply = ""
                        error_text = f"Agent exception: {e}"

                if not reply.strip():
                    # Always answer something
                    reply = hf_direct_reply(model_id, f"{SCENARIOS[scenario_name]['system']}\n\nUser question: {user_q}")

                st.markdown(reply or "No reply.")

                # Debug / result inspector
                inter = result.get("intermediate_steps", [])
                if error_text or not result or inter is None:
                    with st.expander("⚙️ Debug (raw agent result / errors)"):
                        if error_text: st.error(error_text)
                        st.json(result or {"note":"No result from agent."})

                if inter:
                    with st.expander("🔍 Agent Trace (ReAct steps)"):
                        for i, (tool_invocation, observation) in enumerate(inter, start=1):
                            tool_name = getattr(tool_invocation, "tool", "_unknown")
                            st.markdown(f"**Step {i}: `{tool_name}`**")
                            try:
                                args_str = json.dumps(tool_invocation.tool_input, ensure_ascii=False)
                            except Exception:
                                args_str = str(tool_invocation.tool_input)
                            st.code(args_str, language="json")
                            obs_view = observation
                            try:
                                if isinstance(observation, (list, dict)):
                                    obs_view = json.dumps(observation, ensure_ascii=False)[:1200]
                                elif isinstance(observation, str):
                                    obs_view = observation[:1200]
                                else:
                                    obs_view = str(observation)[:1200]
                            except Exception:
                                obs_view = str(observation)[:1200]
                            st.text_area("Observation", value=obs_view, height=160, key=f"obs_{i}_{tool_name}")

            st.session_state.messages.append({"role":"assistant","content":reply})

# -------------------------------
# Scenario Overview
# -------------------------------
st.markdown("---")
st.subheader("📌 Scenario Overview")
st.markdown(f"**{scenario_name}**  \n{SCENARIOS[scenario_name]['overview']}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("GenAI-Tutor is educational. Verify critical info. Follow your organization’s policies.")
