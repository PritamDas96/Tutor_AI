# app.py
# GenAI-Tutor (Agentic RAG + Tools + LangSmith)
# Tools: web_search, read_url, rag_retrieve
# Agent: structured chat agent with HuggingFace LLM
# Observability: LangSmith tracing + thumbs feedback

import os, io, json, time, hashlib, requests
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from langsmith import Client
from langsmith.run_helpers import tracing_context, trace, get_current_run_tree

from pydantic import BaseModel, Field
from langchain_core.tools import tool as lc_tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Streamlit config
st.set_page_config(page_title="GenAI-Tutor (Agentic)", layout="wide")
st.markdown("<h1>🎓 GenAI-Tutor — Agentic Learning Assistant (RAG + Tools + LangSmith)</h1>", unsafe_allow_html=True)

HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN", "")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

os.environ["LANGSMITH_API_KEY"] = st.secrets.get("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_TRACING"] = "true"
LS_PROJECT = "GenAI-Tutor-Agentic"
ls_client = Client()

# --- HF models
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
]

# --- Scenarios
SCENARIOS = {
    "Prompt Engineering Basics": {
        "overview": "Core prompting concepts, patterns, templates, hallucination reduction",
        "system": "You are GenAI-Tutor for prompt engineering."
    },
    "Responsible & Secure GenAI at Work": {
        "overview": "Safe inputs, approvals, risks, policy alignment",
        "system": "You are GenAI-Tutor for responsible GenAI usage."
    },
}
SCENARIO_NAMES = list(SCENARIOS.keys())

# --- Sidebar
with st.sidebar:
    scenario_name = st.selectbox("Learning Scenario", SCENARIO_NAMES, index=0)
    model_id = st.selectbox("HF Model", HF_MODELS, index=0)
    enable_agent = st.checkbox("Enable Tools (Agentic)", value=True)
    max_steps = st.slider("Max tool calls per turn", 1, 5, 3)
st.caption(f"Model: {model_id} • Scenario: {scenario_name}")

# ======================================================
# RAG setup (chunks + embeddings + retriever)
# ======================================================
TOP_K, K_CANDIDATES = 7, 30
WORDS_PER_CHUNK, OVERLAP_WORDS = 450, 80

@st.cache_resource
def load_embedder(): return SentenceTransformer("BAAI/bge-small-en-v1.5")
@st.cache_resource
def load_reranker(): return CrossEncoder("BAAI/bge-reranker-v2-m3")

def embed_texts(texts, model): return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

@st.cache_resource
def build_faiss(vectors):
    import faiss
    d = vectors.shape[1]; idx = faiss.IndexFlatIP(d); idx.add(vectors)
    return idx

def fetch_and_clean(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla"}, timeout=20); r.raise_for_status()
        if url.endswith(".pdf") or "pdf" in r.headers.get("Content-Type","").lower():
            return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(r.content)).pages)
        else:
            soup = BeautifulSoup(r.content, "html.parser")
            for t in soup(["script","style"]): t.decompose()
            return "\n".join(ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip())
    except: return ""

def chunk_text(text, url, title):
    words=text.split(); chunks=[]; start=0; k=0
    while start<len(words):
        end=min(start+WORDS_PER_CHUNK,len(words))
        piece=" ".join(words[start:end])
        chunks.append({"chunk_id":f"{hashlib.sha1(url.encode()).hexdigest()}#{k:04d}","title":title,"url":url,"text":piece})
        if end==len(words): break
        start=max(0,end-OVERLAP_WORDS); k+=1
    return chunks

@st.cache_resource
def build_rag_index():
    docs=[{"title":"Ethical GenAI","url":"https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2025.1565938/full"}]
    embedder=load_embedder(); all_chunks=[]
    for d in docs: all_chunks+=chunk_text(fetch_and_clean(d["url"]),d["url"],d["title"])
    vectors=embed_texts([c["text"] for c in all_chunks],embedder)
    return build_faiss(vectors), {"chunks":all_chunks}

def retrieve(q,index,side):
    import faiss; emb=load_embedder(); rer=load_reranker()
    qv=embed_texts([q],emb); scores,idx=index.search(qv,K_CANDIDATES)
    cands=[{**side["chunks"][i],"score_ann":float(s)} for i,s in zip(idx[0],scores[0]) if i>=0]
    pairs=[(q,c["text"]) for c in cands]; rs=rer.predict(pairs).tolist()
    for c,s in zip(cands,rs): c["score_rerank"]=float(s)
    return sorted(cands,key=lambda x:x["score_rerank"],reverse=True)[:TOP_K]

_global_rag={"index":None,"side":None}

# ======================================================
# Tools
# ======================================================
class WebSearchInput(BaseModel):
    query:str=Field(...); max_results:int=Field(5,ge=1,le=10)
@lc_tool("web_search",args_schema=WebSearchInput)
def web_search_tool(query:str,max_results:int=5):
    out=[]; 
    with DDGS() as ddg:
        for r in ddg.text(query,max_results=max_results): 
            out.append({"title":r["title"],"url":r["href"],"snippet":r["body"]})
    return out

class ReadUrlInput(BaseModel):
    url:str=Field(...); max_chars:int=Field(6000)
@lc_tool("read_url",args_schema=ReadUrlInput)
def read_url_tool(url:str,max_chars:int=6000):
    return {"title":url,"url":url,"text":fetch_and_clean(url)[:max_chars]}

class RAGRetrieveInput(BaseModel):
    query:str=Field(...); top_k:int=Field(7)
@lc_tool("rag_retrieve",args_schema=RAGRetrieveInput)
def rag_retrieve_tool(query:str,top_k:int=7):
    if not _global_rag["index"]: return [{"error":"No RAG index"}]
    return [{"title":c["title"],"url":c["url"],"chunk":c["text"][:500]} for c in retrieve(query,_global_rag["index"],_global_rag["side"])]

# ======================================================
# Agent
# ======================================================
@st.cache_resource
def get_agent(model_id:str,use_tools:bool,max_iterations:int):
    endpoint=HuggingFaceEndpoint(repo_id=model_id,task="text-generation",
                                 huggingfacehub_api_token=HF_TOKEN,
                                 max_new_tokens=512,temperature=0.3,top_p=0.9)
    llm=ChatHuggingFace(llm=endpoint)
    tools=[rag_retrieve_tool] if not use_tools else [web_search_tool,read_url_tool,rag_retrieve_tool]
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    prompt=ChatPromptTemplate.from_messages([
        ("system","You are GenAI-Tutor.\nHere are the tools:\n{tools}\nYou may call: {tool_names}"),
        MessagesPlaceholder("chat_history"),
        ("human","{input}"),
        ("assistant","{agent_scratchpad}"),
    ])
    agent=create_structured_chat_agent(llm=llm,tools=tools,prompt=prompt)
    return AgentExecutor(agent=agent,tools=tools,verbose=True,max_iterations=max_iterations,
                         return_intermediate_steps=True,handle_parsing_errors=True,memory=memory)

# ======================================================
# Chat state
# ======================================================
if "messages" not in st.session_state:
    st.session_state.messages=[{"role":"system","content":SCENARIOS[scenario_name]["system"]},
                               {"role":"assistant","content":"Hi! I’m GenAI-Tutor."}]

# ======================================================
# Chat UI
# ======================================================
st.subheader("💬 Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"] if m["role"] in ["user","assistant"] else "assistant"):
        st.markdown(m["content"])

q=st.chat_input("Ask me something...")
if q:
    st.session_state.messages.append({"role":"user","content":q})
    agent=get_agent(model_id,enable_agent,max_steps)
    result={}; reply=""
    try:
        result=agent.invoke({"input":q})
        reply=result.get("output","")
    except Exception as e:
        reply=f"⚠️ Agent error: {e}"
    with st.chat_message("assistant"):
        st.markdown(reply or "No reply.")
        inter=result.get("intermediate_steps",[])
        if inter:
            with st.expander("🔍 Agent Trace"):
                for i,(call,obs) in enumerate(inter,1):
                    st.markdown(f"**Step {i}: {call.tool}**")
                    st.code(json.dumps(call.tool_input),language="json")
                    obs_view=json.dumps(obs)[:1200] if isinstance(obs,(list,dict)) else str(obs)[:1200]
                    st.text_area("Observation",value=obs_view,height=160,
                                 key=f"obs_{i}_{call.tool}")  # ✅ unique key
    st.session_state.messages.append({"role":"assistant","content":reply})

