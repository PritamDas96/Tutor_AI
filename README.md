<div align="center">

# 🎓 GenAI-Tutor

### An Intelligent Conversational Learning Assistant for Generative-AI Upskilling

*Four progressively advanced Streamlit applications — from a plain chatbot to a fully agentic, citation-grounded tutor — that teach employees how to use Generative AI safely and effectively at work.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Inference-FFD21E?logo=huggingface&logoColor=black)
![RAG](https://img.shields.io/badge/RAG-FAISS%20%2B%20Reranker-005571)
![LangChain](https://img.shields.io/badge/LangChain-Agentic-1C3C3C?logo=langchain&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-00A67E)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [The Four Editions](#-the-four-editions)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration (Secrets)](#configuration-secrets)
  - [Running the Apps](#running-the-apps)
- [Usage Guide](#-usage-guide)
- [The RAG Pipeline in Detail](#-the-rag-pipeline-in-detail)
- [The Agentic System in Detail](#-the-agentic-system-in-detail)
- [Observability & Evaluation](#-observability--evaluation)
- [Supported Models](#-supported-models)
- [Deployment (Streamlit Community Cloud)](#-deployment-streamlit-community-cloud)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License & Disclaimer](#-license--disclaimer)

---

## 🌟 Overview

**GenAI-Tutor** is an educational assistant that helps non-technical and technical employees learn how to work with Generative AI — covering prompt engineering, responsible & secure usage, task automation, business writing, summarization, and evaluation.

The project is deliberately structured as **four standalone applications** that form a learning curriculum in modern LLM application engineering. Each edition solves the previous one's core limitation:

```
Chatbot  ──►  + Retrieval (RAG)  ──►  + Observability & Eval  ──►  + Agentic Tool-Use
   v1              v2                        v3                          v4
```

Every edition shares the same UX contract:
- A **sidebar with two dropdowns** — *Learning Scenario* and *HF Model*.
- A **Scenario Overview** card, an expandable **Personalized Study Notes** generator, and a **Tutor Chat**.

All large-language-model inference runs on **open-source models via the Hugging Face Inference API** — no proprietary API keys required beyond a free Hugging Face token.

---

## 🧩 The Four Editions

| Edition | File | Core Capability | Best For |
|---|---|---|---|
| **v1 — Chatbot** | `TUtor_AI.py` | Multi-turn chat + curated-link study notes | Fast, lightweight tutoring; minimal dependencies |
| **v2 — RAG** | `Tutor_AI_RAG.py` | Retrieval-Augmented Generation with citations | Grounded answers from a curated research corpus |
| **v3 — RAG + Observability** | `Tutor_AI-RAG-Langsmith.py` | RAG + LangSmith tracing + RAGAS evaluation | Measuring & monitoring answer quality |
| **v4 — Agentic** | `Tutor_AI_AGENTIC.py` | Autonomous planner with RAG + web search + URL reading | The flagship: dynamic, tool-using, self-correcting tutor |

> 💡 **New here? Start with v1** to see the core experience, then explore v4 for the full agentic system.

---

## ✨ Key Features

- 🎯 **Scenario-driven tutoring** — six learning scenarios (prompt engineering, responsible AI, automation, writing, summarization, evaluation), each with a tailored system persona.
- 📝 **Personalized study-note generator** — builds a study guide from a profile form (role, team, level, goals, pain points, preferred style, time budget) with "Other" free-text escape hatches.
- 📚 **Citation-grounded RAG** *(v2–v4)* — answers cite `[n]` markers mapped to real source URLs; the model is instructed never to invent links.
- 🌐 **Live web search** *(v4)* — free DuckDuckGo search + full-article reading when the curated corpus is insufficient.
- 🤖 **Agentic planning** *(v4)* — an LLM planner chooses tools, reflects on progress, and stops when it has enough evidence, with a transparent reasoning trace.
- 🔬 **Observability & evaluation** *(v3)* — LangSmith tracing, 👍/👎 feedback logging, and RAGAS faithfulness / answer-relevancy scoring (no OpenAI dependency).
- 🔒 **Secure by design** — tokens loaded only from Streamlit Secrets / environment; educational safety disclaimers throughout.
- ⬇️ **Exportable notes** — download generated study guides as Markdown *(v1)*.

---

## 🏗 Architecture

### RAG data flow (v2 / v3)

```mermaid
flowchart LR
    A[Curated Source URLs] --> B[Fetch & Clean<br/>HTML / PDF]
    B --> C[Chunk<br/>~450 words, 80 overlap]
    C --> D[Embed<br/>BAAI/bge-small-en-v1.5]
    D --> E[FAISS IndexFlatIP<br/>cosine via normalized vectors]
    F[User Query] --> G[Embed Query]
    G --> E
    E --> H[Top-30 Candidates]
    H --> I[Cross-Encoder Rerank<br/>ms-marco-MiniLM-L-6-v2]
    I --> J[Top-7 Evidence]
    J --> K[Build Context + Citations]
    K --> L[HF Chat Model]
    L --> M[Grounded Answer with n citations]
```

### Agentic loop (v4)

```mermaid
flowchart TD
    Q[User Question] --> P{Planner LLM<br/>JSON action}
    P -->|rag_retrieve| R[Curated Corpus]
    P -->|web_search| W[DuckDuckGo]
    P -->|read_url| U[Fetch full article]
    R --> E[Evidence Store<br/>dedup + quality scoring]
    W --> E
    U --> E
    E --> P
    P -->|reflect every 3 steps| P
    P -->|stop: enough evidence| S[Final Synthesis LLM]
    S --> ANS[Citation-Grounded Answer + Reasoning Trace]
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **UI / App framework** | Streamlit |
| **LLM inference** | Hugging Face `InferenceClient` (open models via Inference Providers) |
| **Embeddings** | `sentence-transformers` — `BAAI/bge-small-en-v1.5` |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Vector index** | FAISS (`IndexFlatIP`, exact cosine) with a NumPy fallback |
| **Parsing** | BeautifulSoup (HTML), pypdf (PDF) |
| **Web search** *(v4)* | LangChain Community + DuckDuckGo (`ddgs`) |
| **Agent orchestration** *(v4)* | Custom ReAct-style loop (no framework lock-in) |
| **Observability** *(v3)* | LangSmith tracing & feedback |
| **Evaluation** *(v3)* | RAGAS (Faithfulness, Answer Relevancy) with HF judge & embeddings |

---

## 📂 Repository Structure

```
Tutor_AI/
├── TUtor_AI.py                 # v1 - Chatbot + curated study notes
├── Tutor_AI_RAG.py             # v2 - RAG (FAISS + cross-encoder reranker)
├── Tutor_AI-RAG-Langsmith.py   # v3 - RAG + LangSmith tracing + RAGAS eval
├── Tutor_AI_AGENTIC.py         # v4 - Agentic system (RAG + web + read-url)
├── rag_core.py                 # Shared RAG pipeline used by v2/v3/v4
├── ui.py                       # Shared UI helpers (theme, hero, status chips)
├── run_app.py                  # Windows-safe launcher (preloads torch)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   ├── config.toml             # Theme + server config
│   └── secrets.toml            # HF_TOKEN etc.  (git-ignored, create locally)
├── assets/                     # Screenshots
├── PROJECT_DOCUMENT.md         # Project write-up (also as PDF)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** (tested on 3.13)
- A free **Hugging Face account** and an **access token** with *read* scope — <https://huggingface.co/settings/tokens>
- *(Optional)* A **LangSmith API key** for tracing/eval in v3/v4 — <https://smith.langchain.com>
- ~3 GB free disk for model weights and PyTorch (for v2–v4)

### Installation

```bash
# 1) Clone
git clone https://github.com/PritamDas96/Tutor_AI.git
cd Tutor_AI

# 2) Create an isolated virtual environment
python -m venv .venv

# 3) Activate it
#   Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
#   macOS / Linux:
source .venv/bin/activate

# 4) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ The first install pulls PyTorch, FAISS and Transformers and may take several minutes.

### Configuration (Secrets)

Create `.streamlit/secrets.toml` (this file is **git-ignored** — never commit it):

```toml
# Required — Hugging Face token (read scope)
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Optional — LangSmith (only used by v3 / v4). Leave blank to disable tracing.
LANGSMITH_API_KEY = ""
LANGSMITH_TRACING  = false
LANGSMITH_PROJECT  = "GenAI-Tutor"
```

Alternatively, export the same values as environment variables (`HF_TOKEN`, `LANGSMITH_API_KEY`, …).

> 🔐 **Security:** the token is read only from Secrets / environment and is never written to code or logs. If a token is ever exposed, rotate it immediately in your Hugging Face settings.

### Running the Apps

```bash
# v1 — Chatbot
streamlit run TUtor_AI.py

# v2 — RAG
streamlit run Tutor_AI_RAG.py

# v3 — RAG + Observability + Evaluation
streamlit run "Tutor_AI-RAG-Langsmith.py"

# v4 — Agentic (flagship)
streamlit run Tutor_AI_AGENTIC.py
```

The app opens at **http://localhost:8501**. On the first RAG/agent query, the embedder and reranker weights download once and are cached locally.

> 🪟 **Windows note:** if a RAG/agent app crashes with `OSError: [WinError 1114] ... c10.dll`, launch it through the included helper instead — it preloads PyTorch in the main thread before Streamlit starts:
> ```bash
> python run_app.py Tutor_AI_RAG.py 8502
> python run_app.py Tutor_AI_AGENTIC.py 8504
> ```
> This is a Windows-only quirk (Streamlit runs the app in a worker thread where torch's DLL can fail to initialize); Linux / Streamlit Cloud is unaffected.

---

## 📘 Usage Guide

1. **Pick a Learning Scenario and Model** in the sidebar.
2. **Read the Scenario Overview** card for the topic's key points.
3. **Generate Personalized Study Notes** — expand the panel, fill in your profile (role, goals, pain points, style, time/day), and click **Generate Notes**. In v2–v4 with RAG enabled, notes are grounded in the corpus with citations. You can insert notes into the chat context or download them.
4. **Chat with the Tutor** — ask anything about the scenario. In v2–v4, expand **Evidence** to see the exact sources behind each answer.
5. **(v4) Inspect the reasoning trace** — expand *Agent Reasoning Trace* to see which tools the agent chose, what it found, and why it stopped.
6. **(v3) Evaluate** — after a few chats, open *Observe & Evaluate* and run RAGAS to score faithfulness and answer relevancy.

---

## 🔎 The RAG Pipeline in Detail

| Stage | Implementation | Default |
|---|---|---|
| **Ingest** | `requests` with a browser User-Agent; cached with `@st.cache_data` | 5 curated sources |
| **Clean** | BeautifulSoup strips `script/style/nav/header/footer/form`; pypdf extracts PDF text | — |
| **Chunk** | Word-based sliding window with overlap | 450 words / 80 overlap |
| **Embed** | `BAAI/bge-small-en-v1.5`, L2-normalized | 384-dim |
| **Index** | FAISS `IndexFlatIP` (cosine via inner product on normalized vectors) | exact search |
| **Retrieve** | ANN recall → cross-encoder rerank → top-k | 30 candidates → top-7 |
| **Generate** | Context + numbered citations injected as a system message; model instructed to use *only* the context | — |

The curated corpus (2025 research on Generative AI in education) includes Frontiers, Google, HEPI, Nature, and ACL/COLING sources. Unreachable sources (e.g. those returning HTTP 403) are skipped gracefully at build time.

**Caching design:** `@st.cache_data` caches downloaded bytes; `@st.cache_resource` caches models, embeddings and the FAISS index so they persist across Streamlit reruns. Use **Refresh RAG Corpus** to rebuild.

---

## 🤖 The Agentic System in Detail

The v4 planner communicates in strict JSON and follows a ReAct-style *think → act → observe* loop:

```json
{"thought": "reasoning about what to do next"}
{"tool": "rag_retrieve|web_search|read_url", "input": {"query": "..."}}
{"stop": true, "reason": "sufficient evidence gathered"}
```

**Tools**
- `rag_retrieve` — curated corpus search, returns a quality signal (`avg_score`: HIGH ≥ 0.5 / MEDIUM ≥ 0.3 / LOW).
- `web_search` — DuckDuckGo, with keyword-broadening retry.
- `read_url` — fetches a page (browser headers + retry + a reader-proxy fallback for blocked pages).

**Robustness features**
- Resilient JSON extraction with trailing-comma repair and a safe default action.
- **Evidence store** with content-hash deduplication and a *blocked-domain / failed-URL memory* to avoid re-hitting dead sources.
- **Sliding context window** (last 5 interactions) to bound prompt size.
- **Reflection checkpoints** every 3 steps for self-assessment.
- Deterministic auto-follow of the top 1–2 web results.
- Multiple termination guards (max steps, invalid-call cap, no-evidence cap).
- **Two-phase generation:** the loop only gathers evidence; a separate synthesis call writes the final, citation-grounded answer.

---

## 📊 Observability & Evaluation

*(v3 — `Tutor_AI-RAG-Langsmith.py`)*

- **LangSmith tracing** — each chat/notes turn is a hierarchical trace (chain → retriever → LLM) with metadata. Tracing activates **only when `LANGSMITH_API_KEY` is set**; otherwise it is disabled cleanly (no log noise).
- **Human feedback** — 👍/👎 buttons log a `user_score` to the corresponding run.
- **RAGAS evaluation** — runs **Faithfulness** and **Answer Relevancy** over recent chats, using a Hugging Face judge LLM and HF embeddings (no OpenAI fallback). Aggregate scores can be logged back to LangSmith.

---

## 🧠 Supported Models

Selectable from the sidebar (served via Hugging Face Inference Providers):

| Model | Notes |
|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | Fast default |
| `meta-llama/Llama-3.3-70B-Instruct` | Large, high quality |
| `Qwen/Qwen2.5-72B-Instruct` | Large, high quality |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | Strong reasoning |
| `deepseek-ai/DeepSeek-V3-0324` | Very capable |

> ℹ️ Model availability on Hugging Face changes over time. If a model returns *"not supported by provider"*, pick another from the list or update the `HF_MODELS` array. The client uses automatic provider routing.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing to your chosen entry file (e.g. `Tutor_AI_AGENTIC.py`).
3. Add secrets under **App → Settings → Secrets** (same keys as `secrets.toml`).
4. Deploy.

> ⚠️ **Free-tier memory:** the free tier provides ~1 GB RAM. This project uses the lightweight `ms-marco-MiniLM-L-6-v2` reranker to fit within that budget. Avoid swapping in multi-gigabyte rerankers (e.g. `bge-reranker-v2-m3`) unless you deploy on a larger instance.

---

## 🧯 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Missing HF token` and the app stops | No `HF_TOKEN` in Secrets/env | Add it to `.streamlit/secrets.toml` |
| Every chat errors with *"Model not supported by provider"* | The selected model is no longer served | Choose another model or update `HF_MODELS` |
| RAG returns "No chunks ingested" | All sources unreachable/blocked | Check connectivity; replace blocked URLs; click *Refresh RAG Corpus* |
| `OSError: [WinError 1114] ... c10.dll` (v2/v3/v4) | Windows: torch DLL fails to init in Streamlit's worker thread | Launch via `python run_app.py <app> <port>` (preloads torch in the main thread) |
| Very slow first query | Model weights downloading | One-time; weights are cached afterwards |
| LangSmith 401 errors in logs | Tracing enabled without a valid key | Leave `LANGSMITH_API_KEY` blank (tracing auto-disables) |
| Web search returns nothing (v4) | DuckDuckGo rate-limiting | Retry; simplify the query; or provide a direct URL to read |

---

## 🗺 Roadmap

- [ ] Extract a shared `rag_core.py` module to eliminate duplication across editions.
- [ ] Persist the FAISS index to disk to avoid re-embedding on cold start.
- [ ] Token-accurate chunking using the embedder's tokenizer.
- [ ] Add unit/integration tests and CI.
- [ ] Expand and health-check the curated corpus.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss significant changes first, keep edits scoped to a single edition where possible, and never commit secrets.

---

## 📄 License & Disclaimer

Released under the **MIT License** (add a `LICENSE` file if not present).

> **Educational use only.** GenAI-Tutor provides learning assistance and may produce inaccuracies. Verify critical information and follow your organization's security, privacy and compliance policies. Do not enter confidential data or PII.

<div align="center">

*Built with ❤️ using Streamlit, Hugging Face, FAISS and LangChain.*

</div>
