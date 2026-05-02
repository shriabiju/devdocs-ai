<div align="center">

# ⚡ DevDocs AI

### Production-Grade Retrieval Augmented Generation for Technical Documentation

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=flat-square)](https://groq.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat-square)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-blue?style=flat-square)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**DevDocs AI** is a domain-specific, production-quality document intelligence system built for software engineers and technical teams. Upload any technical PDF — API references, architecture guides, CS textbooks, system design docs — and query it with natural language. Every answer is grounded in your document, cited to the exact source chunk, and hallucination-resistant by design.

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Evaluation](#-evaluation) · [Roadmap](#-roadmap)

---

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://devdocs-ai.streamlit.app)

**🚀 [Try the live demo → devdocs-ai.streamlit.app](https://devdocs-ai.streamlit.app)**

</div>

---

## 🧠 Why DevDocs AI?

Most RAG demos are toy systems — single vector search, no reranking, no hallucination control, no evaluation. DevDocs AI is built the way production AI teams actually build RAG:

| Concern | Typical RAG Demo | DevDocs AI |
|---|---|---|
| Retrieval | Vector search only | **Hybrid BM25 + Vector** |
| Precision | Top-K raw chunks | **CrossEncoder reranking** |
| Hallucination | Prompt says "be honest" | **Confidence gating + refusal** |
| Evaluation | Manual spot checks | **Automated faithfulness scoring** |
| Prompts | Hardcoded strings | **Versioned prompt config** |
| Domain fit | Generic Q&A | **Software/technical tuned** |

---

## ✨ Features

### 🔍 Hybrid Retrieval Engine
Combines **BM25 keyword search** with **dense vector semantic search** — because vector search alone misses exact technical terms (function names, error codes, version numbers) while BM25 alone misses paraphrased intent. The union of both is deduplicated and passed to the reranker.

### ⚖️ CrossEncoder Reranking
A `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker scores every (query, chunk) pair jointly — unlike bi-encoders, it evaluates them together, dramatically improving precision. Only the highest-scoring chunks reach the LLM.

### 🛡️ Confidence Gating & Hallucination Prevention
If the reranker's best score falls below a calibrated threshold, the system **refuses to answer** rather than fabricating a plausible-sounding response. The LLM is explicitly instructed to cite sources and stay within the retrieved context.

### 📊 Automated Faithfulness Evaluation
An offline evaluation pipeline measures three dimensions:
- **Answer Similarity** — cosine similarity between generated and reference answers
- **Context Relevance** — how well retrieved chunks support the answer
- **Faithfulness** — LLM-as-judge scoring whether every claim is grounded in context

CI integration: `evaluate.py` exits with code `1` if quality drops below threshold, failing the build.

### 🎨 Domain-Specific Responsive UI
Terminal-aesthetic dark interface built for developers — `JetBrains Mono`, green-on-dark palette, source cards with page citations, monospace chat input. Fully responsive across laptops, tablets, and mobile devices. No generic purple-gradient AI look.

### 📦 Versioned Prompt Architecture
System prompts and answer templates live as named constants in `rag_pipeline.py` — not buried in UI code. Changing prompt strategy is a one-line edit with a clear diff history.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DevDocs AI                              │
│                                                                 │
│  ┌──────────┐    ┌────────────────────────────────────────┐    │
│  │  PDF     │───▶│           Ingestion Pipeline           │    │
│  │  Upload  │    │  PyPDFLoader → RecursiveTextSplitter   │    │
│  └──────────┘    │  chunk_size=800, overlap=100           │    │
│                  └──────────────┬─────────────────────────┘    │
│                                 │                               │
│                    ┌────────────┴────────────┐                  │
│                    ▼                         ▼                  │
│           ┌──────────────┐         ┌──────────────┐            │
│           │  FAISS Index │         │  BM25 Index  │            │
│           │  (semantic)  │         │  (keyword)   │            │
│           └──────┬───────┘         └──────┬───────┘            │
│                  │                         │                    │
│                  └──────────┬──────────────┘                   │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Hybrid Merge   │  deduplicate             │
│                    │  + Dedup        │  by content hash         │
│                    └────────┬────────┘                          │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  CrossEncoder   │  ms-marco-MiniLM-L-6-v2  │
│                    │  Reranker       │  joint (q, chunk) scoring │
│                    └────────┬────────┘                          │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Confidence     │  score < threshold →     │
│                    │  Gate           │  refuse, don't hallucinate│
│                    └────────┬────────┘                          │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Groq LLM       │  llama-3.3-70b-versatile │
│                    │  Generation     │  versioned system prompt │
│                    └────────┬────────┘                          │
│                             ▼                                   │
│                    ┌─────────────────┐                          │
│                    │  Answer +       │  cited to source chunks  │
│                    │  Citations      │  with page numbers       │
│                    └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Groq · `llama-3.3-70b-versatile` | Free API, fastest inference available |
| **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace) | Lightweight, strong semantic quality |
| **Vector Store** | FAISS (Meta) | In-memory, blazing fast, production-proven |
| **Keyword Search** | BM25 (rank-bm25) | Exact term matching for technical queries |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | State-of-the-art passage reranking |
| **Orchestration** | LangChain | Document loading, splitting, retrieval |
| **UI** | Streamlit | Fast, Python-native, deploy-ready |
| **Evaluation** | sentence-transformers + Groq judge | Faithfulness, similarity, relevance |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (no credit card required)
- A free [HuggingFace token](https://huggingface.co/settings/tokens) (for embedding model downloads)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/shriabiju/devdocs-ai.git
cd devdocs-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export GROQ_API_KEY=gsk_your_key_here
export HF_TOKEN=hf_your_token_here

# Windows
set GROQ_API_KEY=gsk_your_key_here
set HF_TOKEN=hf_your_token_here

# 5. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501` and upload a PDF to start querying.

### CLI Usage

```bash
# Test the pipeline directly without the UI
python main.py --pdf path/to/your.pdf --query "what is inheritance?"
```

---

## 📁 Project Structure

```
devdocs-ai/
│
├── app.py                  # Streamlit UI — terminal-aesthetic frontend
├── rag_pipeline.py         # Core RAG engine — retrieval, reranking, generation
├── main.py                 # CLI entrypoint for testing the pipeline
├── evaluate.py             # Offline evaluation — faithfulness, similarity, CI gate
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 📊 Evaluation

DevDocs AI ships with a full offline evaluation pipeline. To run it:

```bash
python evaluate.py
```

The script measures three metrics across a golden dataset of question-answer pairs:

```
===== RAG Evaluation =====

Q: What is synchronous parallel processing?
  Answer Similarity : 0.8821
  Context Relevance : 0.7643
  Faithfulness      : 1.0000

[REFUSAL TEST] Q: What is quantum computing?
  Correctly refused: True

===== FINAL SCORES =====
Avg Answer Similarity : 0.8821  (threshold: 0.60)
Avg Context Relevance : 0.7643  (threshold: 0.50)
Avg Faithfulness      : 1.0000  (threshold: 0.70)

✅ EVALUATION PASSED
```

**CI Integration:** The script exits with code `1` on failure — wire it into GitHub Actions to fail PRs that degrade quality.

```yaml
# .github/workflows/eval.yml
- name: Run RAG Evaluation
  run: python evaluate.py
  env:
    GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
```

---

## 🔬 Retrieval Deep Dive

### Why Hybrid Retrieval?

Pure vector search fails on exact technical terms. A query for `pthread_mutex_lock` needs keyword matching — semantic similarity alone won't find it reliably. BM25 handles this perfectly. Conversely, a vague query like "how do threads share state" benefits from semantic understanding. Combining both gives you the best of each approach.

### Why Reranking?

Initial retrieval (both vector and BM25) optimizes for recall — get potentially relevant chunks. Reranking optimizes for precision — score each chunk against the query jointly so the LLM only sees the most relevant context. The CrossEncoder processes (query, chunk) as a pair, capturing cross-attention between them, which is far more accurate than independent embeddings.

### Confidence Gating

Every retrieved set has a maximum reranker score. If that score falls below the calibrated threshold (`-2.0`), the system refuses to answer rather than passing weak context to the LLM. This is the primary mechanism for preventing hallucination — not just prompting.

---

## 🗺️ Roadmap

- [ ] Multi-document support (index entire documentation sites)
- [ ] Streaming responses
- [ ] Conversation memory (multi-turn Q&A)
- [ ] RAGAS integration for evaluation
- [ ] Docker deployment
- [ ] Web scraping ingestion (ingest docs directly from URLs)
- [ ] Query expansion with HyDE (Hypothetical Document Embeddings)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change. Please ensure `evaluate.py` passes before submitting a PR.

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---

<div align="center">

Built with ⚡ by a developer who got tired of hallucinating chatbots.

**[⭐ Star this repo](https://github.com/shriabiju/devdocs-ai)** if it helped you.

</div>