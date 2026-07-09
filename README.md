# RAG Pipeline

> A modular, offline-first **Retrieval-Augmented Generation (RAG)** system that enables semantic search over PDF documents and delivers grounded, source-cited answers using Google Gemini AI.

---

## Overview

Traditional AI models answer questions from their training data alone — they have no knowledge of *your* documents. This project solves that by combining a local vector search engine with a large language model (LLM).

**How it works:**
1. Your PDF/TXT documents are loaded, split into chunks, and converted into numerical vectors (embeddings) using a local transformer model.
2. Those vectors are stored in **ChromaDB** — a persistent vector database — so processing only happens once.
3. At query time, the user's question is embedded and matched against stored vectors to find the most relevant document chunks.
4. The retrieved chunks are sent as context to **Google Gemini**, which generates a precise, grounded answer with source citations.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                      │
│  (Run once — persists to disk)                              │
│                                                             │
│  PDF / TXT Files                                            │
│       │                                                     │
│       ▼                                                     │
│  document.py  →  Load files via LangChain loaders           │
│       │                                                     │
│       ▼                                                     │
│  chunks.py    →  Split into ~1000-char chunks (200 overlap) │
│       │                                                     │
│       ▼                                                     │
│  embeddings.py →  Convert text → 384-dim vectors (local)    │
│       │                                                     │
│       ▼                                                     │
│  vector_store.py → Store vectors in ChromaDB on disk        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                          │
│  (Runs on every user question)                              │
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  rag_retriever.py → Embed query → Similarity search         │
│       │                                                     │
│       ▼                                                     │
│  llm_output.py    → Build prompt → Gemini API → Answer      │
│       │                                                     │
│       ▼                                                     │
│  rag_advanced_class.py → Citations + Summary + History      │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Document Loading | LangChain (`TextLoader`, `PyMuPDFLoader`, `DirectoryLoader`) |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embedding Model | `all-MiniLM-L6-v2` via `sentence-transformers` (local) |
| Vector Database | ChromaDB (persistent, on-disk) |
| LLM | Google Gemini 2.5 Flash via `langchain-google-genai` |
| Environment | Python 3.12+, managed with `uv` |

---

## Setup & Usage

### 1. Clone the repository

```bash
git clone https://github.com/namanbhola1888/RAG-Pipeline.git
cd RAG-Pipeline
```

### 2. Install dependencies

```bash
uv sync
# or
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Add your documents

Place your `.pdf` files in `data/pdf/` and `.txt` files in `data/text_files/`.

### 5. Run the pipeline

```bash
# Ingest documents and query (from the notebook/ directory)
cd notebook
python rag_advanced_class.py
```

---

## License

This project is open for educational and personal use.