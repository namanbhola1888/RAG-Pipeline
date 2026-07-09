# RAG Pipeline — Complete Project Analysis

---

## 🔍 What is RAG? (In Plain English)

Imagine you have a pile of PDF books and you want to ask a question about them. A regular AI (like ChatGPT) can only answer from what it was trained on. But **RAG (Retrieval-Augmented Generation)** lets the AI *look up the answer from YOUR documents first*, and then generate a reply. It's like giving the AI a personal library card.

---

## 📁 Folder Structure

```
RAG Pipeline/
│
├── main.py                         ← Entry point (currently a placeholder)
├── .env                            ← Secret keys (Gemini API key)
├── pyproject.toml                  ← Project dependencies & Python version
├── requirements.txt                ← Alternative list of packages
├── config_sentence_transformers.json ← Version info for the embedding model
├── README.md                       ← Project overview
│
├── data/
│   ├── pdf/                        ← Your input PDF files go here
│   ├── text_files/                 ← Your input .txt files go here
│   └── vector_store/               ← ChromaDB saves processed data here
│
├── models/
│   └── models/
│       └── all-MiniLM-L6-v2/      ← The local embedding model (offline)
│
└── notebook/                       ← All the core pipeline logic lives here
    ├── document.py                 ← Step 1: Load documents (PDF / TXT)
    ├── chunks.py                   ← Step 2: Split documents into pieces
    ├── embeddings.py               ← Step 3: Convert text to numbers (vectors)
    ├── vector_store.py             ← Step 4: Save vectors to ChromaDB
    ├── rag_retriever.py            ← Step 5: Search for relevant chunks
    ├── llm_output.py               ← Step 6: Send context to Gemini AI and get answer
    └── rag_advanced_class.py       ← Step 7: Full advanced pipeline with citations & history
```

---

## 🗺️ Complete Control Flow (Start → End)

```
User Question
     │
     ▼
[document.py] → Load PDF/TXT files
     │
     ▼
[chunks.py] → Break documents into manageable pieces
     │
     ▼
[embeddings.py] → Convert each piece into a vector (list of numbers)
     │
     ▼
[vector_store.py] → Store those vectors in ChromaDB (database)
     │
     ▼
[rag_retriever.py] → Convert the user's question to a vector, find similar vectors
     │
     ▼
[llm_output.py / rag_advanced_class.py] → Feed the matched text to Gemini AI → Get final answer
```

---

## 📄 File-by-File Deep Dive

---

### 1. [main.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/main.py) — Entry Point (Placeholder)

| | |
|---|---|
| **Purpose** | Official project entry point |
| **Input** | None |
| **What it does** | Prints `"Hello from rag-pipeline!"` — it's a skeleton, not yet wired to the pipeline |
| **Output** | Console print statement |
| **Who uses its output** | Nobody yet — the real pipeline runs from the `notebook/` files directly |

> **Note**: This file exists because the project was scaffolded with `uv` (a Python project tool). The actual working pipeline is in the `notebook/` directory.

#### Functions in execution order:

**`main()`**
- **Purpose**: Placeholder to confirm the project runs
- **Input**: Nothing
- **Process**: Calls `print()`
- **Return**: None

---

### 2. [notebook/document.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/document.py) — Document Loader

| | |
|---|---|
| **Purpose** | Read raw files (TXT and PDF) and convert them into LangChain `Document` objects |
| **Input** | Files from `data/text_files/` and `data/pdf/` folders |
| **What it does** | Uses LangChain's loaders to open and read files. Think of it as "opening the book." |
| **Output** | `document` (single TXT), `documents` (all TXTs), `documents_pdf` (all PDFs) |
| **Who uses its output** | `vector_store.py` imports `documents_pdf` |

#### What is a LangChain Document?
A Python object with two parts:
- `page_content` → the actual text
- `metadata` → information like filename, page number, author

#### How it loads files (in execution order):

**1. `TextLoader` (single file)**
- Opens `data/text_files/python_intro.txt`
- Calls `.load()` → returns a list containing one Document
- Stored in variable `document`

**2. `DirectoryLoader` (all TXT files)**
- Scans all `.txt` files inside `data/text_files/`
- Uses `TextLoader` for each file with UTF-8 encoding
- Calls `.load()` → returns a list of Documents (one per file)
- Stored in variable `documents`

**3. `DirectoryLoader` (all PDF files)**
- Scans all `.pdf` files inside `data/pdf/`
- Uses `PyMuPDFLoader` for each file (PyMuPDF is fast and handles complex PDFs well)
- Calls `.load()` → returns a list of Documents (one per page, typically)
- Stored in variable `documents_pdf`

---

### 3. [notebook/chunks.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/chunks.py) — Text Splitter

| | |
|---|---|
| **Purpose** | Break large documents into smaller, bite-sized pieces called "chunks" |
| **Input** | A list of LangChain Documents |
| **What it does** | Splits each document intelligently so that chunks are not too big for the AI model |
| **Output** | A longer list of smaller Document objects (chunks) |
| **Who uses its output** | `vector_store.py` calls `split_documents()` |

#### Why chunk at all?
AI embedding models have a maximum text size they can process at once. Also, smaller chunks = more precise search results. If you ask about one specific line, you don't want to retrieve a 20-page document.

#### Functions in execution order:

**`split_documents(documents, chunk_size=1000, chunk_overlap=200)`**
- **Input**: List of Documents, optional chunk size (default 1000 characters) and overlap (default 200 characters)
- **Process**:
  - Creates a `RecursiveCharacterTextSplitter` — this tries to split at `\n\n` first, then `\n`, then spaces, then individual characters, to keep text natural
  - Chunk overlap means consecutive chunks share 200 characters — this prevents cutting off important context at boundaries
  - Calls `.split_documents(documents)`
  - Prints how many documents → how many chunks
  - Prints a preview of the first chunk
- **Return**: `split_docs` — a list of smaller Document objects, each with the same metadata as the original

---

### 4. [notebook/embeddings.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/embeddings.py) — Embedding Engine

| | |
|---|---|
| **Purpose** | Convert text into numbers (vectors) so a computer can compare meaning mathematically |
| **Input** | A list of strings (text from chunks) |
| **What it does** | Loads a local AI model (`all-MiniLM-L6-v2`) and uses it to turn text into 384-dimensional vectors |
| **Output** | A NumPy array of shape `(N, 384)` — N vectors, each 384 numbers long |
| **Who uses its output** | `vector_store.py` uses the vectors to store; `rag_retriever.py` uses it to embed the query |

#### What is an Embedding?
Text like "Python is a programming language" becomes a list of 384 floating-point numbers. Two similar sentences will produce vectors that are mathematically *close* to each other. This is how semantic (meaning-based) search works.

#### The Model: `all-MiniLM-L6-v2`
- A lightweight, fast model from HuggingFace/SBERT
- Runs **completely offline** from `models/models/all-MiniLM-L6-v2/`
- Produces 384-dimension embeddings
- Great balance of speed and accuracy

#### Class: `EmbeddingManager`

**`__init__(self, model_name="all-MiniLM-L6-v2")`**
- **Purpose**: Set up the manager, trigger model loading
- **Input**: Optional model name
- **Process**: Sets attributes, calls `_load_model()`
- **Return**: Initialised object

**`_load_model(self)`**
- **Purpose**: Actually load the SentenceTransformer model from local disk
- **Input**: Uses `MODEL_PATH` (resolved from the project folder structure)
- **Process**: Calls `SentenceTransformer(MODEL_PATH)`, prints embedding dimension (384)
- **Return**: None (sets `self.model`)

**`generate_embeddings(self, texts: List[str]) → np.ndarray`**
- **Purpose**: Turn a list of text strings into vectors
- **Input**: A list of strings
- **Process**: Calls `self.model.encode(texts, show_progress_bar=True)` — this runs the neural network on every string
- **Return**: NumPy array of shape `(len(texts), 384)`

#### Module-level (runs on import):
```python
embedding_manager = EmbeddingManager()
```
This creates one shared instance so all other files can reuse the same loaded model.

---

### 5. [notebook/vector_store.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/vector_store.py) — Vector Database Manager

| | |
|---|---|
| **Purpose** | Save text + its vector into ChromaDB, a vector database that can be searched by similarity |
| **Input** | Chunks (from `chunks.py`) + embeddings (from `embeddings.py`) |
| **What it does** | Orchestrates the full ingestion pipeline: split → embed → store |
| **Output** | Persisted ChromaDB collection on disk at `data/vector_store/` |
| **Who uses its output** | `rag_retriever.py` connects to the same ChromaDB to search it |

#### What is ChromaDB?
A vector database — like a regular database but instead of searching by exact ID or name, you search by *similarity*. "Find me the 5 most similar chunks to this question."

#### Class: `VectorStore`

**`__init__(self, collection_name="pdf_documents", persist_directory=DEFAULT_PERSIST_DIR)`**
- **Purpose**: Initialise the store, create folder if needed
- **Input**: Collection name and path
- **Process**: Calls `_initialize_store()`
- **Return**: Initialised object

**`_initialize_store(self)`**
- **Purpose**: Connect to ChromaDB and get (or create) a collection
- **Input**: Uses `self.persist_directory`
- **Process**:
  - Creates the folder with `os.makedirs(..., exist_ok=True)`
  - Creates a `PersistentClient` — this means data survives after the program ends
  - Calls `get_or_create_collection()` — if the collection already exists, it reuses it; otherwise creates fresh
  - Prints how many documents are already in the collection
- **Return**: None (sets `self.client` and `self.collection`)

**`add_documents(self, documents: List[Any], embeddings: np.ndarray)`**
- **Purpose**: Insert chunks + their vectors into the database
- **Input**: List of Document objects and their corresponding NumPy embeddings
- **Process**:
  - Validates that the count of documents equals count of embeddings
  - For each document:
    - Generates a unique ID using UUID: `doc_a3f1c2b8_0`
    - Copies the document's metadata and adds `doc_index` and `content_length`
    - Extracts `page_content` as plain text
    - Converts the NumPy vector to a plain Python list (ChromaDB requirement)
  - Calls `self.collection.add(ids, embeddings, metadatas, documents)` to bulk-insert
  - Prints success and total document count
- **Return**: None

#### Module-level pipeline (runs on import):
```python
vector_store = VectorStore()          # Connect to ChromaDB
chunks = split_documents(documents_pdf) # Step 1: Split
embeddings = embedding_manager.generate_embeddings([doc.page_content for doc in chunks])  # Step 2: Embed
vector_store.add_documents(chunks, embeddings)  # Step 3: Store
```

---

### 6. [notebook/rag_retriever.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/rag_retriever.py) — Search Engine

| | |
|---|---|
| **Purpose** | Given a user question, find the most relevant stored chunks |
| **Input** | A user query string |
| **What it does** | Converts the question to a vector, asks ChromaDB to find the closest vectors |
| **Output** | A list of dictionaries, each containing: matched text, metadata, similarity score, rank |
| **Who uses its output** | `llm_output.py` and `rag_advanced_class.py` use the retrieved chunks as context |

#### Class: `RAGRetriever`

**`__init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager)`**
- **Purpose**: Wire together the database and the embedding model
- **Input**: An existing `VectorStore` object and `EmbeddingManager` object
- **Process**: Saves both as instance attributes
- **Return**: Initialised object

**`retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) → List[Dict]`**
- **Purpose**: Perform semantic search
- **Input**:
  - `query` — the user's question in plain English
  - `top_k` — how many results to return (default 5)
  - `score_threshold` — minimum similarity score (0.0 to 1.0, default 0.0 means no filter)
- **Process**:
  1. Converts the query to a vector: `embedding_manager.generate_embeddings([query])[0]`
  2. Asks ChromaDB: `collection.query(query_embeddings=[...], n_results=top_k)`
  3. ChromaDB returns documents, metadatas, distances, and IDs
  4. Converts ChromaDB's "distance" to a "similarity score": `similarity = 1 - distance`
     - Distance 0 = identical → similarity 1.0
     - Distance 1 = completely different → similarity 0.0
  5. Filters out results below `score_threshold`
  6. Packages each result into a clean dictionary with id, content, metadata, score, and rank
- **Return**: `List[Dict]` — each item is one matched chunk with its score

#### Module-level (runs on import):
```python
rag_retriever = RAGRetriever(vector_store, embedding_manager)
results = rag_retriever.retrieve("What is Python Programming Language ?")
```
This is a test run that demonstrates the retrieval works.

---

### 7. [notebook/llm_output.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/llm_output.py) — Answer Generator

| | |
|---|---|
| **Purpose** | Take the retrieved chunks and use Gemini AI to produce a human-readable answer |
| **Input** | User query, the retriever object, the LLM object |
| **What it does** | Builds a prompt with the context from retrieved chunks and sends it to Gemini Flash |
| **Output** | A string answer (either from documents or from LLM general knowledge as fallback) |
| **Who uses its output** | `rag_advanced_class.py` imports the `llm` object; end user reads the answer |

#### How Gemini is set up:
```python
load_dotenv()  # reads .env file
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=..., max_output_tokens=200)
```
The API key is read from the `.env` file — never hardcoded.

#### Functions in execution order:

**`rag_simple(query, retriever, llm, top_k=3)`**
- **Purpose**: Simple, straightforward RAG — retrieve context, ask Gemini, return answer
- **Input**: User's query string, retriever object, LLM object, number of results
- **Process**:
  1. Calls `retriever.retrieve(query, top_k=top_k)` to get relevant chunks
  2. Joins all chunk contents with blank lines as `context`
  3. If context was found: builds a grounded prompt — *"Answer using ONLY this context"*
  4. If no context: falls back to Gemini's general knowledge
  5. Calls `llm.invoke([prompt])` and reads `response.content`
- **Return**:
  - `"[From Docs]\n<answer>"` if documents matched
  - `"[LLM Answer - No Docs]\n<answer>"` if no documents matched

**`rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False)`**
- **Purpose**: Production-grade RAG with source attribution and confidence scoring
- **Input**: Query, retriever, LLM, top_k, minimum score threshold, flag to include context in output
- **Process**:
  1. Retrieves documents with score filtering (`score_threshold=min_score`)
  2. If no results: returns a structured dict with empty fields and confidence 0
  3. Builds context from all matched chunks
  4. Builds a `sources` list — each source includes filename, page number, similarity score, and 120-char preview
  5. Calculates `confidence` = the highest similarity score among retrieved results
  6. Sends the grounded prompt to Gemini
  7. Optionally appends the full context to the result
- **Return**: A dictionary with `answer`, `sources`, `confidence`, and optionally `context`

---

### 8. [notebook/rag_advanced_class.py](file:///c:/Users/Admin/OneDrive/Projects/RAG%20Pipeline/notebook/rag_advanced_class.py) — Advanced Pipeline (Final Layer)

| | |
|---|---|
| **Purpose** | A complete, reusable RAG system with streaming, citations, query history, and summarization |
| **Input** | A question string + configuration flags |
| **What it does** | Combines retrieval + LLM answering + citations + optional summary + history tracking |
| **Output** | A rich dictionary: answer with citations, sources, summary, and full query history |
| **Who uses its output** | The end user / calling application |

#### Class: `AdvancedRAGPipeline`

**`__init__(self, retriever, llm)`**
- **Purpose**: Wire together the retriever and LLM; initialise an empty history list
- **Input**: `RAGRetriever` object, LLM object
- **Process**: Saves both + sets `self.history = []`
- **Return**: Initialised object

**`query(self, question, top_k=5, min_score=0.2, stream=False, summarize=False) → Dict`**
- **Purpose**: The main method — answers a question end-to-end
- **Input**:
  - `question` — user's question
  - `top_k` — number of chunks to retrieve
  - `min_score` — minimum similarity threshold (chunks below this are ignored)
  - `stream` — if True, simulate streaming by printing the prompt in 80-char chunks with a delay
  - `summarize` — if True, ask Gemini to create a 2-sentence summary of the answer
- **Process**:
  1. **Retrieve**: `retriever.retrieve(question, top_k, score_threshold=min_score)`
  2. **Build context**: joins chunk text
  3. **Build sources list**: filename, page, score, 120-char preview
  4. **Optional streaming**: prints prompt character-by-character with `time.sleep(0.05)` between each 80-char block
  5. **Generate answer**: `llm.invoke([prompt])`
  6. **Add citations**: appends `[1] filename (page N)` to the end of the answer
  7. **Optional summarize**: sends the answer back to Gemini with a "summarize in 2 sentences" prompt
  8. **Store history**: appends question + answer + sources + summary to `self.history`
- **Return**: Dictionary with `question`, `answer` (with citations), `sources`, `summary`, `history`

#### Module-level demo (runs on import):
```python
adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
result = adv_rag.query("what is Python programming", top_k=3, min_score=0.1, stream=True, summarize=True)
print(result['answer'])
print(result['summary'])
```

---

## 🔗 Module Relationship Map

```
document.py
    │ (documents_pdf)
    ▼
vector_store.py ──imports──► chunks.py (split_documents)
    │               └──imports──► embeddings.py (EmbeddingManager, embedding_manager)
    │ (VectorStore, vector_store)
    ▼
rag_retriever.py ──imports──► embeddings.py (EmbeddingManager, embedding_manager)
    │ (RAGRetriever, rag_retriever)
    ▼
llm_output.py ──imports──► rag_retriever.py (rag_retriever)
    │ (llm, rag_simple, rag_advanced)
    ▼
rag_advanced_class.py ──imports──► rag_retriever.py (rag_retriever)
                        └──imports──► llm_output.py (llm)
```

---

## 🌊 Complete Data Flow

| Stage | Data Form | Description |
|---|---|---|
| Raw files | `.pdf`, `.txt` | Your source documents on disk |
| After `document.py` | `List[Document]` | LangChain Document objects with text + metadata |
| After `chunks.py` | `List[Document]` (more, smaller) | Same structure, just split into ~1000-char pieces |
| After `embeddings.py` | `np.ndarray (N, 384)` | Each chunk → 384 floating point numbers |
| After `vector_store.py` | ChromaDB collection on disk | Numbers + original text saved persistently |
| After `rag_retriever.py` | `List[Dict]` | Top-K most relevant chunks for the query |
| After `llm_output.py` | `str` or `Dict` | Final human-readable answer from Gemini |

---

## 🛠️ Technologies Used

| Technology | What it is | Why it's used here |
|---|---|---|
| **LangChain** | A framework for building AI apps | Provides loaders, splitters, and document abstractions |
| **PyMuPDF** | A Python library to read PDFs | Fast, reliable PDF text extraction |
| **SentenceTransformers** | A library for embedding models | Converts text to vectors locally (no API needed) |
| **all-MiniLM-L6-v2** | A small but effective embedding model | Fast, accurate, runs offline — good for local RAG |
| **ChromaDB** | A vector database | Stores vectors + text, supports similarity search |
| **Gemini 2.5 Flash** | Google's AI model | Generates final natural language answers |
| **python-dotenv** | Environment variable loader | Keeps API keys out of source code |
| **NumPy** | Numerical computing library | Handles vector arrays efficiently |
| **uv** | Python package manager | Manages dependencies via `pyproject.toml` |

---

## 📌 Project Summary

This project is a **local Retrieval-Augmented Generation (RAG) pipeline** built from scratch in Python. It allows users to ask natural language questions about their own PDF documents and receive accurate, source-cited answers powered by Google's Gemini AI.

**The pipeline has two phases:**

1. **Ingestion** (one-time, offline): Load PDFs → Split into chunks → Convert to vectors → Save to ChromaDB
2. **Querying** (on every question): Embed the question → Find similar chunks → Send to Gemini → Return cited answer

The project is structured as a progressive learning journey — starting with basic document loading, then building up to a full-featured pipeline with streaming, citation tracking, confidence scores, and query history. The embedding model runs completely offline, keeping costs low, while Gemini handles final answer generation via API.

---

## ❓ Common Questions & Clear Answers

**Q: What is RAG and why is it useful?**
A: RAG stands for Retrieval-Augmented Generation. Instead of relying only on an AI's training data, RAG first searches your own documents for relevant information, then lets the AI answer using that information. This makes answers more accurate and relevant to your specific data.

---

**Q: Why split documents into chunks?**
A: Embedding models can only process a limited amount of text at a time. Smaller chunks also improve precision — when you ask about a specific topic, you retrieve only the relevant paragraph instead of an entire 50-page document.

---

**Q: What is chunk overlap and why does it matter?**
A: The 200-character overlap means consecutive chunks share some text. This prevents important sentences from being cut in half at a chunk boundary, preserving context across chunk edges.

---

**Q: Why use a local embedding model instead of an API?**
A: The model `all-MiniLM-L6-v2` runs on your own machine. No API calls, no costs, no internet needed for the ingestion step. It's fast and produces high-quality 384-dimensional vectors.

---

**Q: What is a vector/embedding?**
A: A list of 384 numbers that represents the *meaning* of a piece of text. Two sentences with similar meanings will produce vectors that are mathematically close to each other, which is how semantic search works.

---

**Q: What is ChromaDB and why is it used?**
A: ChromaDB is a vector database. Unlike a traditional database that finds exact matches, ChromaDB finds *similar* vectors. It's used here to store all chunk embeddings and quickly find the most relevant chunks for any query. It saves data to disk (`data/vector_store/`) so you don't have to re-process PDFs every time.

---

**Q: What does `similarity_score = 1 - distance` mean?**
A: ChromaDB returns a "distance" — how far apart two vectors are. Distance 0 means identical; higher = more different. By doing `1 - distance`, we flip it into a "similarity score" where 1.0 means identical and 0.0 means completely unrelated. This is more intuitive.

---

**Q: What is the difference between `rag_simple` and `rag_advanced`?**
A: `rag_simple` is a quick, no-frills version — retrieves chunks, sends to Gemini, returns answer. `rag_advanced` adds source attribution (which file + page each chunk came from), confidence scoring (best similarity score), and optional full context return. `AdvancedRAGPipeline` goes further with streaming simulation, citation formatting, query history, and optional summarization.

---

**Q: What happens if no relevant documents are found?**
A: `rag_simple` falls back to Gemini's general knowledge (and labels the answer as `[LLM Answer - No Docs]`). `rag_advanced` returns a structured response with `"No relevant context found."` and zero confidence. `AdvancedRAGPipeline` does the same — no hallucination from missing context.

---

**Q: What is the confidence score?**
A: It's the highest similarity score among all retrieved chunks for a given query. If the best-matching chunk has a similarity of 0.87, the confidence is 0.87. It gives a quick sense of how relevant the retrieved documents are to the question.

---

**Q: Why is `main.py` just a placeholder?**
A: The project was created using `uv` (a Python project tool) which auto-generates a basic `main.py`. The actual pipeline logic is in the `notebook/` directory, developed iteratively as individual learning modules. Wiring them into `main.py` would be the natural next step.

---

**Q: What are some potential improvements?**

| Improvement | Why it matters |
|---|---|
| Wire `main.py` to run the full pipeline | Makes the project deployable as a proper application |
| Add a query CLI or web UI (e.g., with Streamlit) | Makes it usable by non-programmers |
| Support re-ingestion detection (skip already stored docs) | Avoids adding duplicate documents to ChromaDB on every run |
| Add MMR (Maximal Marginal Relevance) retrieval | Reduces redundant chunks in results, improves answer diversity |
| Add reranking (e.g., CrossEncoder) | Two-stage retrieval: broad recall → precise rerank |
| Replace simulated streaming with real streaming | Gemini supports actual token streaming; gives better UX |
| Add metadata filters in retrieval | Let users filter by source file, page range, etc. |
| Persist query history to a file or database | Enables multi-session memory |
| Add evaluation metrics (e.g., faithfulness, relevance) | Measures how good the RAG answers actually are |
| Package as a Python module | Enables `pip install` and easier reuse |
