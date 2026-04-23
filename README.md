# DocMind — RAG Document Q&A System

> **Upload any PDF or TXT. Ask questions. Get answers grounded in your document — powered by OpenAI, Groq, or Gemini.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-0066CC?style=flat)
![License](https://img.shields.io/badge/license-MIT-green?style=flat)

---

## What it does

DocMind is a Retrieval-Augmented Generation (RAG) system that lets you have a conversation with any document. Upload a PDF or text file — a research paper, legal contract, product manual, financial report — and ask natural language questions. The system retrieves the most relevant passages and uses an LLM to generate precise, document-grounded answers.

**Three LLM providers, switchable from the sidebar:**

| Provider | Cost | Model options |
|----------|------|---------------|
| **OpenAI** | ~$0.001/question | gpt-4o-mini, gpt-4o, gpt-3.5-turbo |
| **Groq** | Free tier | llama3-8b-8192, llama3-70b-8192, mixtral |
| **Gemini** | Free tier (1500 req/day) | gemini-1.5-flash, gemini-1.5-pro |

**Key design:** Embeddings always run locally using `sentence-transformers/all-MiniLM-L6-v2` — free, fast, no API call. Only the answer generation step hits the LLM API, minimizing cost.

---

## Live Demo

```bash
streamlit run app.py
# → http://localhost:8501
```

1. Enter your API key in the sidebar (OpenAI, Groq, or Gemini)
2. Upload a PDF or TXT file
3. Ask any question about the document

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RAG Pipeline                        │
│                                                         │
│  PDF/TXT → [Text Extraction] → [Chunking]               │
│                                      ↓                  │
│                           [sentence-transformers]        │
│                           all-MiniLM-L6-v2 (local)      │
│                                      ↓                  │
│                           [FAISS Vector Index]           │
│                                                         │
│  Question → [Embed query] → [Similarity Search (top-k)] │
│                                      ↓                  │
│                           [Retrieved chunks + question] │
│                                      ↓                  │
│                           [LLM: OpenAI / Groq / Gemini] │
│                                      ↓                  │
│                           [Grounded answer + sources]   │
└─────────────────────────────────────────────────────────┘
```

### Component breakdown

**Text extraction** (`pypdf`)  
Extracts text page-by-page from PDFs. TXT files decoded directly.

**Chunking** (`RecursiveCharacterTextSplitter`)  
Splits text into overlapping chunks (default: 800 tokens, 120 overlap). Overlapping ensures no sentence is cut mid-thought. The separator hierarchy `["\n\n", "\n", ". ", " "]` respects natural paragraph and sentence boundaries.

**Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)  
A 22M-parameter bi-encoder that maps text to 384-dimensional vectors. Runs entirely locally — no API key, no cost. Cached after first download (~90MB).

**Vector store** (`FAISS`)  
Facebook AI Similarity Search — an in-memory index using cosine similarity. Retrieves top-k most semantically similar chunks to the user's question in milliseconds.

**LLM chain** (`LangChain RetrievalQA`)  
A "stuff" chain: retrieved chunks are concatenated into the context window, then passed to the LLM with a custom prompt that instructs it to answer *only from the provided context*, preventing hallucination.

---

## Project Structure

```
rag-docqa/
├── app.py              ← Streamlit UI (provider switcher, chat interface)
├── rag_engine.py       ← Core RAG logic (extraction, chunking, embedding, QA)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/joypatel-cs/rag-docqa.git
cd rag-docqa
pip install -r requirements.txt
```

### 2. Get a free API key (pick one)

**Groq (recommended — 100% free, no card):**
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up → API Keys → Create key
3. Paste in the sidebar

**OpenAI ($5 free credits on new accounts):**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create key → paste in sidebar

**Gemini (free tier, 1500 req/day):**
1. Go to [aistudio.google.com](https://aistudio.google.com/app/apikey)
2. Get API key → paste in sidebar

### 3. Run

```bash
streamlit run app.py
```

No `.env` file needed — API keys are entered directly in the UI.

---

## Key Technical Decisions

**Why FAISS over a cloud vector DB (Pinecone, Weaviate)?**  
For a portfolio project processing one document at a time, FAISS in-memory is faster to set up, free, and demonstrates the core similarity search concept without infrastructure overhead. Production systems with persistent multi-document retrieval would use a managed vector DB.

**Why `all-MiniLM-L6-v2` for embeddings?**  
It's one of the best speed/quality tradeoffs for semantic search: 22M parameters, 384-dim output, runs in ~50ms per chunk on CPU. OpenAI `text-embedding-3-small` is marginally better but costs money per token. For a document Q&A use case, the quality difference is negligible.

**Why chunk overlap?**  
Without overlap, a sentence split across two chunks loses context in both. A 120-token overlap means any 800-token window shares context with its neighbors, preventing information loss at boundaries.

**Why `chain_type="stuff"` over `map_reduce`?**  
For typical documents (< 50 pages), stuffing 4 retrieved chunks into a single prompt is faster and more coherent than map-reduce's multi-step summarization. For very long documents needing full-document synthesis, `map_reduce` or `refine` chains would be more appropriate.

---

## Usage examples

```
"Summarize this document in 5 bullet points."
"What are the key risks mentioned in section 3?"
"What methodology did the authors use?"
"Are there any financial projections? What are the numbers?"
"What does the author conclude about X?"
"Compare the findings in the introduction vs the conclusion."
```

---

## Future Work

- [ ] Multi-document support — index a folder of PDFs and query across all of them  
- [ ] Persistent vector store — save/load FAISS indexes to disk between sessions  
- [ ] Streaming responses — stream token-by-token for better UX  
- [ ] Hybrid search — combine BM25 keyword search with semantic search for better recall  
- [ ] Conversation memory — maintain multi-turn context for follow-up questions  
- [ ] Deploy to Streamlit Cloud with session-scoped API key storage  

---

## Author

**Joy Patel** — MS Computer Science, The University of Texas at Dallas  
`joypatel.cs@gmail.com` · [LinkedIn](https://linkedin.com/in/joypatel-cs) · [GitHub](https://github.com/joypatel-cs)

---

*Embeddings run locally — no API key needed for indexing. Only answer generation requires an LLM API key.*
