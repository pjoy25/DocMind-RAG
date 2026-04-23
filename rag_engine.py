"""
rag_engine.py — Provider-agnostic RAG pipeline
Author: Joy Patel — MS Computer Science, UT Dallas

Supports 3 LLM providers: OpenAI · Groq · Google Gemini
Embeddings: HuggingFace all-MiniLM-L6-v2 (free, local) OR OpenAI (fallback)
"""

import io
import pypdf
from typing import Tuple, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS


# ── Embeddings ─────────────────────────────────────────────
def _get_embeddings(openai_api_key: str = None):
    """
    Primary:  HuggingFace all-MiniLM-L6-v2 — free, runs locally, no API key.
    Fallback: OpenAI text-embedding-3-small — if HF model not cached yet.
    On first run, HF will download ~90MB model and cache it permanently.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        # Fallback to OpenAI embeddings if HF unavailable
        if openai_api_key:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=openai_api_key,
            )
        raise RuntimeError(
            "Could not load embeddings. Either:\n"
            "1. Connect to internet so HuggingFace model can download (~90MB, one-time), OR\n"
            "2. Use OpenAI as provider — its API key will also cover embeddings."
        )


# ── LLM providers ──────────────────────────────────────────
def get_llm(provider: str, api_key: str, model: str = None):
    """Return the appropriate LangChain chat model for the chosen provider."""
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key,
            temperature=0.2,
            max_tokens=1024,
        )
    elif provider == "Groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama3-8b-8192",
            api_key=api_key,
            temperature=0.2,
            max_tokens=1024,
        )
    elif provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.2,
        )
    raise ValueError(f"Unknown provider: {provider}")


# ── Document extraction ────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """Extract full text from PDF bytes. Returns (text, page_count)."""
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n\n".join(pages), len(reader.pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


# ── Vector store ───────────────────────────────────────────
def build_vectorstore(text: str, chunk_size: int = 800,
                      chunk_overlap: int = 120, openai_api_key: str = None):
    """
    Split text into overlapping chunks, embed with sentence-transformers,
    and build a FAISS in-memory index for fast similarity search.

    chunk_size:    tokens per chunk (800 is a good default for most docs)
    chunk_overlap: overlap between adjacent chunks to avoid boundary cuts
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs   = [Document(page_content=c, metadata={"chunk_id": i})
              for i, c in enumerate(chunks)]
    embeddings  = _get_embeddings(openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore, len(chunks)


# ── RAG prompt ─────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    template="""You are an expert document analyst. Use ONLY the context provided below to answer the question.
If the answer is not contained in the context, respond with: "I couldn't find that in the document."
Be concise, specific, and reference details from the context where relevant.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


# ── Question answering ─────────────────────────────────────
def ask_question(question: str, vectorstore, provider: str,
                 api_key: str, model: str = None, k: int = 4) -> Dict:
    """
    1. Embed the question
    2. Retrieve top-k semantically similar chunks from FAISS
    3. Pass chunks + question to LLM with a grounding prompt
    4. Return answer + source chunks for display
    """
    llm = get_llm(provider, api_key, model)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )
    result = chain.invoke({"query": question})
    return {
        "answer":          result["result"].strip(),
        "source_chunks":   [doc.page_content[:300] + "..."
                            for doc in result["source_documents"]],
        "provider":        provider,
        "model":           model or _default_model(provider),
        "chunks_retrieved": len(result["source_documents"]),
    }


def _default_model(provider: str) -> str:
    return {
        "OpenAI": "gpt-4o-mini",
        "Groq":   "llama3-8b-8192",
        "Gemini": "gemini-1.5-flash",
    }.get(provider, "")


# ── Provider metadata (used by the UI) ────────────────────
PROVIDERS = {
    "OpenAI": {
        "models":    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "key_url":   "https://platform.openai.com/api-keys",
        "cost":      "~$0.001 / question",
        "free_tier": False,
        "color":     "#10A37F",
        "note":      "Best quality. New accounts get $5 free credits.",
    },
    "Groq": {
        "models":    ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        "key_url":   "https://console.groq.com/keys",
        "cost":      "Free tier (generous daily limits)",
        "free_tier": True,
        "color":     "#F55036",
        "note":      "100% free. No credit card needed. Very fast.",
    },
    "Gemini": {
        "models":    ["gemini-1.5-flash", "gemini-1.5-pro"],
        "key_url":   "https://aistudio.google.com/app/apikey",
        "cost":      "Free tier (1500 req/day)",
        "free_tier": True,
        "color":     "#4285F4",
        "note":      "Free tier very generous. Get key from Google AI Studio.",
    },
}
