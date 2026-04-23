"""
DocMind — RAG Document Q&A System
Author: Joy Patel — MS Computer Science, UT Dallas

Upload any PDF or TXT. Ask questions. Get answers grounded in your document.
Switch between OpenAI, Groq, and Gemini from the sidebar.
"""

import streamlit as st
import time
import os
from rag_engine import (
    extract_text_from_pdf, extract_text_from_txt,
    build_vectorstore, ask_question, PROVIDERS,
)

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp { background: #0A0C10; }

[data-testid="stSidebar"] {
    background: #0D0F14;
    border-right: 1px solid #1A1D25;
}

/* Chat bubbles */
.msg-user {
    background: #151820;
    border: 1px solid #1E2230;
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.5rem 3rem;
    font-size: 14px;
    color: #C8CCE0;
    line-height: 1.6;
}
.msg-bot {
    background: #0F1520;
    border: 1px solid #1A2540;
    border-left: 3px solid #4A90D9;
    border-radius: 4px 12px 12px 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 3rem 0.5rem 0;
    font-size: 14px;
    color: #C8CCE0;
    line-height: 1.7;
}
.msg-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.msg-label-user { color: #5A5F80; }
.msg-label-bot  { color: #4A90D9; }

/* Source chunks */
.source-block {
    background: #080C12;
    border: 1px solid #161C28;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #4A5070;
    margin-top: 6px;
    line-height: 1.5;
}

/* Provider badge */
.provider-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.07em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
}

/* Doc info card */
.doc-card {
    background: #0F1218;
    border: 1px solid #1A1D25;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 13px;
    color: #8A8FA8;
}
.doc-card strong { color: #C8CCE0; }

/* Upload area hint */
.upload-hint {
    text-align: center;
    padding: 3rem 2rem;
    color: #3A3F55;
    font-size: 14px;
    border: 1px dashed #1E2230;
    border-radius: 16px;
    margin: 1rem 0;
}
.upload-hint h3 { color: #5A5F80; font-size: 1.1rem; margin-bottom: 0.5rem; }

/* Typing indicator */
.typing { color: #4A90D9; font-size: 13px; font-family: 'IBM Plex Mono', monospace; }

/* Stats row */
.stat-pill {
    display: inline-block;
    background: #0F1218;
    border: 1px solid #1A1D25;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    color: #5A5F80;
    margin-right: 6px;
    font-family: 'IBM Plex Mono', monospace;
}

[data-testid="metric-container"] {
    background: #0F1218;
    border: 1px solid #1A1D25;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "doc_meta"    not in st.session_state: st.session_state.doc_meta    = None
if "total_questions" not in st.session_state: st.session_state.total_questions = 0


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 DocMind")
    st.markdown('<p style="color:#3A3F55;font-size:12px;margin-top:-0.5rem;">RAG Document Q&A · Multi-Provider</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Provider selector
    st.markdown("**LLM Provider**")
    provider = st.selectbox(
        "Provider", list(PROVIDERS.keys()), label_visibility="collapsed"
    )
    pinfo = PROVIDERS[provider]

    # Provider info
    free_tag = "🟢 Free" if pinfo["free_tier"] else "💳 Paid"
    st.markdown(
        f'<div style="background:#0F1218;border:1px solid #1A1D25;border-radius:8px;'
        f'padding:0.7rem 0.9rem;font-size:12px;color:#8A8FA8;margin-bottom:0.75rem;">'
        f'<strong style="color:{pinfo["color"]};">{provider}</strong> &nbsp; {free_tag}<br>'
        f'{pinfo["note"]}<br>'
        f'<a href="{pinfo["key_url"]}" target="_blank" style="color:#4A90D9;font-size:11px;">Get API key →</a>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # API key input
    api_key = st.text_input(
        f"{provider} API Key",
        type="password",
        placeholder=f"Paste your {provider} key here...",
        help=f"Get your key at {pinfo['key_url']}"
    )

    # Model selector
    model = st.selectbox("Model", pinfo["models"])

    st.markdown("---")

    # Document upload
    st.markdown("**Upload Document**")
    uploaded = st.file_uploader(
        "PDF or TXT", type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Max ~50 pages for best performance"
    )

    # Chunk settings (advanced)
    with st.expander("⚙ Advanced settings"):
        chunk_size    = st.slider("Chunk size (tokens)", 400, 1500, 800, step=50)
        chunk_overlap = st.slider("Chunk overlap", 50, 300, 120, step=10)
        top_k         = st.slider("Chunks retrieved (k)", 2, 8, 4)
        show_sources  = st.checkbox("Show source chunks", value=True)

    st.markdown("---")

    # Clear chat
    if st.button("🗑  Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_questions = 0
        st.rerun()

    # Reset document
    if st.button("📄  Load new document", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.doc_meta    = None
        st.session_state.messages    = []
        st.session_state.total_questions = 0
        st.rerun()


# ── Process uploaded document ──────────────────────────────
if uploaded and st.session_state.vectorstore is None:
    with st.spinner("📖 Reading document..."):
        file_bytes = uploaded.read()
        if uploaded.type == "application/pdf":
            raw_text, page_count = extract_text_from_pdf(file_bytes)
            file_type = "PDF"
        else:
            raw_text   = extract_text_from_txt(file_bytes)
            page_count = raw_text.count("\n\n") + 1
            file_type  = "TXT"

    with st.spinner("🔢 Building vector index..."):
        vectorstore, n_chunks = build_vectorstore(raw_text, chunk_size, chunk_overlap)
        st.session_state.vectorstore = vectorstore
        st.session_state.doc_meta = {
            "name":       uploaded.name,
            "type":       file_type,
            "pages":      page_count,
            "chunks":     n_chunks,
            "char_count": len(raw_text),
            "words":      len(raw_text.split()),
        }
        st.session_state.messages = []
        st.success(f"✓ Indexed {n_chunks} chunks from {uploaded.name}")


# ── Main content ───────────────────────────────────────────
col_main, col_info = st.columns([2.2, 1], gap="large")

with col_main:
    st.markdown("## DocMind &nbsp;·&nbsp; Ask Your Documents")
    st.markdown(
        '<p style="color:#3A3F55;font-size:13px;margin-top:-0.5rem;">'
        'RAG pipeline · FAISS vector search · sentence-transformers embeddings · switchable LLM</p>',
        unsafe_allow_html=True
    )

    # Doc info strip
    if st.session_state.doc_meta:
        m = st.session_state.doc_meta
        st.markdown(
            f'<span class="stat-pill">📄 {m["name"]}</span>'
            f'<span class="stat-pill">{m["pages"]} pages</span>'
            f'<span class="stat-pill">{m["chunks"]} chunks</span>'
            f'<span class="stat-pill">{m["words"]:,} words</span>'
            f'<span class="stat-pill" style="color:{PROVIDERS[provider]["color"]};">{provider} · {model}</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # Empty state
    if st.session_state.vectorstore is None:
        st.markdown("""
        <div class="upload-hint">
            <h3>Upload a document to get started</h3>
            <p>Supports PDF and TXT files · Upload from the sidebar<br>
            Then ask anything — summaries, key facts, comparisons, explanations</p>
        </div>
        """, unsafe_allow_html=True)

        # Show example questions
        st.markdown("**Example questions you can ask:**")
        examples = [
            "What is the main argument of this document?",
            "Summarize the key findings in 3 bullet points.",
            "What does the author recommend?",
            "Are there any limitations mentioned?",
            "What methodology was used?",
        ]
        for ex in examples:
            st.markdown(f'<span class="stat-pill" style="cursor:pointer;">💬 {ex}</span>', unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="msg-label msg-label-user">You</div>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            provider_color = PROVIDERS.get(msg.get("provider",""), {}).get("color","#4A90D9")
            provider_tag   = f'<span class="provider-badge" style="background:{provider_color}22;color:{provider_color};">{msg.get("provider","AI")} · {msg.get("model","")}</span>'
            st.markdown(
                f'<div class="msg-bot"><div class="msg-label msg-label-bot">DocMind {provider_tag}</div>'
                f'{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if show_sources and msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source chunks retrieved"):
                    for i, chunk in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="source-block"><strong style="color:#5A5F80;">chunk {i}</strong><br>{chunk}</div>',
                            unsafe_allow_html=True
                        )

    # Input
    if st.session_state.vectorstore is not None:
        question = st.chat_input("Ask anything about your document...")

        if question:
            if not api_key:
                st.error(f"Please enter your {provider} API key in the sidebar.")
                st.stop()

            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})

            # Generate answer
            with st.spinner(f"Thinking with {provider} ({model})..."):
                try:
                    t0     = time.time()
                    result = ask_question(
                        question    = question,
                        vectorstore = st.session_state.vectorstore,
                        provider    = provider,
                        api_key     = api_key,
                        model       = model,
                        k           = top_k,
                    )
                    elapsed = round(time.time() - t0, 2)

                    st.session_state.messages.append({
                        "role":     "assistant",
                        "content":  result["answer"],
                        "sources":  result["source_chunks"],
                        "provider": provider,
                        "model":    model,
                        "elapsed":  elapsed,
                    })
                    st.session_state.total_questions += 1

                except Exception as e:
                    err = str(e)
                    if "api_key" in err.lower() or "401" in err or "authentication" in err.lower():
                        st.error(f"Invalid API key for {provider}. Check your key and try again.")
                    elif "rate" in err.lower():
                        st.error("Rate limit hit. Wait a moment and try again.")
                    else:
                        st.error(f"Error: {err}")

            st.rerun()


# ── Right panel: stats + architecture ─────────────────────
with col_info:
    st.markdown("#### Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Questions", st.session_state.total_questions)
    with c2:
        chunks = st.session_state.doc_meta["chunks"] if st.session_state.doc_meta else 0
        st.metric("Chunks indexed", chunks)

    st.markdown("---")
    st.markdown("#### How RAG works")
    st.markdown("""
<div style="font-size:12px;color:#5A5F80;line-height:1.8;">

<span style="color:#4A90D9;">① Ingestion</span><br>
Document → split into overlapping chunks → embedded with <code style="font-size:11px;">all-MiniLM-L6-v2</code> → stored in FAISS index

<br><br>
<span style="color:#4A90D9;">② Retrieval</span><br>
Question → embedded → cosine similarity search → top-k most relevant chunks returned

<br><br>
<span style="color:#4A90D9;">③ Generation</span><br>
Retrieved chunks + question → prompt → LLM (OpenAI / Groq / Gemini) → grounded answer

<br><br>
<span style="color:#3A3F55; font-size:11px;">Embeddings are always local (free). Only the generation step uses the LLM API.</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Provider comparison")

    for pname, pdata in PROVIDERS.items():
        active = "border-left: 2px solid " + pdata["color"] + ";" if pname == provider else ""
        st.markdown(
            f'<div style="background:#0F1218;border:1px solid #1A1D25;{active}'
            f'border-radius:8px;padding:0.6rem 0.8rem;margin-bottom:6px;">'
            f'<strong style="color:{pdata["color"]};font-size:13px;">{pname}</strong>'
            f'<span style="float:right;font-size:10px;color:#3A3F55;">{"🟢 Free" if pdata["free_tier"] else "💳"}</span><br>'
            f'<span style="font-size:11px;color:#5A5F80;">{pdata["cost"]}</span><br>'
            f'<span style="font-size:10px;color:#3A3F55;">{", ".join(pdata["models"][:2])}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    if st.session_state.doc_meta:
        st.markdown("---")
        m = st.session_state.doc_meta
        st.markdown("#### Document info")
        st.markdown(
            f'<div style="font-size:12px;color:#5A5F80;line-height:2;">'
            f'<b style="color:#8A8FA8;">File:</b> {m["name"]}<br>'
            f'<b style="color:#8A8FA8;">Type:</b> {m["type"]}<br>'
            f'<b style="color:#8A8FA8;">Pages:</b> {m["pages"]}<br>'
            f'<b style="color:#8A8FA8;">Words:</b> {m["words"]:,}<br>'
            f'<b style="color:#8A8FA8;">Chunks:</b> {m["chunks"]}<br>'
            f'<b style="color:#8A8FA8;">Chunk size:</b> {chunk_size} tokens<br>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#1E2230;font-size:11px;text-align:center;">'
    'Built by Joy Patel · MS Computer Science, UT Dallas · '
    '<a href="https://github.com/joypatel-cs/rag-docqa" style="color:#2A2F45;">GitHub</a>'
    '</p>',
    unsafe_allow_html=True
)
