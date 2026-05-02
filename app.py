"""
app.py — DevDocs AI — Fully Responsive Software Documentation Assistant
"""

import os
import tempfile
import streamlit as st
from langchain_community.retrievers import BM25Retriever
from rag_pipeline import load_and_chunk, build_vectorstore, generate_answer

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="DevDocs AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Subtle grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(48,255,144,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(48,255,144,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stAppViewContainer"] > * { position: relative; z-index: 1; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { background: #010409 !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }
#MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

/* ── Top nav bar ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.5rem;
    border-bottom: 1px solid #21262d;
    background: rgba(13,17,23,0.97);
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #f0f6fc;
    white-space: nowrap;
}
.nav-logo .logo-icon {
    width: 26px; height: 26px;
    background: linear-gradient(135deg, #30ff90, #00d4ff);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.nav-logo .logo-slash { color: #30ff90; margin: 0 2px; }
.nav-pills {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}
.nav-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #8b949e;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    white-space: nowrap;
}
.nav-pill.active {
    color: #30ff90;
    border-color: rgba(48,255,144,0.4);
    background: rgba(48,255,144,0.06);
}

/* Hide nav pills on very small screens */
@media (max-width: 480px) {
    .nav-pills { display: none; }
    .nav-bar { padding: 0.75rem 1rem; }
}

/* ── Hero ── */
.hero {
    max-width: 860px;
    margin: 3rem auto 2.5rem;
    padding: 0 1.5rem;
    text-align: center;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #30ff90;
    letter-spacing: 0.08em;
    margin-bottom: 1rem;
    opacity: 0.9;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(1.8rem, 5vw, 3.2rem) !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.03em !important;
    color: #f0f6fc !important;
    margin-bottom: 1rem !important;
}
.hero h1 .hl {
    background: linear-gradient(90deg, #30ff90 0%, #00d4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    font-size: clamp(0.88rem, 2vw, 1rem) !important;
    color: #8b949e !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    max-width: 520px;
    margin: 0 auto !important;
}

@media (max-width: 480px) {
    .hero { margin: 2rem auto 1.5rem; padding: 0 1rem; }
}

/* ── Upload zone ── */
.upload-wrap {
    max-width: 640px;
    margin: 0 auto 1.5rem;
    padding: 0 1.5rem;
}
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(48,255,144,0.5) !important;
    background: rgba(48,255,144,0.03) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #30ff90 !important; }

@media (max-width: 480px) {
    .upload-wrap { padding: 0 1rem; }
}

/* ── Doc info bar ── */
.doc-bar {
    max-width: 860px;
    margin: 0 auto 1.2rem;
    padding: 0 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.doc-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.28rem 0.65rem;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #161b22;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    white-space: nowrap;
}
.doc-tag.green { color: #30ff90; border-color: rgba(48,255,144,0.35); background: rgba(48,255,144,0.06); }
.doc-tag.blue  { color: #58a6ff; border-color: rgba(88,166,255,0.35); background: rgba(88,166,255,0.06); }

@media (max-width: 480px) {
    .doc-bar { padding: 0 1rem; gap: 0.4rem; }
    .doc-tag { font-size: 0.65rem; padding: 0.22rem 0.5rem; }
}

/* ── Chat area ── */
.chat-area {
    max-width: 860px;
    margin: 0 auto;
    padding: 0 1.5rem 7rem;
}

@media (max-width: 480px) {
    .chat-area { padding: 0 0.75rem 6rem; }
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
    gap: 0.6rem !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px 10px 2px 10px !important;
    padding: 0.75rem 1rem !important;
    font-size: clamp(0.85rem, 2vw, 0.95rem) !important;
    color: #e6edf3 !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
    background: rgba(48,255,144,0.04) !important;
    border: 1px solid rgba(48,255,144,0.12) !important;
    border-radius: 2px 10px 10px 10px !important;
    padding: 0.85rem 1.1rem !important;
    font-size: clamp(0.85rem, 2vw, 0.93rem) !important;
    line-height: 1.75 !important;
}

/* Code blocks inside answers */
[data-testid="stChatMessage"] code {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    padding: 0.15rem 0.4rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.75rem, 1.8vw, 0.82rem) !important;
    color: #30ff90 !important;
    word-break: break-word !important;
}
[data-testid="stChatMessage"] pre {
    background: #010409 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    padding: 0.85rem !important;
    overflow-x: auto !important;
}
[data-testid="stChatMessage"] pre code {
    background: transparent !important;
    border: none !important;
    color: #e6edf3 !important;
    font-size: clamp(0.78rem, 1.8vw, 0.85rem) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1.5px solid #30363d !important;
    border-radius: 12px !important;
    max-width: 860px !important;
    margin: 0 auto !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(48,255,144,0.5) !important;
    box-shadow: 0 0 0 3px rgba(48,255,144,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.82rem, 2vw, 0.88rem) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3d444d !important; }

/* ── Source expander ── */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
    margin-top: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.7rem, 1.8vw, 0.76rem) !important;
    color: #8b949e !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stExpander"] summary:hover { color: #30ff90 !important; }

.src-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 3px solid #30ff90;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: clamp(0.76rem, 1.8vw, 0.83rem);
    color: #8b949e;
    line-height: 1.65;
    font-family: 'JetBrains Mono', monospace;
    word-break: break-word;
    overflow-wrap: break-word;
}
.src-label {
    font-size: clamp(0.65rem, 1.6vw, 0.7rem);
    color: #30ff90;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    font-weight: 500;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background: rgba(255,166,0,0.06) !important;
    border: 1px solid rgba(255,166,0,0.25) !important;
    border-radius: 8px !important;
    color: #e3b341 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.78rem, 1.8vw, 0.85rem) !important;
}
[data-testid="stSuccess"] div {
    background: rgba(48,255,144,0.06) !important;
    border: 1px solid rgba(48,255,144,0.25) !important;
    border-radius: 8px !important;
    color: #30ff90 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.78rem, 1.8vw, 0.85rem) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    color: #30ff90 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: clamp(0.78rem, 1.8vw, 0.85rem) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #30ff90; }

/* ── Mobile touch improvements ── */
@media (max-width: 768px) {
    [data-testid="stChatMessage"] {
        gap: 0.4rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown {
        padding: 0.65rem 0.85rem !important;
    }
}

/* ── Tablet adjustments ── */
@media (min-width: 481px) and (max-width: 1024px) {
    .hero { margin: 2.5rem auto 2rem; }
    .nav-pill { font-size: 0.65rem; padding: 0.22rem 0.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Nav bar ──
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        <div class="logo-icon">⚡</div>
        Dev<span class="logo-slash">/</span>Docs&nbsp;AI
    </div>
    <div class="nav-pills">
        <div class="nav-pill active">chat</div>
        <div class="nav-pill">hybrid retrieval</div>
        <div class="nav-pill">crossencoder</div>
        <div class="nav-pill">llama-3.3-70b</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">// technical documentation assistant</div>
    <h1>Query your <span class="hl">docs</span><br>like a developer</h1>
    <p>Drop in any technical PDF — API references, architecture guides, CS textbooks —
    and get precise, grounded answers with source citations.</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──
for key, val in [
    ("vectorstore", None), ("bm25_retriever", None),
    ("messages", []), ("doc_name", None), ("chunk_count", 0)
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Upload ──
st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "upload",
    type="pdf",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and uploaded_file.name != st.session_state.doc_name:
    with st.spinner("$ indexing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        chunks = load_and_chunk(file_path)
        vectorstore = build_vectorstore(chunks)
        st.session_state.vectorstore = vectorstore

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        st.session_state.bm25_retriever = bm25_retriever
        st.session_state.doc_name = uploaded_file.name
        st.session_state.chunk_count = len(chunks)
        st.session_state.messages = []

    st.success(f"✓ indexed {uploaded_file.name}")

# ── Doc bar ──
if st.session_state.doc_name:
    # Truncate long filenames on mobile
    fname = st.session_state.doc_name
    display_name = fname if len(fname) <= 30 else fname[:27] + "..."
    st.markdown(f"""
    <div class="doc-bar">
        <div class="doc-tag green">● ready</div>
        <div class="doc-tag">📄 {display_name}</div>
        <div class="doc-tag blue">{st.session_state.chunk_count} chunks</div>
        <div class="doc-tag">bm25 + vector</div>
        <div class="doc-tag">reranked</div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat ──
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"// {len(msg['sources'])} source chunk(s)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    <div class="src-card">
                        <div class="src-label">src_{i} · page {src['page']}</div>
                        {src['text']}
                    </div>
                    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Input ──
placeholder = "// ask a technical question..." if st.session_state.doc_name else "// upload a PDF to get started"
query = st.chat_input(placeholder)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            st.warning("// no document loaded. upload a PDF first.")
        else:
            with st.spinner("// retrieving · reranking · generating..."):
                answer, docs, below_threshold = generate_answer(
                    query,
                    st.session_state.vectorstore,
                    st.session_state.bm25_retriever
                )

            if below_threshold:
                st.warning("// not found in documentation.")

            st.markdown(answer)

            sources = []
            if docs:
                with st.expander(f"// {len(docs)} source chunk(s)"):
                    for i, doc in enumerate(docs, 1):
                        page = doc.metadata.get("page", "?")
                        text = doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else "")
                        sources.append({"page": page, "text": text})
                        st.markdown(f"""
                        <div class="src-card">
                            <div class="src-label">src_{i} · page {page}</div>
                            {text}
                        </div>
                        """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })