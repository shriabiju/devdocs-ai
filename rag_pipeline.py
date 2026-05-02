"""
rag_pipeline.py — Core RAG pipeline
Domain: Software / Technical Documentation
Uses: Groq API (llama-3.3-70b-versatile), FAISS, BM25, CrossEncoder reranker

Elite improvements:
1. Metadata-aware retrieval (page, source filename, section heading)
2. Heading-aware semantic chunking
3. Embedding + retrieval caching
4. Conversation memory (multi-turn context)
5. Multi-document support (merge multiple PDFs into one index)
"""

import os
import re
import hashlib
import functools
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from sentence_transformers import CrossEncoder
from groq import Groq

# ------------------ CONFIG ------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Confidence threshold — below this score, refuse to answer
RERANKER_SCORE_THRESHOLD = -2.0

# Max conversation turns to include as context (memory window)
MEMORY_WINDOW = 4

# ------------------ PROMPT CONFIG (versioned) ------------------
SYSTEM_PROMPT = """\
You are a senior software engineer and technical documentation assistant. \
Your job is to answer questions about software systems, code, APIs, algorithms, \
data structures, architecture, and CS concepts.

Rules:
1. Use the provided context as your primary source. Prioritize it over your own knowledge.
2. If the context covers the topic partially, answer fully using BOTH the context AND \
your own accurate technical knowledge. Clearly separate them: first state what the \
document says, then add what it omits, labeled as (general knowledge).
3. If the context is completely unrelated to the question, say: \
"This doesn't appear to be covered in the provided documentation."
4. Structure answers clearly: definition first, then types/details/variants, \
then a code example only if present in the context.
5. Use bullet points or numbered lists when enumerating types, states, or steps.
6. Cite inline as (Source 1, p.X) including page number for context claims.
7. Use conversation history to understand follow-up questions — maintain context across turns.
8. Be concise but complete.\
"""

ANSWER_PROMPT_TEMPLATE = """\
Technical documentation context:

{context}

{history_section}Developer question: {query}

Answer using the context as primary source. Maintain awareness of the conversation \
history for follow-up questions. If context only partially covers the topic, \
supplement with accurate technical knowledge and clearly label each part:\
"""

# ------------------ INIT ------------------
_reranker = CrossEncoder(RERANKER_MODEL)

# Cache embeddings model — loaded once, reused forever
@functools.lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY))


# ------------------ IMPROVEMENT 2: HEADING-AWARE CHUNKING ------------------
def _detect_section_heading(text: str) -> str:
    """Extract likely section heading from the start of a chunk."""
    lines = text.strip().split("\n")
    for line in lines[:4]:
        line = line.strip()
        # Heading patterns: ALL CAPS, Title Case short line, numbered sections
        if (
            len(line) > 3 and len(line) < 80
            and (
                line.isupper()
                or re.match(r"^(\d+[\.\)]|•|-)\s+[A-Z]", line)
                or (line.istitle() and len(line.split()) <= 8)
            )
        ):
            return line
    return ""


def load_and_chunk(pdf_path: str, source_name: str = None) -> list:
    """
    Load PDF, split with heading-aware chunking, enrich metadata.
    Improvement 1: metadata includes page, source, section heading
    Improvement 2: heading-aware splitting with smaller secondary chunks
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    source_label = source_name or os.path.basename(pdf_path)

    # Primary splitter — larger chunks for context richness
    primary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Secondary splitter — smaller chunks for precision retrieval
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    primary_chunks = primary_splitter.split_documents(documents)
    secondary_chunks = secondary_splitter.split_documents(documents)

    all_chunks = primary_chunks + secondary_chunks

    filtered = []
    seen_hashes = set()

    for chunk in all_chunks:
        text = chunk.page_content.strip()
        word_count = len(set(text.lower().split()))

        if len(text) < 120 or word_count < 30:
            continue

        # Deduplicate by content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        # Improvement 1: enrich metadata
        section = _detect_section_heading(text)
        chunk.metadata["source"] = source_label
        chunk.metadata["section"] = section
        chunk.metadata["page"] = chunk.metadata.get("page", "?")
        chunk.metadata["content_hash"] = content_hash

        filtered.append(chunk)

    return filtered


# ------------------ IMPROVEMENT 5: MULTI-DOCUMENT SUPPORT ------------------
def load_multiple_pdfs(pdf_paths: list) -> list:
    """Load and chunk multiple PDFs, tagging each chunk with its source filename."""
    all_chunks = []
    for path in pdf_paths:
        chunks = load_and_chunk(path, source_name=os.path.basename(path))
        all_chunks.extend(chunks)
    return all_chunks


def build_vectorstore(chunks: list) -> FAISS:
    """Build in-memory FAISS vectorstore from chunks."""
    embeddings = _get_embeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)


def merge_into_vectorstore(existing: FAISS, new_chunks: list) -> FAISS:
    """Merge new document chunks into an existing FAISS index."""
    embeddings = _get_embeddings()
    new_store = FAISS.from_documents(new_chunks, embedding=embeddings)
    existing.merge_from(new_store)
    return existing


# ------------------ IMPROVEMENT 3: RETRIEVAL CACHING ------------------
@functools.lru_cache(maxsize=128)
def _cached_rerank_scores(query: str, doc_contents_tuple: tuple) -> tuple:
    """Cache reranker scores for identical (query, docs) combinations."""
    pairs = [(query, content) for content in doc_contents_tuple]
    scores = _reranker.predict(pairs)
    return tuple(scores)


def hybrid_retrieve(query: str, vectorstore: FAISS, bm25_retriever: BM25Retriever, k: int = 5) -> list:
    """Combine vector search + BM25, deduplicate by content hash."""
    vec_docs = vectorstore.similarity_search(query, k=k)
    bm25_docs = bm25_retriever.invoke(query)

    seen = set()
    combined = []
    for doc in vec_docs + bm25_docs:
        content_hash = doc.metadata.get("content_hash") or hashlib.md5(
            doc.page_content.strip().encode()
        ).hexdigest()
        if content_hash not in seen:
            combined.append(doc)
            seen.add(content_hash)

    return combined


def rerank_with_scores(query: str, docs: list) -> list:
    """Rerank docs using cached CrossEncoder scores."""
    if not docs:
        return []
    doc_contents = tuple(doc.page_content for doc in docs)
    scores = _cached_rerank_scores(query, doc_contents)
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def retrieve_docs(query: str, vectorstore: FAISS, bm25_retriever: BM25Retriever, top_k: int = 3):
    """
    Full retrieval: hybrid search -> rerank -> confidence gate.
    Returns (docs, below_threshold: bool)
    """
    combined = hybrid_retrieve(query, vectorstore, bm25_retriever)
    scored = rerank_with_scores(query, combined)

    if not scored:
        return [], True

    best_score = scored[0][1]
    below_threshold = best_score < RERANKER_SCORE_THRESHOLD
    top_docs = [doc for doc, _ in scored[:top_k]]
    return top_docs, below_threshold


# ------------------ IMPROVEMENT 4: CONVERSATION MEMORY ------------------
def _build_history_section(chat_history: list) -> str:
    """Format recent chat history for inclusion in the prompt."""
    if not chat_history:
        return ""

    # Take last MEMORY_WINDOW turns (user + assistant pairs)
    recent = chat_history[-(MEMORY_WINDOW * 2):]
    lines = ["Conversation history (for follow-up context):"]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300]  # truncate long answers
        lines.append(f"{role}: {content}")
    lines.append("")
    return "\n".join(lines) + "\n"


def generate_answer(
    query: str,
    vectorstore: FAISS,
    bm25_retriever: BM25Retriever,
    chat_history: list = None
):
    """
    Full RAG pipeline: retrieve -> gate -> generate with Groq.
    Improvement 4: accepts chat_history for multi-turn conversation memory.
    Returns (answer: str, docs: list, below_threshold: bool)
    """
    docs, below_threshold = retrieve_docs(query, vectorstore, bm25_retriever)

    if below_threshold or not docs:
        refusal = "This doesn't appear to be covered in the provided documentation."
        return refusal, docs, True

    # Build context with metadata — Improvement 1: include page + source + section
    context_parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "")
        section = doc.metadata.get("section", "")
        header = f"[Source {i} | {source} | p.{page}"
        if section:
            header += f" | {section}"
        header += "]"
        context_parts.append(f"{header}\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)

    # Improvement 4: inject conversation history
    history_section = _build_history_section(chat_history or [])

    user_message = ANSWER_PROMPT_TEMPLATE.format(
        context=context,
        history_section=history_section,
        query=query
    )

    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.choices[0].message.content
    return answer, docs, False