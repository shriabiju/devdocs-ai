"""
rag_pipeline.py — Core RAG pipeline
Domain: Software / Technical Documentation
Uses: Groq API (llama-3.3-70b-versatile), FAISS, BM25, CrossEncoder reranker
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from groq import Groq

# ------------------ CONFIG ------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_MODEL = "all-MiniLM-L6-v2"
PDF_PATH = "data/sample.pdf"

# Confidence threshold — below this score, refuse to answer
RERANKER_SCORE_THRESHOLD = -2.0

# ------------------ PROMPT CONFIG (versioned) ------------------
# Domain: Software / Technical Documentation
# Tuned for: APIs, codebases, architecture docs, CS textbooks, system design

SYSTEM_PROMPT = """\
You are a senior software engineer and technical documentation assistant. \
Your job is to answer questions about software systems, code, APIs, algorithms, \
data structures, architecture, and CS concepts.

Rules:
1. Use the provided context as your primary source. Prioritize it over your own knowledge.
2. If the context covers the topic partially (e.g. mentions some types but not all), \
answer fully using BOTH the context AND your own accurate technical knowledge. \
Clearly separate them: first state what the document says, then add what it omits.
3. If the context is completely unrelated to the question, say: \
"This doesn't appear to be covered in the provided documentation."
4. Structure answers clearly: definition first, then all relevant types/details/variants, \
then a code example only if present in the context.
5. Use bullet points or numbered lists when enumerating types, states, or steps.
6. Cite inline as (Source 1) for context. Label additions as (general knowledge).
7. Be concise but complete.\
"""

ANSWER_PROMPT_TEMPLATE = """\
Technical documentation context:

{context}

Developer question: {query}

Answer using the context as primary source. If the context only partially covers \
the topic, supplement with accurate technical knowledge and clearly label each part:\
"""

# ------------------ INIT ------------------
_reranker = CrossEncoder(RERANKER_MODEL)
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY", GROQ_API_KEY))


def load_and_chunk(pdf_path: str) -> list:
    """Load PDF and return cleaned chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    filtered = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        word_count = len(set(text.lower().split()))
        if len(text) > 120 and word_count > 40:
            filtered.append(chunk)

    return filtered


def build_vectorstore(chunks: list) -> FAISS:
    """Build in-memory FAISS vectorstore from chunks."""
    return FAISS.from_documents(chunks, embedding=_embeddings)


def hybrid_retrieve(query: str, vectorstore: FAISS, bm25_retriever: BM25Retriever, k: int = 5) -> list:
    """Combine vector search + BM25, deduplicate."""
    vec_docs = vectorstore.similarity_search(query, k=k)
    bm25_docs = bm25_retriever.invoke(query)

    seen = set()
    combined = []
    for doc in vec_docs + bm25_docs:
        content = doc.page_content.strip()
        if content not in seen:
            combined.append(doc)
            seen.add(content)

    return combined


def rerank_with_scores(query: str, docs: list) -> list:
    """Return list of (doc, score) sorted by relevance descending."""
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = _reranker.predict(pairs)
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


def generate_answer(query: str, vectorstore: FAISS, bm25_retriever: BM25Retriever):
    """
    Full RAG pipeline: retrieve -> gate -> generate with Groq.
    Returns (answer: str, docs: list, below_threshold: bool)
    """
    docs, below_threshold = retrieve_docs(query, vectorstore, bm25_retriever)

    if below_threshold or not docs:
        refusal = "This doesn't appear to be covered in the provided documentation."
        return refusal, docs, True

    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[Source {i}]\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)

    user_message = ANSWER_PROMPT_TEMPLATE.format(context=context, query=query)

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