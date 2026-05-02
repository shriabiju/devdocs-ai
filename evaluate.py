"""
evaluate.py — Offline RAG evaluation script
Measures: Answer similarity, Context relevance, Faithfulness (claim-level via Groq)
Run: python evaluate.py

CI integration: exits with code 1 if scores fall below threshold.
"""

import os
import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from langchain_community.retrievers import BM25Retriever

from rag_pipeline import (
    load_and_chunk,
    build_vectorstore,
    generate_answer,
    PDF_PATH,
    GROQ_MODEL,
)

# ------------------ THRESHOLDS (CI gates) ------------------
MIN_ANSWER_SIMILARITY = 0.60
MIN_CONTEXT_RELEVANCE = 0.50
MIN_FAITHFULNESS = 0.70

# ------------------ GOLDEN DATASET ------------------
# Expand to 50-200 pairs for production evaluation
GOLDEN_DATASET = [
    {
        "question": "What is synchronous parallel processing?",
        "reference": "Synchronous Parallel Processing is a structured parallel computing model where computation is divided into super steps with synchronization barriers between them."
    },
    {
        "question": "What are super steps in parallel computing?",
        "reference": "Super steps are stages in synchronous parallel processing where computation is performed followed by communication and a synchronization barrier."
    },
    {
        "question": "What is quantum computing?",  # Out-of-domain — should trigger refusal
        "reference": "Quantum computing is a type of computation that uses quantum mechanical phenomena.",
        "expect_refusal": True
    },
]

# ------------------ HELPERS ------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))


def check_faithfulness(answer: str, context_chunks: list) -> float:
    """
    Use Groq/llama3 to judge whether the answer is faithful to the context.
    Returns a score between 0.0 and 1.0.
    """
    context = "\n\n".join([f"[Chunk {i+1}] {c}" for i, c in enumerate(context_chunks)])

    prompt = f"""You are an evaluation judge. Given a context and an answer, determine whether \
every factual claim in the answer is directly supported by the context.

Context:
{context}

Answer:
{answer}

Respond with ONLY a JSON object, no markdown, no explanation outside the JSON:
{{"faithful": true/false, "score": 0.0-1.0, "reason": "brief explanation"}}

Score 1.0 = fully supported, 0.5 = partially supported, 0.0 = unsupported or hallucinated."""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


def run_evaluation(vectorstore, bm25_retriever):
    answer_scores = []
    context_scores = []
    faithfulness_scores = []
    refusal_correct = []

    print("\n===== RAG Evaluation =====\n")

    for item in GOLDEN_DATASET:
        question = item["question"]
        reference = item["reference"]
        expect_refusal = item.get("expect_refusal", False)

        answer, docs, below_threshold = generate_answer(question, vectorstore, bm25_retriever)
        contexts = [doc.page_content for doc in docs]

        if expect_refusal:
            refused = below_threshold or "don't have enough information" in answer.lower()
            refusal_correct.append(1.0 if refused else 0.0)
            print(f"[REFUSAL TEST] Q: {question}")
            print(f"  Correctly refused: {refused}")
            print()
            continue

        answer_vec = embed_model.encode([answer])[0]
        reference_vec = embed_model.encode([reference])[0]
        answer_sim = float(cosine_similarity([answer_vec], [reference_vec])[0][0])

        if contexts:
            context_vecs = embed_model.encode(contexts)
            ctx_relevance = float(np.mean(cosine_similarity(context_vecs, [answer_vec])))
        else:
            ctx_relevance = 0.0

        faithfulness = check_faithfulness(answer, contexts) if contexts else 0.0

        answer_scores.append(answer_sim)
        context_scores.append(ctx_relevance)
        faithfulness_scores.append(faithfulness)

        print(f"Q: {question}")
        print(f"  Generated: {answer[:120]}...")
        print(f"  Answer Similarity : {answer_sim:.4f}")
        print(f"  Context Relevance : {ctx_relevance:.4f}")
        print(f"  Faithfulness      : {faithfulness:.4f}")
        print()

    avg_answer = np.mean(answer_scores) if answer_scores else 0.0
    avg_context = np.mean(context_scores) if context_scores else 0.0
    avg_faithful = np.mean(faithfulness_scores) if faithfulness_scores else 0.0
    refusal_rate = np.mean(refusal_correct) if refusal_correct else None

    print("===== FINAL SCORES =====")
    print(f"Avg Answer Similarity : {avg_answer:.4f}  (threshold: {MIN_ANSWER_SIMILARITY})")
    print(f"Avg Context Relevance : {avg_context:.4f}  (threshold: {MIN_CONTEXT_RELEVANCE})")
    print(f"Avg Faithfulness      : {avg_faithful:.4f}  (threshold: {MIN_FAITHFULNESS})")
    if refusal_rate is not None:
        print(f"Refusal Accuracy      : {refusal_rate:.4f}")

    failed = (
        avg_answer < MIN_ANSWER_SIMILARITY
        or avg_context < MIN_CONTEXT_RELEVANCE
        or avg_faithful < MIN_FAITHFULNESS
    )

    if failed:
        print("\n❌ EVALUATION FAILED — quality below threshold.")
        sys.exit(1)
    else:
        print("\n✅ EVALUATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

    print(f"Loading: {PDF_PATH}")
    chunks = load_and_chunk(PDF_PATH)
    vectorstore = build_vectorstore(chunks)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    run_evaluation(vectorstore, bm25_retriever)