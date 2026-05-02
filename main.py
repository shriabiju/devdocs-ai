"""
main.py — CLI entrypoint for testing the RAG pipeline locally
Usage: python main.py --pdf path/to/your.pdf --query "your question here"

Example:
    python main.py --pdf docs/java_notes.pdf --query "what is inheritance?"
"""

import os
import argparse
from langchain_community.retrievers import BM25Retriever

from rag_pipeline import (
    load_and_chunk,
    build_vectorstore,
    generate_answer,
    GROQ_MODEL,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DevDocs AI — CLI interface for testing the RAG pipeline"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Path to the PDF file to query"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question to ask about the document"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)"
    )
    return parser.parse_args()


def main():
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Run: set GROQ_API_KEY=your_key (Windows) or export GROQ_API_KEY=your_key (Mac/Linux)"
        )

    args = parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    print(f"\n{'='*60}")
    print(f"  DevDocs AI — CLI Test")
    print(f"  Model : {GROQ_MODEL}")
    print(f"  PDF   : {args.pdf}")
    print(f"  Query : {args.query}")
    print(f"{'='*60}\n")

    print("⏳ Loading and chunking document...")
    chunks = load_and_chunk(args.pdf)
    print(f"✓ {len(chunks)} chunks indexed")

    print("⏳ Building vector store...")
    vectorstore = build_vectorstore(chunks)

    print("⏳ Building BM25 retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = args.top_k

    print("⏳ Retrieving and generating answer...\n")
    answer, docs, below_threshold = generate_answer(query=args.query, vectorstore=vectorstore, bm25_retriever=bm25_retriever)

    if below_threshold:
        print("⚠️  Low confidence — question may not be covered in this document.\n")

    print("── ANSWER " + "─" * 50)
    print(answer)

    print("\n── SOURCES " + "─" * 49)
    if docs:
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            print(f"\n[Source {i}] Page {page}")
            print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
            print("-" * 60)
    else:
        print("No sources retrieved.")


if __name__ == "__main__":
    main()