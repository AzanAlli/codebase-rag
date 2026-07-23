"""
Interactive CLI for the codebase RAG pipeline. Calls the retrieval +
generation pipeline directly (no need to run the API server separately) —
useful for quick demos or testing without juggling curl commands.

Usage:
    python3 cli.py
    python3 cli.py "how does click parse command line arguments"   # single query, then exit
"""

import sys

from core.retrieval import hybrid_search
from core.generation import generate_answer


def ask(question: str):
    print(f"\nSearching...")
    results = hybrid_search(question, top_k=5)
    chunks = [chunk for chunk, _score in results]

    print("Generating answer...")
    generation = generate_answer(question, chunks)

    print(f"\n{'=' * 70}")
    print("ANSWER")
    print(f"{'=' * 70}")
    print(generation["answer"])

    print(f"\n{'=' * 70}")
    print("SOURCES")
    print(f"{'=' * 70}")
    for i, (chunk, score) in enumerate(results, 1):
        location = f"{chunk['symbol_type']} {chunk['symbol_name']}"
        if chunk.get("parent_class"):
            location += f" (in class {chunk['parent_class']})"
        print(f"{i}. [{score:.3f}] {location}")
        print(f"   {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}")

    faithfulness = generation["faithfulness"]
    status = " FAITHFUL" if faithfulness["is_faithful"] else "⚠️  UNVERIFIED"
    print(f"\n{'=' * 70}")
    print(f"FAITHFULNESS: {status}")
    print(f"{'=' * 70}")
    print(f"  Citations found: {len(faithfulness['citations_found'])}")
    if faithfulness["ungrounded_citations"]:
        print(f"  Ungrounded citations: {faithfulness['ungrounded_citations']}")


def main():
    if len(sys.argv) > 1:
        # single query mode
        question = " ".join(sys.argv[1:])
        ask(question)
        return

    # interactive mode
    print("Codebase RAG — ask a question about the indexed repo.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Bye.")
            break

        ask(question)
        print()


if __name__ == "__main__":
    main()
