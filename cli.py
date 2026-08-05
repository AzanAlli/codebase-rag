"""
Interactive CLI for the codebase RAG pipeline. Calls the retrieval +
generation pipeline directly (no need to run the API server separately) —
useful for quick demos or testing without juggling curl commands.

Routes conceptual/non-technical questions ("what does this project do")
to a plain-language answer using the project summary, and everything else
through the normal technical retrieval + generation pipeline.

Usage:
    python3 cli.py
    python3 cli.py "how does click parse command line arguments"
    python3 cli.py --graph "how does click parse command line arguments"      # with call-graph expansion
    python3 cli.py --multihop "how does click parse args and format help"     # with query decomposition
    python3 cli.py "what does this project do"                                # routes to conceptual answer
"""

import sys

from core.retrieval import hybrid_search, expand_with_call_graph, multi_hop_search
from core.generation import generate_answer
from core.conceptual import is_conceptual_query, get_stored_summary, generate_conceptual_answer


def ask(question: str, use_graph: bool = False, use_multihop: bool = False):
    if is_conceptual_query(question):
        print("\n(Recognized as a general question about the project)")
        summary = get_stored_summary()
        if summary is None:
            print("No project summary found — run ingestion/generate_project_summary.py first.")
            print("Falling back to the technical pipeline...\n")
        else:
            generation = generate_conceptual_answer(question, summary)
            print(f"\n{'=' * 70}")
            print("ANSWER")
            print(f"{'=' * 70}")
            print(generation["answer"])
            print(f"\n{'=' * 70}")
            print("(Conceptual answer — no code citations, based on the project's README)")
            print(f"{'=' * 70}")
            return

    sub_questions = None
    if use_multihop:
        print("\nDecomposing query...")
        results, sub_questions = multi_hop_search(question, top_k=5)
        if len(sub_questions) > 1:
            print(f"Split into {len(sub_questions)} sub-questions:")
            for i, sq in enumerate(sub_questions, 1):
                print(f"  {i}. {sq}")
        else:
            print("Not a compound question — searched as-is.")
        chunks = [chunk for chunk, _score in results]
    else:
        print(f"\nSearching...")
        results = hybrid_search(question, top_k=5)
        chunks = [chunk for chunk, _score in results]

    if use_graph:
        before = len(chunks)
        chunks = expand_with_call_graph(chunks, max_extra=3)
        added = len(chunks) - before
        print(f"Call-graph expansion added {added} chunk(s)")

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

    if use_graph and len(chunks) > len(results):
        print(f"\n{'-' * 70}")
        print("CALL-GRAPH EXPANSION (added via caller/callee relationships)")
        print(f"{'-' * 70}")
        for chunk in chunks[len(results):]:
            location = f"{chunk['symbol_type']} {chunk['symbol_name']}"
            if chunk.get("parent_class"):
                location += f" (in class {chunk['parent_class']})"
            print(f"+ {location}")
            print(f"   {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}")
            print(f"   (related to: {chunk['expansion_source_symbol']})")

    faithfulness = generation["faithfulness"]
    status = "FAITHFUL" if faithfulness["is_faithful"] else "UNVERIFIED"
    print(f"\n{'=' * 70}")
    print(f"FAITHFULNESS: {status}")
    print(f"{'=' * 70}")
    print(f"  Citations found: {len(faithfulness['citations_found'])}")
    if faithfulness["ungrounded_citations"]:
        print(f"  Ungrounded citations: {faithfulness['ungrounded_citations']}")


def main():
    args = sys.argv[1:]
    use_graph = "--graph" in args
    use_multihop = "--multihop" in args
    if use_graph:
        args.remove("--graph")
    if use_multihop:
        args.remove("--multihop")

    if args:
        # single query mode
        question = " ".join(args)
        ask(question, use_graph=use_graph, use_multihop=use_multihop)
        return

    # interactive mode
    print("Codebase RAG — ask a question about the indexed repo.")
    print("Type 'quit' or 'exit' to stop. Prefix with '--graph' or '--multihop' for those modes.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            print("Bye.")
            break

        graph_flag = raw.startswith("--graph")
        multihop_flag = raw.startswith("--multihop")
        question = raw
        if graph_flag:
            question = raw[len("--graph"):].strip()
        elif multihop_flag:
            question = raw[len("--multihop"):].strip()

        ask(question, use_graph=graph_flag, use_multihop=multihop_flag)
        print()


if __name__ == "__main__":
    main()
