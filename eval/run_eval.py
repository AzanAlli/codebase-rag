"""
Day 5, step 2: run the eval set and score both retrieval and generation.

Retrieval metrics (baseline vector-only vs hybrid+rerank+filter), for
k in [1, 3, 5, 10]:
    - Hit rate @ k: fraction of queries where the expected chunk appears
      somewhere in the top k results
    - Mean Reciprocal Rank (MRR): average of 1/rank of the expected chunk
      across all queries (0 if not found)

Generation metric:
    - Faithfulness rate: fraction of generated answers where every citation
      is grounded in a retrieved chunk (uses core/generation.py's
      containment-based faithfulness check)

Usage:
    python3 eval/run_eval.py eval/eval_set.json
"""

import json
import sys
from pathlib import Path

# allow running as `python3 eval/run_eval.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retrieval import vector_only_search, hybrid_search
from core.generation import generate_answer


def rank_of_expected(results: list[dict], expected_chunk_id: str) -> int | None:
    """1-indexed rank of the expected chunk in results, or None if absent."""
    for i, chunk in enumerate(results):
        if chunk["id"] == expected_chunk_id:
            return i + 1
    return None


def compute_retrieval_metrics(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict:
    n = len(ranks)
    metrics = {}
    for k in ks:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"hit_rate@{k}"] = hits / n if n else 0.0
    reciprocal_ranks = [1.0 / r if r is not None else 0.0 for r in ranks]
    metrics["mrr"] = sum(reciprocal_ranks) / n if n else 0.0
    return metrics


def main():
    eval_path = sys.argv[1] if len(sys.argv) > 1 else "eval/eval_set.json"
    with open(eval_path) as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} eval questions from {eval_path}\n")

    baseline_ranks = []
    hybrid_ranks = []
    faithfulness_results = []
    detail_records = []

    for i, item in enumerate(eval_set, 1):
        query = item["query"]
        expected_id = item["expected_chunk_id"]

        # --- retrieval comparison ---
        baseline_results = vector_only_search(query, top_k=10)
        baseline_rank = rank_of_expected(baseline_results, expected_id)
        baseline_ranks.append(baseline_rank)

        hybrid_results = hybrid_search(query, top_k=10)
        hybrid_chunks = [chunk for chunk, _score in hybrid_results]
        hybrid_rank = rank_of_expected(hybrid_chunks, expected_id)
        hybrid_ranks.append(hybrid_rank)

        # --- generation faithfulness (on the hybrid pipeline's top 5) ---
        top5_chunks = hybrid_chunks[:5]
        generation = generate_answer(query, top5_chunks)
        is_faithful = generation["faithfulness"]["is_faithful"]
        faithfulness_results.append(is_faithful)

        detail_records.append(
            {
                "query": query,
                "answer": generation["answer"],
                "citations_found": generation["faithfulness"]["citations_found"],
                "ungrounded_citations": generation["faithfulness"]["ungrounded_citations"],
                "has_citations": generation["faithfulness"]["has_citations"],
                "is_faithful": is_faithful,
            }
        )

        b_status = baseline_rank if baseline_rank else "miss"
        h_status = hybrid_rank if hybrid_rank else "miss"
        f_status = "faithful" if is_faithful else "UNFAITHFUL"
        print(f"[{i}/{len(eval_set)}] baseline={b_status:<5} hybrid={h_status:<5} gen={f_status:<11} :: {query[:55]}")

    baseline_metrics = compute_retrieval_metrics(baseline_ranks)
    hybrid_metrics = compute_retrieval_metrics(hybrid_ranks)
    faithfulness_rate = sum(faithfulness_results) / len(faithfulness_results) if faithfulness_results else 0.0

    print(f"\n{'=' * 65}")
    print(f"RETRIEVAL RESULTS ({len(eval_set)} queries)")
    print(f"{'=' * 65}")
    print(f"{'Metric':<15}{'Baseline (vector only)':<28}{'Hybrid+rerank+filter':<20}")
    for key in baseline_metrics:
        b = baseline_metrics[key]
        h = hybrid_metrics[key]
        delta = h - b
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
        print(f"{key:<15}{b:<28.3f}{h:<15.3f} {arrow} {abs(delta):.3f}")

    print(f"\n{'=' * 65}")
    print(f"GENERATION FAITHFULNESS")
    print(f"{'=' * 65}")
    print(f"Faithful answers: {sum(faithfulness_results)}/{len(faithfulness_results)} ({faithfulness_rate:.1%})")

    results_out = {
        "n_queries": len(eval_set),
        "baseline_metrics": baseline_metrics,
        "hybrid_metrics": hybrid_metrics,
        "faithfulness_rate": faithfulness_rate,
    }
    with open("eval/eval_results.json", "w") as f:
        json.dump(results_out, f, indent=2)
    print("\nWrote eval/eval_results.json")

    with open("eval/eval_details.json", "w") as f:
        json.dump(detail_records, f, indent=2)
    print("Wrote eval/eval_details.json (full answers + citation debug info)")


if __name__ == "__main__":
    main()