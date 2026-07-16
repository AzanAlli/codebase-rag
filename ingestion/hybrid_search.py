"""
Day 3: hybrid search (vector + full-text) with cross-encoder reranking.

Pipeline:
    1. Run vector similarity search (top N candidates)
    2. Run full-text keyword search (top N candidates)
    3. Merge the two ranked lists via Reciprocal Rank Fusion (RRF) —
       a simple, weight-free way to combine rankings: a chunk that
       appears near the top of *either* list gets a high fused score.
    4. Take the fused top-k and rerank them with a cross-encoder
       (which scores query+chunk pairs directly, more accurate than
       cosine similarity, but too slow to run over the whole table —
       hence only reranking a small shortlist).

Usage:
    python3 ingestion/hybrid_search.py "how does click parse command line arguments"
"""

import os
import sys

import psycopg2
import vertexai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]

VECTOR_TOP_N = 20      # candidates pulled from vector search
FULLTEXT_TOP_N = 20    # candidates pulled from full-text search
RRF_K = 60             # standard RRF damping constant
RERANK_TOP_N = 10       # how many fused candidates go into the cross-encoder
FINAL_TOP_K = 5          # how many results to show at the end

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def embed_query(query: str) -> list[float]:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    query_input = TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")
    return model.get_embeddings([query_input])[0].values


def vector_search(cur, query_embedding, top_n: int) -> list[dict]:
    cur.execute(
        """
        SELECT id, file_path, symbol_name, symbol_type, parent_class,
               start_line, end_line, docstring, source
        FROM code_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, top_n),
    )
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def fulltext_search(cur, query: str, top_n: int) -> list[dict]:
    cur.execute(
        """
        SELECT id, file_path, symbol_name, symbol_type, parent_class,
               start_line, end_line, docstring, source
        FROM code_chunks
        WHERE search_vector @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(search_vector, plainto_tsquery('english', %s)) DESC
        LIMIT %s;
        """,
        (query, query, top_n),
    )
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def reciprocal_rank_fusion(*ranked_lists, k=RRF_K) -> list[dict]:
    """
    Merge multiple ranked lists of chunks (each a list of dicts with an
    'id' key) into one fused ranking. Each chunk's score is the sum of
    1/(k + rank) across every list it appears in — chunks that rank highly
    in *either* list, or appear in *both*, float to the top.
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_id = chunk["id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
            chunk_lookup[chunk_id] = chunk

    fused_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [chunk_lookup[cid] for cid in fused_ids]


def cross_encoder_rerank(query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
    model = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = []
    for chunk in chunks:
        text = f"{chunk['symbol_type']} {chunk['symbol_name']}"
        if chunk.get("docstring"):
            text += f"\n{chunk['docstring']}"
        text += f"\n{chunk['source'][:1000]}"  # cap length for speed
        pairs.append((query, text))

    scores = model.predict(pairs)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def print_results(title: str, results, show_score=True):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    for i, item in enumerate(results, 1):
        chunk, score = item if show_score else (item, None)
        location = f"{chunk['symbol_type']} {chunk['symbol_name']}"
        if chunk.get("parent_class"):
            location += f" (in class {chunk['parent_class']})"
        score_str = f"[{score:.3f}] " if score is not None else ""
        print(f"{i}. {score_str}{location}")
        print(f"   {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}")
        if chunk.get("docstring"):
            first_line = chunk["docstring"].strip().split("\n")[0]
            print(f'   "{first_line}"')


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "how does click parse command line arguments"
    print(f"Query: {query}")

    print("\nEmbedding query...")
    query_embedding = embed_query(query)

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    print(f"Running vector search (top {VECTOR_TOP_N})...")
    vector_results = vector_search(cur, query_embedding, VECTOR_TOP_N)

    print(f"Running full-text search (top {FULLTEXT_TOP_N})...")
    fulltext_results = fulltext_search(cur, query, FULLTEXT_TOP_N)

    cur.close()
    conn.close()

    print("Fusing rankings (Reciprocal Rank Fusion)...")
    fused = reciprocal_rank_fusion(vector_results, fulltext_results)
    fused_shortlist = fused[:RERANK_TOP_N]

    print(f"Reranking top {len(fused_shortlist)} fused candidates with cross-encoder...")
    reranked = cross_encoder_rerank(query, fused_shortlist, FINAL_TOP_K)

    # for comparison: show what vector-only search alone would have returned
    print_results(
        "BASELINE: vector search only (Day 2 approach)",
        vector_results[:FINAL_TOP_K],
        show_score=False,
    )

    print_results(
        "NEW: hybrid search + cross-encoder reranking (Day 3)",
        reranked,
        show_score=True,
    )


if __name__ == "__main__":
    main()
