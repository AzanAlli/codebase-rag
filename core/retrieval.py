"""
Shared retrieval pipeline: hybrid search (vector + full-text, fused via RRF)
with cross-encoder reranking. Used by both the CLI script
(ingestion/hybrid_search.py) and the API (api/main.py), so retrieval logic
lives in exactly one place.
"""

import os

import psycopg2
import vertexai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]

VECTOR_TOP_N = 20
FULLTEXT_TOP_N = 20
RRF_K = 60
RERANK_TOP_N = 10
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embed_model = None
_cross_encoder = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
        _embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    return _embed_model


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _get_connection():
    return psycopg2.connect(DATABASE_URL)


def embed_query(query: str) -> list[float]:
    model = _get_embed_model()
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
    scores: dict[str, float] = {}
    lookup: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            lookup[cid] = chunk
    fused_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    return [lookup[cid] for cid in fused_ids]


def cross_encoder_rerank(query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
    model = _get_cross_encoder()
    pairs = []
    for chunk in chunks:
        text = f"{chunk['symbol_type']} {chunk['symbol_name']}"
        if chunk.get("docstring"):
            text += f"\n{chunk['docstring']}"
        text += f"\n{chunk['source'][:1000]}"
        pairs.append((query, text))
    scores = model.predict(pairs)
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hybrid_search(query: str, top_k: int = 5) -> list[tuple[dict, float]]:
    """
    Full pipeline: embed query, run vector + full-text search, fuse via RRF,
    rerank the fused shortlist with a cross-encoder. Returns a list of
    (chunk, relevance_score) tuples, best first.
    """
    query_embedding = embed_query(query)

    conn = _get_connection()
    cur = conn.cursor()
    vector_results = vector_search(cur, query_embedding, VECTOR_TOP_N)
    fulltext_results = fulltext_search(cur, query, FULLTEXT_TOP_N)
    cur.close()
    conn.close()

    fused = reciprocal_rank_fusion(vector_results, fulltext_results)
    shortlist = fused[:RERANK_TOP_N]
    return cross_encoder_rerank(query, shortlist, top_k)
