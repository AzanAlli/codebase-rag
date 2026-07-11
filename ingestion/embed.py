"""
Day 2: embed all chunks from chunks.jsonl using Vertex AI's text embedding
model, and store them (with metadata) in Supabase Postgres via pgvector.

Usage:
    python3 ingestion/embed.py chunks.jsonl
"""

import json
import os
import sys
import time

import psycopg2
import psycopg2.extras
import vertexai
from dotenv import load_dotenv
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]

EMBEDDING_MODEL_NAME = "text-embedding-004"
EMBEDDING_DIM = 768
BATCH_SIZE = 20  # Vertex AI batches embedding requests; keep well under API limits

TABLE_DDL = """
CREATE TABLE IF NOT EXISTS code_chunks (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    symbol_name TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    parent_class TEXT,
    start_line INT NOT NULL,
    end_line INT NOT NULL,
    docstring TEXT,
    source TEXT NOT NULL,
    embedding VECTOR(768)
);
"""

UPSERT_SQL = """
INSERT INTO code_chunks
    (id, file_path, symbol_name, symbol_type, parent_class, start_line, end_line, docstring, source, embedding)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    file_path = EXCLUDED.file_path,
    symbol_name = EXCLUDED.symbol_name,
    symbol_type = EXCLUDED.symbol_type,
    parent_class = EXCLUDED.parent_class,
    start_line = EXCLUDED.start_line,
    end_line = EXCLUDED.end_line,
    docstring = EXCLUDED.docstring,
    source = EXCLUDED.source,
    embedding = EXCLUDED.embedding;
"""


def load_chunks(path: str) -> list[dict]:
    chunks = []
    with open(path) as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_embedding_text(chunk: dict) -> str:
    """
    Combine structural + intent signals into one string to embed.
    Docstring carries natural-language intent; source carries the
    actual implementation; symbol/parent info gives structural context.
    """
    parts = []
    location = f"{chunk['symbol_type']} {chunk['symbol_name']}"
    if chunk.get("parent_class"):
        location += f" in class {chunk['parent_class']}"
    location += f" ({chunk['file_path']})"
    parts.append(location)

    if chunk.get("docstring"):
        parts.append(f"Docstring: {chunk['docstring']}")

    parts.append(f"Code:\n{chunk['source']}")
    return "\n\n".join(parts)


def embed_batch(model: TextEmbeddingModel, texts: list[str]) -> list[list[float]]:
    inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
    embeddings = model.get_embeddings(inputs)
    return [e.values for e in embeddings]


def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute(TABLE_DDL)
    conn.commit()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ingestion/embed.py <chunks.jsonl>")
        sys.exit(1)

    chunks_path = sys.argv[1]
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    print(f"Initializing Vertex AI (project={GCP_PROJECT_ID}, region={GCP_REGION})...")
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)

    print("Connecting to Supabase...")
    conn = psycopg2.connect(DATABASE_URL)
    ensure_table(conn)

    total = len(chunks)
    inserted = 0
    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [build_embedding_text(c) for c in batch]

        try:
            embeddings = embed_batch(model, texts)
        except Exception as e:
            print(f"  [warn] batch {i}-{i+len(batch)} failed: {e}")
            continue

        rows = []
        for chunk, embedding in zip(batch, embeddings):
            rows.append(
                (
                    chunk["chunk_id"],
                    chunk["file_path"],
                    chunk["symbol_name"],
                    chunk["symbol_type"],
                    chunk.get("parent_class"),
                    chunk["start_line"],
                    chunk["end_line"],
                    chunk.get("docstring"),
                    chunk["source"],
                    embedding,
                )
            )

        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_SQL, rows)
        conn.commit()

        inserted += len(rows)
        elapsed = time.time() - start_time
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"  {inserted}/{total} chunks embedded and stored ({rate:.1f}/sec)")

    conn.close()
    print(f"\nDone. {inserted}/{total} chunks embedded and stored in Supabase.")


if __name__ == "__main__":
    main()
