"""
Sanity check for Day 2: embed a natural-language query the same way we
embedded the chunks, then run a raw cosine-similarity search against
Supabase to confirm retrieval actually surfaces relevant code.
"""

import os
import sys

import psycopg2
import vertexai
from dotenv import load_dotenv
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]

QUERY = sys.argv[1] if len(sys.argv) > 1 else "how does click parse command line arguments"
TOP_K = 5

vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION)
model = TextEmbeddingModel.from_pretrained("text-embedding-004")

print(f"Query: {QUERY}\n")

query_input = TextEmbeddingInput(text=QUERY, task_type="RETRIEVAL_QUERY")
query_embedding = model.get_embeddings([query_input])[0].values

conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

# pgvector: <=> is cosine distance (lower = more similar)
cur.execute(
    """
    SELECT file_path, symbol_name, symbol_type, parent_class, start_line, end_line,
           docstring, 1 - (embedding <=> %s::vector) AS similarity
    FROM code_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """,
    (query_embedding, query_embedding, TOP_K),
)

results = cur.fetchall()

print(f"Top {TOP_K} results:\n")
for i, row in enumerate(results, 1):
    file_path, symbol_name, symbol_type, parent_class, start_line, end_line, docstring, similarity = row
    location = f"{symbol_type} {symbol_name}"
    if parent_class:
        location += f" (in class {parent_class})"
    print(f"{i}. [{similarity:.3f}] {location}")
    print(f"   {file_path}:{start_line}-{end_line}")
    if docstring:
        first_line = docstring.strip().split("\n")[0]
        print(f"   \"{first_line}\"")
    print()

cur.close()
conn.close()
