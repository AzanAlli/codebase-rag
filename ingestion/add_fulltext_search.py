"""
Day 3, step 1: add full-text search to the existing code_chunks table.

Adds a generated tsvector column combining symbol_name, docstring, and
source, weighted so symbol name matches rank highest (a query mentioning
"parse_args" should strongly favor the function literally named that).
Then indexes it with GIN for fast keyword search.

Run once. Safe to re-run (uses IF NOT EXISTS / OR REPLACE where possible).
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]

MIGRATION_SQL = """
-- Add a generated tsvector column: symbol name weighted highest (A),
-- then docstring (B), then source code (C).
ALTER TABLE code_chunks
ADD COLUMN IF NOT EXISTS search_vector tsvector
GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(symbol_name, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(docstring, '')), 'B') ||
    setweight(to_tsvector('english', coalesce(source, '')), 'C')
) STORED;

CREATE INDEX IF NOT EXISTS code_chunks_search_idx
ON code_chunks USING GIN (search_vector);
"""

if __name__ == "__main__":
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("Adding full-text search column + index to code_chunks...")
    cur.execute(MIGRATION_SQL)
    conn.commit()

    cur.execute("SELECT count(*) FROM code_chunks WHERE search_vector IS NOT NULL;")
    count = cur.fetchone()[0]
    print(f"Done. {count} rows now have a populated search_vector.")

    cur.close()
    conn.close()
