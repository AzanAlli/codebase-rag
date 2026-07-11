"""
Quick sanity check: can we connect to Supabase Postgres, and can we
enable the pgvector extension? Run this once before building anything else.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

print("Connecting to Supabase Postgres...")
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute("SELECT version();")
print("Connected. Postgres version:")
print(" ", cur.fetchone()[0])

print("\nEnabling pgvector extension...")
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
result = cur.fetchone()
if result:
    print("pgvector is enabled.")
else:
    print("something went wrong — pgvector not found after CREATE EXTENSION")

cur.close()
conn.close()
print("\nAll good — ready for Day 2.")
