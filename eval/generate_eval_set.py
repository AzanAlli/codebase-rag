"""
Day 5, step 1: generate a synthetic evaluation set.

Rather than hand-labeling ground truth (error-prone, doesn't scale, and
risks me guessing wrong about a codebase I didn't write), we sample real
chunks from the database and ask Gemini to write a natural question a
developer might ask that this specific chunk answers — without naming the
function directly. This gives us {query, expected_chunk_id} pairs we can
use to measure retrieval quality objectively.

Only samples non-test chunks (consistent with core/retrieval.py's default
behavior of excluding tests/ unless the query is explicitly about testing).

Usage:
    python3 eval/generate_eval_set.py --n 20
"""

import argparse
import json
import os
import re

import psycopg2
from dotenv import load_dotenv
from google import genai

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]

GENERATION_MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are helping build an evaluation set for a code-search tool.

Below is a piece of Python code with its docstring. Write ONE natural-language
question that a developer might type into a search bar, where this code is
the correct answer. The question should describe what the code DOES, in
plain English — do NOT mention the function/class/method name directly, and
do NOT mention the file name.

Symbol type: {symbol_type}
Symbol name: {symbol_name}
Docstring: {docstring}

Respond with ONLY the question, nothing else — no quotes, no preamble."""


def fetch_candidate_chunks(conn, limit: int) -> list[dict]:
    """Sample non-test chunks with substantial docstrings — good eval candidates."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, file_path, symbol_name, symbol_type, parent_class, docstring
        FROM code_chunks
        WHERE docstring IS NOT NULL
          AND length(docstring) > 60
          AND file_path NOT LIKE 'tests/%%'
          AND file_path NOT LIKE '%%/tests/%%'
        ORDER BY random()
        LIMIT %s;
        """,
        (limit,),
    )
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    return rows


def generate_question(client, chunk: dict) -> str | None:
    prompt = PROMPT_TEMPLATE.format(
        symbol_type=chunk["symbol_type"],
        symbol_name=chunk["symbol_name"],
        docstring=chunk["docstring"],
    )
    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        question = response.text.strip()
        # reject if it leaked the exact symbol name (too easy / not a fair test)
        if re.search(re.escape(chunk["symbol_name"]), question, re.IGNORECASE):
            return None
        return question
    except Exception as e:
        print(f"  [warn] generation failed for {chunk['symbol_name']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of eval questions to generate")
    parser.add_argument("--out", default="eval/eval_set.json")
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL)
    candidates = fetch_candidate_chunks(conn, limit=args.n + 15)  # oversample for rejections
    conn.close()

    print(f"Sampled {len(candidates)} candidate chunks. Generating questions...")

    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)

    eval_set = []
    for chunk in candidates:
        if len(eval_set) >= args.n:
            break
        question = generate_question(client, chunk)
        if question is None:
            continue
        eval_set.append(
            {
                "query": question,
                "expected_chunk_id": chunk["id"],
                "expected_symbol_name": chunk["symbol_name"],
                "expected_file_path": chunk["file_path"],
            }
        )
        print(f"  [{len(eval_set)}/{args.n}] {chunk['symbol_name']}: {question}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(eval_set, f, indent=2)

    print(f"\nWrote {len(eval_set)} eval questions to {args.out}")


if __name__ == "__main__":
    main()
