"""
Generates a plain-English project summary (what is this, who's it for, why
would someone use it) from the repo's README, and stores it for use when a
non-technical question comes in (e.g. from a recruiter or non-engineering
interviewer) — see core/conceptual.py for how it's used.

Run once per indexed repo, after parsing/embedding.

Usage:
    python3 ingestion/generate_project_summary.py test_repos/click
"""

import glob
import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from google import genai

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]

GENERATION_MODEL = "gemini-2.5-flash"

TABLE_DDL = """
DROP TABLE IF EXISTS project_summary;
CREATE TABLE project_summary (
    id SERIAL PRIMARY KEY,
    repo_name TEXT NOT NULL,
    summary TEXT NOT NULL
);
"""

PROMPT_TEMPLATE = """You are writing a short project summary for someone with
NO programming background — a recruiter, a hiring manager, a non-technical
interviewer. They will ask things like "what does this project do?" or "why
would someone use this?".

Below is the README of a code repository. Write a warm, plain-English
summary (2-3 short paragraphs) covering:
- What the project actually does, in everyday language
- What problem it solves / why it's useful
- Who would use it

Rules:
- NO code, NO function names, NO technical jargon, NO citations
- Write like you're explaining it to a smart friend outside tech
- Be concise and genuinely clear, not vague marketing-speak

README:
{readme}

Summary:"""


def find_readme(repo_root: Path) -> str | None:
    for pattern in ("README.md", "README.rst", "README.txt", "README"):
        matches = glob.glob(str(repo_root / pattern))
        if matches:
            return Path(matches[0]).read_text(errors="replace")
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ingestion/generate_project_summary.py <repo_root>")
        sys.exit(1)

    repo_root = Path(sys.argv[1])
    repo_name = repo_root.name

    readme = find_readme(repo_root)
    if readme is None:
        print(f"No README found in {repo_root} — can't generate a project summary.")
        sys.exit(1)

    print(f"Found README ({len(readme)} chars). Generating summary...")

    client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)
    # cap README length sent to the model — a summary doesn't need the whole thing
    prompt = PROMPT_TEMPLATE.format(readme=readme[:8000])
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    summary = response.text.strip()

    print(f"\n{'=' * 70}\nGENERATED SUMMARY\n{'=' * 70}\n{summary}\n")

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(TABLE_DDL)
    cur.execute(
        "INSERT INTO project_summary (repo_name, summary) VALUES (%s, %s);",
        (repo_name, summary),
    )
    conn.commit()
    cur.close()
    conn.close()

    print(f"Stored summary for '{repo_name}' in project_summary table.")


if __name__ == "__main__":
    main()
