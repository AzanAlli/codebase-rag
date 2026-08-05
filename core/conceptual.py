"""
Handles non-technical, conceptual questions about a repo — "what does this
project do", "why would a company use this", "who is this for" — the kind
of question a recruiter or non-engineering interviewer would actually ask,
which the normal code-retrieval pipeline isn't built to answer well (it
would either find nothing relevant or return a confusing, jargon-heavy
answer).

Classification is a plain heuristic, not an LLM call — cheap, fast, and
predictable. Genuine ambiguous cases fall through to the normal technical
pipeline rather than risking a wrong classification hiding real answers.
"""

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

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)
    return _client


# Patterns for genuinely conceptual/non-technical questions. Deliberately
# conservative — a question has to clearly match one of these patterns to
# be routed away from the normal technical pipeline, since a false
# positive here (treating a real technical question as conceptual) would
# hide a good answer behind a vague summary.
CONCEPTUAL_PATTERNS = [
    r"\bwhat (is|does)\s+(this|the)\s+(project|repo|repository|tool|library|codebase)\b",
    r"\bwhy would (someone|anyone|a company|i|you)\s+use\s+this\b",
    r"\bwho\s+(is|would)\s+this\s+(for|be for)\b",
    r"\bwhat problem\s+does\s+(this|it)\s+solve\b",
    r"\bexplain\s+(this|the)\s+project\b",
    r"\b(give|tell)\s+me\s+an?\s+overview\b",
    r"\bwhat\s+can\s+(i|you)\s+do\s+with\s+this\b",
    r"\btell\s+me\s+about\s+(this|the)\s+(project|repo|repository)\b",
    r"^\s*what\s+is\s+this\??\s*$",
    r"^\s*what\s+does\s+it\s+do\??\s*$",
]


def is_conceptual_query(query: str) -> bool:
    """True if the query looks like a non-technical, whole-project question."""
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in CONCEPTUAL_PATTERNS)


def get_stored_summary(repo_name: str | None = None) -> str | None:
    """Fetch the most recently generated project summary, if one exists."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    if repo_name:
        cur.execute(
            "SELECT summary FROM project_summary WHERE repo_name = %s ORDER BY id DESC LIMIT 1;",
            (repo_name,),
        )
    else:
        cur.execute("SELECT summary FROM project_summary ORDER BY id DESC LIMIT 1;")
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


CONCEPTUAL_PROMPT_TEMPLATE = """You are explaining a software project to someone
with NO programming background — be warm, clear, and jargon-free, like
you're talking to a smart friend outside tech.

Project summary:
{summary}

They asked: {question}

Answer their specific question using the summary above. Keep it
conversational and concise — a few sentences, not an essay. Do NOT include
code, function names, file paths, or citations."""


def generate_conceptual_answer(question: str, summary: str) -> dict:
    """
    Answer a conceptual question in plain language, using the stored
    project summary. Returns the same shape as core.generation's
    generate_answer() (answer + a faithfulness dict) so callers can treat
    both paths uniformly — faithfulness is trivially true here since
    there are no citations to hallucinate.
    """
    prompt = CONCEPTUAL_PROMPT_TEMPLATE.format(summary=summary, question=question)
    client = _get_client()
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    answer = response.text.strip()

    return {
        "answer": answer,
        "faithfulness": {
            "citations_found": [],
            "ungrounded_citations": [],
            "is_faithful": True,
            "has_citations": False,
        },
        "answer_type": "conceptual",
    }
