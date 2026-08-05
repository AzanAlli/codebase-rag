"""
Multi-hop query decomposition: handles compound questions like "where is
auth handled and how does it interact with the session store", where a
single embedding of the whole question tends to blur two genuinely
distinct things being asked about, and no single retrieval pass finds
good matches for both.

One Gemini call decides whether decomposition is needed at all. For a
simple, single-topic question, it just returns the original question
unchanged — so this degrades gracefully rather than adding overhead to
every query. This mirrors the same "don't force complexity where it isn't
needed" reasoning as core/conceptual.py's routing.
"""

import json
import os
import re

from dotenv import load_dotenv
from google import genai

load_dotenv()

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ["GCP_REGION"]
GENERATION_MODEL = "gemini-2.5-flash"

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)
    return _client


DECOMPOSITION_PROMPT = """You are helping search a codebase. Decide whether
the question below asks about ONE thing, or asks about MULTIPLE distinct
things that would likely be found in different parts of the code (e.g.
"how does X work AND how does it interact with Y" asks about two things).

If it asks about only one thing, respond with a JSON array containing just
the original question, unchanged.

If it asks about multiple things, break it into 2-3 focused sub-questions,
each one specific enough to search for on its own. Do not add questions
that weren't implied by the original.

Respond with ONLY a JSON array of strings, nothing else.

Question: {question}

JSON array:"""


def decompose_query(question: str) -> list[str]:
    """
    Returns a list of sub-questions to search for. For a simple question,
    this is just [question] — a single-element list, so callers can
    always treat the result uniformly (loop over it) without a special
    case for "wasn't decomposed."
    """
    client = _get_client()
    prompt = DECOMPOSITION_PROMPT.format(question=question)

    try:
        response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
        text = response.text.strip()
        # strip markdown code fences if the model wrapped the JSON in them
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        sub_questions = json.loads(text)

        if not isinstance(sub_questions, list) or not sub_questions:
            return [question]
        return [str(q) for q in sub_questions]
    except Exception as e:
        print(f"  [warn] query decomposition failed, using original question: {e}")
        return [question]
