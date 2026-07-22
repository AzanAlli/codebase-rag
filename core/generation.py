"""
Generation layer: takes a user query + retrieved chunks, asks Gemini to
answer using ONLY that context, with file/line citations. Uses the Google
Gen AI SDK (google-genai), not the deprecated vertexai.generative_models
module.

Also includes a lightweight faithfulness check: does the answer actually
cite files/lines that were part of the retrieved context, or did the model
reference something not actually provided (a sign of hallucination)?
"""

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


PROMPT_TEMPLATE = """You are a code assistant answering questions about a codebase using ONLY the context below.

Rules:
- Answer using only the information in the provided code chunks.
- Every claim you make must be followed by a citation in the exact format (file_path:start_line-end_line).
- If the context doesn't contain enough information to answer, say so directly instead of guessing.
- Be concise — a few sentences, not an essay.

Context:
{context}

Question: {question}

Answer:"""


def build_context(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        location = f"{chunk['symbol_type']} {chunk['symbol_name']}"
        if chunk.get("parent_class"):
            location += f" (in class {chunk['parent_class']})"
        header = f"### {location} — {chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
        docstring = f"Docstring: {chunk['docstring']}" if chunk.get("docstring") else ""
        code = f"```python\n{chunk['source']}\n```"
        parts.append("\n".join(filter(None, [header, docstring, code])))
    return "\n\n".join(parts)


def extract_citations(answer: str) -> list[str]:
    """
    Pull out file_path:start-end style citations from the generated answer.
    Matches the citation pattern directly rather than anchoring on the
    surrounding parentheses, since Gemini sometimes lists multiple
    citations in one parenthetical (e.g. "(a.py:1-2, b.py:3-4)"), which a
    parenthesis-anchored regex would fail to split correctly.
    """
    return re.findall(r"[\w./\\-]+\.py:\d+-\d+", answer)


def check_faithfulness(answer: str, chunks: list[dict]) -> dict:
    """
    Lightweight faithfulness check: every citation in the answer should
    fall within the range of a chunk that was actually part of the
    retrieved context. Note this is containment, not exact match — Gemini
    is often given a large chunk (e.g. a whole class) and will cite a more
    precise sub-range within it for a specific claim, which is more useful
    than citing the whole chunk and should count as grounded, not flagged.
    """
    cited = extract_citations(answer)

    def is_grounded(citation: str) -> bool:
        file_path, line_range = citation.rsplit(":", 1)
        cite_start, cite_end = (int(x) for x in line_range.split("-"))
        for c in chunks:
            if c["file_path"] != file_path:
                continue
            if c["start_line"] <= cite_start and cite_end <= c["end_line"]:
                return True
        return False

    ungrounded = [c for c in cited if not is_grounded(c)]

    return {
        "citations_found": cited,
        "ungrounded_citations": ungrounded,
        "is_faithful": len(ungrounded) == 0 and len(cited) > 0,
        "has_citations": len(cited) > 0,
    }


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Generate an answer to `question` grounded in `chunks`. Returns the
    answer text plus a faithfulness report.
    """
    if not chunks:
        return {
            "answer": "No relevant code was found for this question.",
            "faithfulness": {"citations_found": [], "ungrounded_citations": [], "is_faithful": False, "has_citations": False},
        }

    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    client = _get_client()
    response = client.models.generate_content(model=GENERATION_MODEL, contents=prompt)
    answer = response.text.strip()

    faithfulness = check_faithfulness(answer, chunks)

    return {"answer": answer, "faithfulness": faithfulness}