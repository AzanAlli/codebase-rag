"""
FastAPI wrapper around the retrieval + generation pipeline.

Run with:
    uvicorn api.main:app --reload

Then POST to /query:
    curl -X POST http://localhost:8000/query \\
        -H "Content-Type: application/json" \\
        -d '{"question": "how does click parse command line arguments"}'
"""

import sys
from pathlib import Path

# allow running as `uvicorn api.main:app` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from pydantic import BaseModel

from core.retrieval import hybrid_search
from core.generation import generate_answer

app = FastAPI(title="Codebase RAG", description="Ask questions about a codebase, grounded in AST-aware retrieval.")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    file_path: str
    symbol_name: str
    symbol_type: str
    parent_class: str | None
    start_line: int
    end_line: int
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    is_faithful: bool
    citations_found: list[str]
    ungrounded_citations: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    results = hybrid_search(req.question, top_k=req.top_k)
    chunks = [chunk for chunk, _score in results]

    generation = generate_answer(req.question, chunks)

    sources = [
        Source(
            file_path=chunk["file_path"],
            symbol_name=chunk["symbol_name"],
            symbol_type=chunk["symbol_type"],
            parent_class=chunk.get("parent_class"),
            start_line=chunk["start_line"],
            end_line=chunk["end_line"],
            relevance_score=float(score),
        )
        for chunk, score in results
    ]

    return QueryResponse(
        answer=generation["answer"],
        sources=sources,
        is_faithful=generation["faithfulness"]["is_faithful"],
        citations_found=generation["faithfulness"]["citations_found"],
        ungrounded_citations=generation["faithfulness"]["ungrounded_citations"],
    )
