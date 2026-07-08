# Codebase RAG

Retrieval-augmented generation over a codebase, built to answer questions
like *"where is rate limiting implemented?"* or *"what does this decorator
do?"* with real, citable answers pointing at exact files and line numbers —
not a hallucinated guess.

Most RAG-over-code demos chunk files into fixed-size text blocks, which
routinely slices a function in half and destroys its meaning. This project
instead parses each file into its AST (via `tree-sitter`) and extracts whole
functions, classes, and methods as chunks, carrying structural metadata
(file path, line range, parent class, docstring) that a text splitter simply
doesn't have access to.

**Stack:** tree-sitter · Vertex AI embeddings · pgvector · Gemini · FastAPI

## Status: Day 1 complete — ingestion + chunking

## Setup

```bash
pip install tree-sitter==0.21.3 tree-sitter-languages --break-system-packages
```

(Note: pin `tree-sitter` to 0.21.3 — newer versions break the API that
`tree-sitter-languages` expects.)

## Usage

```bash
# clone any repo you want to test on
git clone --depth 1 https://github.com/pallets/click.git test_repos/click

# parse it into chunks
python3 ingestion/parser.py test_repos/click
```

This writes `chunks.jsonl` — one JSON object per function/class/method,
containing:

- `file_path`, `start_line`, `end_line`
- `symbol_name`, `symbol_type` (function / method / class)
- `parent_class` (if it's a method)
- `docstring` (extracted separately from source, useful for retrieval)
- `source` (the actual code, what gets embedded)

## Next steps (Day 2+)

- Embed chunks via Vertex AI, store in pgvector (Supabase or Neon free tier)
- Hybrid retrieval: pgvector cosine similarity + Postgres full-text search
- Cross-encoder reranking
- Gemini generation with file/line citations
- Eval harness: precision@k, faithfulness, hybrid vs. embeddings-only baseline
- Call-graph awareness, multi-hop query decomposition
- Frontend: Monaco code viewer + retrieval trace visualization
