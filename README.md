# Codebase RAG

An AST-aware retrieval system over source code. Instead of naive fixed-size
text chunking (which cuts functions in half), this parses code into its
syntax tree and extracts whole functions/classes/methods as chunks, with
metadata (file path, line range, parent class, docstring) attached.

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
