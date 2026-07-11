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

## Status: Day 2 complete — embeddings + vector retrieval verified

- ✅ Day 1: AST-aware chunking (tree-sitter)
- ✅ Day 2: embeddings (Vertex AI `text-embedding-004`) + storage (Supabase/pgvector), retrieval sanity-checked

## Setup

```bash
pip install tree-sitter==0.21.3 tree-sitter-languages google-cloud-aiplatform psycopg2-binary python-dotenv --break-system-packages
```

(Note: pin `tree-sitter` to 0.21.3 — newer versions break the API that
`tree-sitter-languages` expects.)

Copy `.env.example` to `.env` and fill in:
- `DATABASE_URL` — Supabase Postgres connection string (Session Pooler URI recommended over Direct/Transaction, since it proxies IPv4 for free and avoids IPv6-only connectivity issues)
- `GCP_PROJECT_ID`, `GCP_REGION` — your GCP project with the Vertex AI API (`aiplatform.googleapis.com`) enabled

Requires `gcloud auth application-default login` once, so the Vertex AI SDK can authenticate locally.

## Usage

```bash
# 1. clone any repo you want to test on
git clone --depth 1 https://github.com/pallets/click.git test_repos/click

# 2. parse it into AST-aware chunks
python3 ingestion/parser.py test_repos/click
# writes chunks.jsonl

# 3. embed all chunks and store them in Supabase/pgvector
python3 ingestion/embed.py chunks.jsonl

# 4. sanity-check retrieval with a real query
python3 ingestion/test_retrieval.py "how does click parse command line arguments"
```

`chunks.jsonl` contains one JSON object per function/class/method:

- `file_path`, `start_line`, `end_line`
- `symbol_name`, `symbol_type` (function / method / class)
- `parent_class` (if it's a method)
- `docstring` (extracted separately from source, useful for retrieval)
- `source` (the actual code, what gets embedded)

Each chunk is embedded as a combination of symbol info + docstring + source
(not just raw code), so a query in plain English can match on intent even
when the code itself uses different terminology.

## Known limitation (motivates Day 3)

Pure semantic (embeddings-only) search doesn't reliably distinguish
*implementation* from *tests* — a query like "how does click parse command
line arguments" surfaces the correct function (`parse_args` in
`_OptionParser`) but also several `test_*` functions that merely discuss the
same concepts. Hybrid search (vector + keyword/BM25) and cross-encoder
reranking should sharpen this.

## Next steps (Day 3+)

- Hybrid retrieval: pgvector cosine similarity + Postgres full-text search (BM25-ish), to fix the implementation-vs-test confusion above
- Cross-encoder reranking on top-k candidates
- Gemini generation with file/line citations
- Eval harness: precision@k, faithfulness, hybrid vs. embeddings-only baseline (quantify the improvement above)
- Call-graph awareness, multi-hop query decomposition
- Frontend: Monaco code viewer + retrieval trace visualization
