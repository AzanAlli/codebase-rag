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

## Status: Day 4 complete — generation + API layer, with a working faithfulness check

- ✅ Day 1: AST-aware chunking (tree-sitter) — 1,268 chunks from `pallets/click`
- ✅ Day 2: embeddings (Vertex AI `text-embedding-004`) + storage (Supabase/pgvector), retrieval sanity-checked
- ✅ Day 3: hybrid search (vector + full-text, fused via RRF) + cross-encoder reranking — confirmed to fix the implementation-vs-test confusion found in Day 2
- ✅ Day 4: Gemini generation with file/line citations, wrapped in a FastAPI `/query` endpoint, with an automated faithfulness check

## Day 1: AST-aware chunking

Most "RAG over code" tutorials split files into fixed-size text blocks (e.g.
every 500 characters). That's fast to build but breaks the moment a chunk
boundary lands in the middle of a function — the model retrieves half a
function body with no idea what the other half does, and structural context
(what class a method belongs to, what a function's docstring says) is lost
entirely.

`ingestion/parser.py` fixes this by walking each file's actual **abstract
syntax tree** (via `tree-sitter`) instead of its raw text. It extracts
`function_definition` and `class_definition` nodes directly, which means:

- **Every chunk is a complete, syntactically valid unit** — a whole function, method, or class, never a fragment
- **Docstrings are extracted separately from source**, not left buried inside the code — this matters for embedding later, since a docstring in plain English ("Parses positional arguments...") is often a better semantic match for a natural-language query than the code itself
- **Parent-class attribution**: a method inside a class is tagged with `parent_class`, so retrieval can distinguish `Context.call_on_close` from an unrelated top-level `call_on_close` elsewhere
- **Exact line ranges** (`start_line`, `end_line`) are captured per chunk, which is what lets later stages cite "this is handled in `core.py`, lines 45–60" instead of a vague file-level pointer
- Each chunk gets a stable `chunk_id` (a short hash of file path + line + symbol name), so re-parsing the same file twice produces consistent IDs — useful later for incremental re-indexing without duplicating rows

**Tested against a real repo, not a toy example:** [`pallets/click`](https://github.com/pallets/click) (a well-known, moderately sized Python CLI library — 31 files). Parsing it end-to-end produced:

| Chunk type | Count |
|---|---|
| functions | 749 |
| methods | 402 |
| classes | 117 |
| **total** | **1,268** |

Spot-checking individual chunks (e.g. `Context.call_on_close` in
`src/click/core.py`) confirmed correct docstring extraction, accurate line
ranges, and correct parent-class attribution — no manual correction needed.

**Design decisions worth calling out:**
- Nested functions (a function defined inside another function) are *not* recursed into as separate chunks — they stay embedded in their parent's `source`, since a nested helper rarely makes sense as a standalone retrieval unit divorced from the function that uses it
- Currently Python-only (`function_definition` / `class_definition` node types are language-specific in tree-sitter); the `LANGUAGE_CONFIG` dict in `parser.py` is structured so adding JS/TS/Go support later is a matter of adding new node-type mappings, not rewriting the walker

## Setup

```bash
pip install -r requirements.txt
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

# 5. add full-text search support (one-time migration)
python3 ingestion/add_fulltext_search.py

# 6. hybrid search + reranking — compares against vector-only baseline
python3 ingestion/hybrid_search.py "how does click parse command line arguments"

# 7. run the full API (retrieval + generation + faithfulness check)
uvicorn api.main:app --reload
# in another terminal:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "how does click parse command line arguments"}'
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

## Day 3: hybrid search + cross-encoder reranking

Day 2's retrieval worked, but had a real limitation: pure vector similarity
doesn't distinguish *code that implements something* from *code that merely
talks about the same concept* — a test function asserting behaviour of a
parser scores nearly as high as the parser itself, since both use similar
language.

Day 3 addresses this with a three-stage pipeline:

1. **Vector search** (same as Day 2) — top 20 candidates by cosine similarity
2. **Full-text search** — a Postgres `tsvector` column (`ingestion/add_fulltext_search.py`) indexed with GIN, weighted so a match on `symbol_name` counts more than a match buried in `source`. This catches exact-term matches (e.g. someone searching for `parse_args` by name) that embeddings alone can miss or under-rank.
3. **Reciprocal Rank Fusion (RRF)** — merges the two ranked lists without needing to hand-tune relative weights: a chunk's fused score is the sum of `1/(k + rank)` across every list it appears in, so anything ranking well in *either* list (or both) rises to the top.
4. **Cross-encoder reranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — the fused top 10 are rescored by a cross-encoder, which reads the query and each candidate together (rather than comparing two independent vectors), giving a much more accurate relevance judgment. This is too slow to run over the whole table, which is why it only reranks a short fused list.

### Measured result

Same query, same underlying data, before and after — run yourself with `python3 ingestion/hybrid_search.py "how does click parse command line arguments"`:

| Rank | Day 2 (vector only) | Day 3 (hybrid + reranked) |
|---|---|---|
| 1 | ❌ `test_nargs_envvar` (test) | ✅ `parse_args` (`_OptionParser`) |
| 2 | ✅ `parse_args` (`_OptionParser`) | ✅ `Command` (class) |
| 3 | ❌ `test_command_to_info_dict_multiple_arguments` (test) | ✅ `_OptionParser` (class) |
| 4 | ✅ `__init__` (`_OptionParser`) | ✅ `main` (`Command`) |
| 5 | ❌ `test_unprocessed_options` (test) | ✅ `__init__` (`_OptionParser`) |

Day 2's top 5 had 3 test functions crowding out real implementation, and the
correct answer (`parse_args`) only placed 2nd. Day 3's top 5 is **100% real
implementation, zero tests**, with `parse_args` correctly placed 1st.

## Day 4: generation + API

Retrieval alone isn't a usable tool — Day 4 wires the retrieved chunks into
a Gemini prompt that answers the user's question, with a citation after
every claim in `(file_path:start_line-end_line)` format, then wraps the
whole pipeline in a FastAPI endpoint.

**Architecture:**
- `core/retrieval.py` — the Day 3 hybrid search pipeline, extracted into a shared module (previously duplicated between the CLI script and what would've become the API)
- `core/generation.py` — builds a grounded prompt from retrieved chunks, calls Gemini (via the current `google-genai` SDK — the older `vertexai.generative_models` module is on Google's deprecation path), and checks the answer's faithfulness
- `api/main.py` — a FastAPI app exposing `POST /query`, returning the answer, the source chunks used, and the faithfulness report

**Faithfulness check:** rather than just trusting Gemini's citations, we parse every `(file:start-end)` citation out of the generated answer and verify it falls *inside* the line range of a chunk that was actually retrieved. A citation pointing somewhere outside every retrieved chunk would indicate hallucination — the model citing a location it wasn't actually given.

**A real bug found and fixed during testing:** the first version of the citation-extraction regex assumed each parenthetical contained exactly one citation, e.g. `(file.py:12-34)`. Gemini often lists several citations in one parenthetical instead, e.g. `(a.py:12-34, b.py:56-78)` — which silently failed to match at all under the old regex, since it required a `)` immediately after the first citation. This meant real, valid citations were being dropped and wrongly reported as failures. Fixed by matching the citation pattern directly rather than anchoring on the surrounding parentheses. Worth calling out as an example of why testing the full pipeline against a real query (not just unit-testing each piece in isolation) caught a bug that would otherwise have quietly under-reported faithfulness.

**Example**, run via `curl -X POST http://localhost:8000/query -d '{"question": "how does click parse command line arguments"}'`: Gemini correctly traced the real call flow — `Command.main` → `make_context` → `Command.parse_args` → `_OptionParser.parse_args` — citing exact line ranges for each step, all of which checked out as grounded in the retrieved chunks (`is_faithful: true`).

**Known limitation:** the faithfulness check only recognizes citations in `file:start-end` range format; Gemini occasionally cites a single line number instead (e.g. `file.py:1479`), which the current regex doesn't parse. Rare in practice, but worth tightening the prompt or the regex if it comes up often in the eval harness.



- Eval harness: build a labeled query set (20–50 questions) and measure precision@k / faithfulness across all of them, not just the one query walked through above
- Call-graph awareness, multi-hop query decomposition
- Frontend: Monaco code viewer + retrieval trace visualization