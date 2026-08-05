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

## Status: Day 7 complete — multi-hop query decomposition + closed a known citation gap

- ✅ Day 1: AST-aware chunking (tree-sitter) — 1,268 chunks from `pallets/click`
- ✅ Day 2: embeddings (Vertex AI `text-embedding-004`) + storage (Supabase/pgvector), retrieval sanity-checked
- ✅ Day 3: hybrid search (vector + full-text, fused via RRF) + cross-encoder reranking — confirmed to fix the implementation-vs-test confusion found in Day 2
- ✅ Day 4: Gemini generation with file/line citations, wrapped in a FastAPI `/query` endpoint, with an automated faithfulness check
- ✅ Day 5: synthetic eval harness — 20 real questions, hybrid pipeline beats vector-only baseline on hit_rate@1 and MRR, 100% generation faithfulness after fixing two real bugs surfaced by testing at scale
- ✅ Day 6: name-based call graph (surfaces caller/callee context automatically) + audience-aware routing (plain-language answers for non-technical questions, e.g. from a recruiter, instead of forcing code citations onto every answer)
- ✅ Day 7: multi-hop query decomposition for compound questions, plus closed a known citation-parsing gap (single-line citations) — re-verified against the full eval set with zero regression

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

# 8. or use the CLI (no server needed)
python3 cli.py
python3 cli.py "how does click parse command line arguments"

# 9. eval harness: generate a synthetic query set, then score both pipelines
python3 eval/generate_eval_set.py --n 20
python3 eval/run_eval.py

# 10. build the call graph (name-based caller/callee relationships)
python3 ingestion/build_call_graph.py
python3 cli.py --graph "how does click parse command line arguments"

# 11. generate a plain-language project summary, for non-technical questions
python3 ingestion/generate_project_summary.py test_repos/click
python3 cli.py "what does this project do"

# 12. multi-hop: compound questions get decomposed and merged automatically
python3 cli.py --multihop "how does click parse arguments and how does it format help text"
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

**Follow-up fix — closing the test-vs-implementation gap for real:** testing the CLI against a second query ("how does click format help text") showed the Day 3 hybrid+rerank pipeline *reduces* but doesn't eliminate test functions leaking into results — `test_help_formatter_write_text` and `test_context_formatter_class` still ranked #2 and #3, since a test that directly calls and asserts on a function is genuinely close in both meaning and wording to the function itself. No amount of ranking tuning fully separates two things that really are about the same code.

Since `file_path` already unambiguously identifies test code (anything under `tests/`), the fix is to filter explicitly rather than rely on semantic ranking to sort it out: `core/retrieval.py` now excludes `tests/` chunks from both vector and full-text search **by default**, with an automatic override — if the query itself mentions "test" or "testing", test files are included instead, since that's clearly what the user wants in that case.

Re-running the same query after the fix: all 5 sources are now real implementation, zero test functions (`format_help_text`, `make_formatter`, `format_help`, `Command`, `make_default_short_help`), and the answer remained fully faithful.

Also added: `cli.py`, an interactive/single-query command-line demo that calls the pipeline directly — no server or curl needed — closing out the original "minimal frontend or CLI for demo purposes" scope for this phase.

## Day 5: eval harness

Everything up to Day 4 was validated on one or two example queries — useful
for finding bugs, but not a real measurement. Day 5 builds a proper eval
set and scores both retrieval and generation across it.

**Building the eval set without hand-labeled guesses:** rather than me
writing "ground truth" questions from memory of what I think `click`'s
internals do (risky — I could just be wrong about a codebase I didn't
write), `eval/generate_eval_set.py` samples real chunks from the database
and asks Gemini to write a natural-language question that chunk answers,
without naming the function directly. This produces `{query,
expected_chunk_id}` pairs that are grounded in the actual indexed code,
not my assumptions about it.

**Metrics:**
- **Retrieval**: hit rate @ k (1, 3, 5, 10) and Mean Reciprocal Rank (MRR), comparing the Day 2 vector-only baseline against the full Day 3/4 hybrid+rerank+test-filter pipeline
- **Generation**: faithfulness rate — the fraction of generated answers where every citation is grounded in a chunk that was actually retrieved

### Results (20 real questions against `pallets/click`)

| Metric | Baseline (vector only) | Hybrid + rerank + filter |
|---|---|---|
| hit_rate@1 | 0.750 | **0.850** ↑ |
| hit_rate@3 | 0.950 | 0.950 |
| hit_rate@5 | 0.950 | **1.000** ↑ |
| hit_rate@10 | 1.000 | 1.000 |
| MRR | 0.850 | **0.904** ↑ |

**Faithfulness: 20/20 (100%)** — every generated answer's citations were grounded in the retrieved context, no hallucinated locations.

### Two more real bugs found and fixed while building this

Running the eval against a real, larger query set (rather than one or two
hand-picked examples) surfaced two genuine bugs in the faithfulness
checker — a good example of why an eval harness earns its keep beyond just
producing a headline number:

1. **Module import path bug**: `eval/run_eval.py` initially failed with `ModuleNotFoundError: No module named 'core'` when run as `python3 eval/run_eval.py` from the project root, since Python only adds the *script's own directory* to the import path, not the project root. Fixed by inserting the project root into `sys.path` at the top of the script (the same fix already used in `api/main.py`).

2. **File path truncation in repeat citations**: the first real eval run showed faithfulness collapse to 50%, then 20% on a second run — and critically, the *same* questions flipped between faithful/unfaithful across runs, which is a strong signal of a checker bug rather than genuine hallucination (real hallucination on unchanged context wouldn't flip like that). Inspecting the raw citation data (`eval/inspect_faithfulness.py`) showed the pattern clearly: Gemini often cites a file's full path once (`src/click/core.py:1132-1141`) and then shortens it to just the basename for a follow-up claim about the same file (`core.py:1136-1139`). The faithfulness checker required an *exact* string match on `file_path`, so the shortened citation never matched, even though it was the same file and a valid sub-range. Fixed by matching on path suffix/basename (`core/generation.py`'s `_files_match`) instead of exact equality — confirmed with a standalone test before shipping it, then reran the full eval set to verify: faithfulness went from 20% → 100% with zero other changes.

## Day 6: call-graph awareness + audience-aware routing

### Call-graph awareness

Retrieval so far treats every chunk independently — it has no notion that
`Command.main` calls `parse_args`, which calls `_process_args_for_options`.
For questions that trace a multi-step process, the retriever has to get
lucky and rank every step highly on its own.

`ingestion/build_call_graph.py` builds a lightweight, **name-based**
call graph (not full semantic analysis — worth stating clearly): for each
function/method chunk, it walks the AST for `call` expressions and matches
the called name against known symbols in the same repo. Run against
`pallets/click`: **7,917 of 9,487 extracted calls (83%) resolved** to a
known chunk in the repo, the rest being external/builtin calls (`len()`,
`isinstance()`, etc.) that a name-based approach can't and shouldn't try to
resolve.

`core/retrieval.py`'s `expand_with_call_graph()` uses this: given the
top-ranked retrieved chunks, it pulls in their immediate callers/callees
(up to a small cap) and tags each with which chunk it's related to, so a
retrieval-trace UI (or just this README) can show *why* it was included.

**Measured effect**, same query as throughout this README
(`python3 cli.py --graph "how does click parse command line arguments"`):
without call-graph expansion, the answer described the high-level flow but
stopped at "processes options, then positional arguments" — with
expansion, the two helper methods that actually implement each step
(`_process_args_for_options`, `_process_args_for_args`) got pulled in
automatically and the answer became substantively more detailed and
accurate, still 100% faithful. The retriever didn't need to get lucky
ranking those helpers on their own; the graph surfaced them because they
were structurally connected to what was already found relevant.

**A real bug caught before shipping**: the first version of `call_edges`
used `(caller_id, callee_name)` as a primary key, which silently collapsed
ambiguous call targets (e.g. a common method name like `__init__` defined
in dozens of classes) down to whichever match got written last — losing
roughly half the extracted edges (9,487 extracted, only 4,078 stored).
Fixed by switching to a surrogate key, since multiple callees for one
ambiguous name are a legitimate result, not a duplicate to be
deduplicated away.

### Audience-aware routing

Not every question about this project is technical. A recruiter or
non-engineering interviewer asking "what does this do?" would previously
get either nothing relevant (no code chunk answers that) or a confusing,
citation-heavy technical answer — neither is useful to them.

Rather than building a bigger system (an LLM-based router, multiple
specialized retrieval indexes, conversation memory) to solve this, the
actual problem only needed two small, cheap additions:

- **`ingestion/generate_project_summary.py`**: a one-time script that reads the repo's README and asks Gemini to write a plain-English, jargon-free summary (what it is, what problem it solves, who it's for), stored for reuse
- **`core/conceptual.py`**: a regex-based classifier (`is_conceptual_query`) — deliberately a heuristic, not an LLM call, so it's free, instant, and predictable — that recognizes patterns like "what does this project do" or "why would a company use this" and routes them to a plain-language answer generated from the stored summary, skipping code retrieval and citations entirely

**Why not the bigger version:** a fuller "multi-agent routing" architecture (an LLM deciding how to route, separate specialized indexes, session memory) was considered and deliberately not built. At this project's actual scale — a single-user portfolio demo, not a live product with real traffic — the extra architecture adds real failure modes (a misrouted query, a garbled multi-agent synthesis) for a result a user can't actually tell apart from the simpler version. It would also cost measurably more per query if this became a real subscription product later (each routing/synthesis step is an extra LLM call, which is a recurring cost multiplied by every user interaction, not a one-time cost) — premature architecture here would be optimizing for scale the project doesn't have yet, at the expense of reliability it needs right now.

**Tested against three real queries:**
- *"what does this project do"* → routed to conceptual, plain language, zero jargon, zero citations
- *"why would a company use this"* → routed to conceptual, and the answer adapted its framing to the specific business-value angle of the question rather than repeating a canned summary verbatim
- *"how does click parse command line arguments"* → correctly stayed on the full technical pipeline, unchanged, still faithful — confirming the new routing doesn't interfere with real technical questions

## Day 7: multi-hop query decomposition

A compound question — "how does click parse arguments and how does it
format help text" — asks about two genuinely distinct parts of the
codebase. A single embedding of the whole sentence tends to blur both
topics together rather than finding good matches for each.

`core/decomposition.py` uses one Gemini call to decide whether a question
needs splitting, and if so, breaks it into 2-3 focused sub-questions.
`core/retrieval.py`'s `multi_hop_search()` runs `hybrid_search`
independently for each sub-question, then merges results — deduping by
chunk id, keeping each chunk's best score across every sub-question it
matched.

**Graceful degradation, verified not just claimed**: for a simple,
non-compound question, decomposition just returns the original question
unchanged, so the search runs exactly once, same as normal. Ran the same
simple query (`"how does click parse command line arguments"`) through
both `hybrid_search` directly and `multi_hop_search` — the source
rankings and relevance scores were byte-identical, confirming the
decomposition step adds zero behavior change for questions that don't
need it (just one extra classification call).

**Tested against a real compound query**
(`python3 cli.py --multihop "how does click parse arguments and how does it format help text"`):
correctly split into "how does click parse arguments" and "how does click
format help text", and the merged sources drew from **both** areas
(`_OptionParser`/`parse_args` for parsing, `format_help_text`/
`make_formatter` for help formatting) — a balance a single non-decomposed
search likely wouldn't have found, since the two topics compete for
representation in one embedding.

**A citation-parsing gap closed while testing this**: the multi-hop run
above surfaced a case documented back in Day 4 as a known limitation —
Gemini occasionally cites a single line (`core.py:1483`) instead of a
range, which the citation regex didn't parse, silently under-counting
faithfulness (a query showed "4 citations found" when the answer actually
contained 5). Fixed by extending the regex to accept an optional range
(`\d+(?:-\d+)?` instead of requiring `\d+-\d+`), and treating a
single-line citation as a zero-width range for the containment check.
Verified with a standalone test before shipping, then re-ran the full
20-question eval set to confirm no regression: retrieval numbers
unchanged, faithfulness still 20/20 (100%).

## Next steps (Day 8+)

- Multi-hop query decomposition — breaking a compound question ("where is X handled and how does it interact with Y") into sub-queries, retrieving for each, then synthesizing
- Multi-tenancy and usage tracking — real architecture needed if/when this becomes a multi-user product, deliberately deferred until there's real usage data to design around (see "Why not the bigger version" above)
- Frontend: repo picker + Monaco code viewer + retrieval trace visualization — turning this from a script into something a non-technical person could actually click through