"""
Day 6, step 1: build a lightweight call graph.

This is a NAME-BASED HEURISTIC, not full semantic analysis — worth being
upfront about the limitation. For each function/method chunk, we walk its
AST for `call` expressions (e.g. `parse_args(...)`, `self.make_parser(...)`)
and extract the called name (the trailing identifier — `parse_args`,
`make_parser`). We then match that name against known symbol_names in the
same repo. This will have false positives (two unrelated classes with a
same-named method both match) and can't resolve dynamic dispatch or
imports from outside the indexed repo — but it's genuinely useful for
surfacing "this function calls that one" relationships that pure
text/vector search has no way to see at all.

Note on ambiguous names: when a called name matches multiple chunks (e.g.
a common method name like `__init__` defined in many classes), we store
an edge to EACH match, since we have no way to know which one is actually
called without real scope resolution. Downstream consumers should treat
multiple callees for one name as "possible calls," not a certain one. This
is also why the table can't use (caller_id, callee_name) as a primary
key — that would silently collapse ambiguous matches down to just one.

Usage:
    python3 ingestion/build_call_graph.py
"""

import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tree_sitter_languages import get_parser

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

TABLE_DDL = """
DROP TABLE IF EXISTS call_edges;
CREATE TABLE call_edges (
    id SERIAL PRIMARY KEY,
    caller_id TEXT NOT NULL,
    callee_id TEXT,
    callee_name TEXT NOT NULL
);
CREATE INDEX call_edges_caller_idx ON call_edges (caller_id);
CREATE INDEX call_edges_callee_idx ON call_edges (callee_id);
"""

INSERT_SQL = """
INSERT INTO call_edges (caller_id, callee_id, callee_name)
VALUES (%s, %s, %s);
"""


def fetch_all_chunks(conn) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT id, file_path, symbol_name, symbol_type, source FROM code_chunks;")
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    return rows


def extract_called_names(source: str, parser) -> set[str]:
    """Walk the AST of one chunk's source, return the set of called names."""
    tree = parser.parse(source.encode("utf-8"))
    called_names = set()

    def get_call_name(func_node) -> str | None:
        # simple call: foo(...)  -> identifier node directly
        if func_node.type == "identifier":
            return func_node.text.decode("utf-8")
        # attribute call: self.foo(...) or obj.foo(...) -> the "attribute" field
        # holds the trailing name (the part after the dot), which is what we want
        if func_node.type == "attribute":
            attr_node = func_node.child_by_field_name("attribute")
            if attr_node is not None:
                return attr_node.text.decode("utf-8")
        return None

    def walk(node):
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node is not None:
                name = get_call_name(func_node)
                if name:
                    called_names.add(name)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return called_names


def main():
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    print("Creating call_edges table...")
    cur.execute(TABLE_DDL)
    conn.commit()

    print("Fetching all chunks...")
    chunks = fetch_all_chunks(conn)
    print(f"Loaded {len(chunks)} chunks")

    # build a name -> [chunk_id, ...] index for resolving callees
    name_index: dict[str, list[str]] = {}
    for c in chunks:
        name_index.setdefault(c["symbol_name"], []).append(c["id"])

    parser = get_parser("python")

    edges = []
    for chunk in chunks:
        if chunk["symbol_type"] not in ("function", "method"):
            continue  # only functions/methods have meaningful call bodies
        try:
            called_names = extract_called_names(chunk["source"], parser)
        except Exception as e:
            print(f"  [warn] failed to parse calls in {chunk['symbol_name']}: {e}")
            continue

        for name in called_names:
            if name == chunk["symbol_name"]:
                continue  # skip self-recursion for now, not interesting for context expansion
            matches = name_index.get(name)
            if matches:
                # ambiguous names (multiple chunks share it) -> record an edge per match
                for callee_id in matches:
                    edges.append((chunk["id"], callee_id, name))
            else:
                # unresolved call (external library, builtin, etc.) -> still record the name, no callee_id
                edges.append((chunk["id"], None, name))

    print(f"Extracted {len(edges)} call edges. Writing to database...")
    psycopg2.extras.execute_batch(cur, INSERT_SQL, edges, page_size=500)
    conn.commit()

    cur.execute("SELECT count(*) FROM call_edges WHERE callee_id IS NOT NULL;")
    resolved = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM call_edges;")
    total = cur.fetchone()[0]
    print(f"Done. {resolved}/{total} edges resolved to a known chunk in this repo.")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
