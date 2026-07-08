"""
AST-aware code chunker.

Instead of splitting files into fixed-size text blocks (which cuts functions
in half and destroys semantic meaning), this walks the tree-sitter AST and
extracts whole functions/classes/methods as chunks, along with metadata
that downstream retrieval and re-ranking can use:

- file path
- start_line / end_line
- symbol name (function/class/method name)
- symbol type (function, method, class)
- parent class (if it's a method)
- docstring (extracted separately, useful for retrieval matching on intent)
- the raw source text of the chunk (what actually gets embedded)

Currently supports Python. Designed so new languages can be added by
implementing a LANGUAGE_CONFIG entry with the right tree-sitter node types.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter_languages import get_parser


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str  # "function" | "method" | "class"
    parent_class: str | None
    start_line: int
    end_line: int
    docstring: str | None
    source: str
    language: str = "python"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "parent_class": self.parent_class,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "source": self.source,
            "language": self.language,
        }


# Node types we care about extracting as standalone chunks, per language.
LANGUAGE_CONFIG = {
    "python": {
        "function_node": "function_definition",
        "class_node": "class_definition",
    }
}


def _make_chunk_id(file_path: str, start_line: int, symbol_name: str) -> str:
    raw = f"{file_path}:{start_line}:{symbol_name}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _extract_docstring(func_or_class_node, source_bytes: bytes) -> str | None:
    """Python docstring = first statement in the body, if it's a string expression."""
    body = None
    for child in func_or_class_node.children:
        if child.type == "block":
            body = child
            break
    if body is None:
        return None
    for stmt in body.children:
        if stmt.type == "expression_statement":
            for c in stmt.children:
                if c.type == "string":
                    text = _get_node_text(c, source_bytes)
                    return text.strip("\"'").strip()
        # first real statement wasn't a string -> no docstring
        if stmt.type not in ("comment",):
            break
    return None


def _get_name(node) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return None


def parse_file(file_path: Path, repo_root: Path) -> list[CodeChunk]:
    """Parse a single Python file into function/class/method-level chunks."""
    source_bytes = file_path.read_bytes()
    parser = get_parser("python")
    tree = parser.parse(source_bytes)
    root = tree.root_node
    rel_path = str(file_path.relative_to(repo_root))

    chunks: list[CodeChunk] = []

    def walk(node, parent_class: str | None):
        if node.type == "class_definition":
            class_name = _get_name(node) or "<unknown_class>"
            docstring = _extract_docstring(node, source_bytes)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunks.append(
                CodeChunk(
                    chunk_id=_make_chunk_id(rel_path, start_line, class_name),
                    file_path=rel_path,
                    symbol_name=class_name,
                    symbol_type="class",
                    parent_class=None,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=docstring,
                    source=_get_node_text(node, source_bytes),
                )
            )
            # recurse into class body to find methods, tagging parent_class
            for child in node.children:
                walk(child, parent_class=class_name)
            return

        if node.type == "function_definition":
            func_name = _get_name(node) or "<unknown_function>"
            docstring = _extract_docstring(node, source_bytes)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            symbol_type = "method" if parent_class else "function"
            chunks.append(
                CodeChunk(
                    chunk_id=_make_chunk_id(rel_path, start_line, func_name),
                    file_path=rel_path,
                    symbol_name=func_name,
                    symbol_type=symbol_type,
                    parent_class=parent_class,
                    start_line=start_line,
                    end_line=end_line,
                    docstring=docstring,
                    source=_get_node_text(node, source_bytes),
                )
            )
            # don't recurse into nested functions for now -> keeps chunks clean
            return

        for child in node.children:
            walk(child, parent_class)

    walk(root, parent_class=None)
    return chunks


def parse_repo(repo_root: Path, extensions=(".py",)) -> list[CodeChunk]:
    """Walk a repo and parse every matching file into chunks."""
    all_chunks: list[CodeChunk] = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}

    for path in repo_root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix not in extensions:
            continue
        try:
            chunks = parse_file(path, repo_root)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [warn] failed to parse {path}: {e}")

    return all_chunks


if __name__ == "__main__":
    import json
    import sys

    repo_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_repos/click")
    chunks = parse_repo(repo_path)

    print(f"Parsed {len(chunks)} chunks from {repo_path}")
    print(f"  functions: {sum(1 for c in chunks if c.symbol_type == 'function')}")
    print(f"  methods:   {sum(1 for c in chunks if c.symbol_type == 'method')}")
    print(f"  classes:   {sum(1 for c in chunks if c.symbol_type == 'class')}")

    out_path = Path("chunks.jsonl")
    with out_path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    print(f"\nWrote chunks to {out_path}")

    # show a sample
    print("\n--- sample chunk ---")
    sample = next((c for c in chunks if c.symbol_type == "method" and c.docstring), chunks[0])
    print(json.dumps(sample.to_dict(), indent=2)[:800])
