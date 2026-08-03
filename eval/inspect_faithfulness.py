"""
Diagnostic: print the ungrounded citations (and a snippet of surrounding
answer text) for every eval query, so we can see the actual failure
pattern instead of guessing.
"""

import json

with open("eval/eval_details.json") as f:
    details = json.load(f)

for i, record in enumerate(details, 1):
    status = "FAITHFUL" if record["is_faithful"] else "UNFAITHFUL"
    print(f"[{i}] {status}")
    print(f"    all citations found: {record['citations_found']}")
    if record["ungrounded_citations"]:
        print(f"    UNGROUNDED: {record['ungrounded_citations']}")
    if not record["has_citations"]:
        print(f"    (no citations extracted at all)")
    print()