import json
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
json_path = base_dir / "docs" / "perf" / "proc1_unset_fwd.json"

with open(json_path, "r", encoding="utf-8") as f:
    d = json.load(f)

print("=" * 80)
print("TRANSLATOR STRINGS LOGGED IN PROC1 (COMMIT D / PERF_BENCHMARK)")
print("=" * 80)

found = 0
for q_str, q_val in d["queries"].items():
    meta = q_val.get("metadata", {})
    if "translator_input" in meta or "translator_output" in meta:
        found += 1
        print(f"Query: '{q_str}'")
        print(f"  Input:  {meta.get('translator_input')!r}")
        print(f"  Output: {meta.get('translator_output')!r}")

print(f"\nTotal queries with translator calls logged: {found}")
