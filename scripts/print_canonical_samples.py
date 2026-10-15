import json
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
docs_dir = base_dir / "docs" / "perf"
sys.path.insert(0, str(base_dir / "scripts"))
from strict_comparator import canonical_query_obj

with open(docs_dir / "proc1_unset_fwd.json", "r", encoding="utf-8") as f:
    p1 = json.load(f)
with open(docs_dir / "proc3_seed1.json", "r", encoding="utf-8") as f:
    p3 = json.load(f)

queries = ["oppression", "kalayaan", "tiktok"]

print("=" * 80)
print("CANONICAL JSON SAMPLES FROM PROC1 (UNSET) AND PROC3 (SEED 1)")
print("=" * 80)

for q in queries:
    c1 = canonical_query_obj(p1["queries"][q])
    c3 = canonical_query_obj(p3["queries"][q])

    s1 = json.dumps(c1, indent=2, sort_keys=True, ensure_ascii=False)
    s3 = json.dumps(c3, indent=2, sort_keys=True, ensure_ascii=False)

    is_identical = (s1 == s3)
    print(f"\nQUERY: '{q}' -> Bit-identical between proc1 and proc3: {is_identical}")
    print(f"--- CANONICAL JSON (proc1): ---")
    # print up to 50 lines to keep concise but full sample
    lines = s1.splitlines()
    for l in lines[:45]:
        print(l)
    if len(lines) > 45:
        print(f"    ... [{len(lines)-45} more lines omitted for length] ...")
        print("}")
