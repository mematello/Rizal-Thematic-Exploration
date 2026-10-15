import json
from pathlib import Path

base_dir = Path(__file__).parent.parent
docs_dir = base_dir / "docs" / "perf"

with open(docs_dir / "final_before_timings.json", "r", encoding="utf-8") as f:
    before = json.load(f)
with open(docs_dir / "final_after_timings.json", "r", encoding="utf-8") as f:
    after = json.load(f)

b_map = {q["query"]: q for q in before["queries"]}
a_map = {q["query"]: q for q in after["queries"]}

key_queries = ["kalayaan", "edukasyon", "pag-ibig", "kamatayan", "bayan"]

print("=" * 95)
print(f"{'Query':<15} | {'Before Warm P50':<18} | {'After Warm P50':<18} | {'Delta':<18} | {'Speedup':<10}")
print("=" * 95)
for q in key_queries:
    b_val = b_map[q]["warm_p50_ms"]
    a_val = a_map[q]["warm_p50_ms"]
    delta = a_val - b_val
    speedup = b_val / a_val if a_val > 0 else 0
    print(f"{q:<15} | {b_val:>15.2f} ms | {a_val:>15.2f} ms | {delta:>15.2f} ms | {speedup:>9.2f}x")
print("=" * 95)

# Also print all 8 single_word_tagalog queries
print("\nAll 8 single_word_tagalog queries:")
print(f"{'Query':<15} | {'Before Warm P50':<18} | {'After Warm P50':<18} | {'Delta':<18} | {'Speedup':<10}")
print("-" * 95)
for q_obj in before["queries"]:
    if q_obj["category"] == "single_word_tagalog":
        q = q_obj["query"]
        b_val = b_map[q]["warm_p50_ms"]
        a_val = a_map[q]["warm_p50_ms"]
        delta = a_val - b_val
        speedup = b_val / a_val if a_val > 0 else 0
        print(f"{q:<15} | {b_val:>15.2f} ms | {a_val:>15.2f} ms | {delta:>15.2f} ms | {speedup:>9.2f}x")
