import requests
import json
import sys
from pathlib import Path

def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prod_guard import verify_not_production_db
    verify_not_production_db(caller_name="fetch_snapshot.py")

    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_snapshot.py <output_file.json> [port] [--allow-prod]")
        sys.exit(1)

    out_file = Path(sys.argv[1])
    port = "8000"
    for arg in sys.argv[2:]:
        if arg.isdigit():
            port = arg

    golden_path = Path(__file__).parent / "golden_queries.json"
    with open(golden_path, 'r', encoding='utf-8') as f:
        golden = json.load(f)

    queries = []
    for cat in golden['categories'].values():
        queries.extend(cat['queries'])

    if "--reverse" in sys.argv:
        queries.reverse()
        print("[INFO] Running queries in REVERSE order")

    snap = {}
    for i, q in enumerate(queries, 1):
        r = requests.get(f"http://127.0.0.1:{port}/api/v1/search", params={"q": q, "novel": "both"}, timeout=60)
        d = r.json()
        snap[q] = {
            "noli": d.get("results", {}).get("noli", []),
            "elfili": d.get("results", {}).get("elfili", []),
            "metadata": d.get("metadata", {})
        }
        print(f"[{i:02d}/40] {q}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"version": out_file.stem, "queries": snap}, f, indent=2, ensure_ascii=False)
    print(f"Saved snapshot to {out_file}")

if __name__ == "__main__":
    main()
