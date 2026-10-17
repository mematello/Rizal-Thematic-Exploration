import requests
import hashlib
import json
import sys
from pathlib import Path


API_BASE = "https://mematello-rizal-thematic-search-api.hf.space"

def test_live():
    print(f"Testing live HF Space: {API_BASE}\n")
    
    # 1. /health/db
    print("--- 1. Testing GET /health/db ---")
    r_db = requests.get(f"{API_BASE}/health/db", timeout=60)
    print(f"Status Code: {r_db.status_code}")
    print(f"Headers: Cache-Control = {r_db.headers.get('cache-control')}")
    print(f"Body: {r_db.text}")
    
    if r_db.status_code == 503:
        print("\nCRITICAL ALERT: /health/db returned 503! STOPPING AS INSTRUCTED.")
        sys.exit(1)
        
    assert r_db.status_code == 200, f"Expected 200, got {r_db.status_code}"
    assert r_db.json() == {"status": "ok", "db": "ok"}
    assert r_db.headers.get("cache-control") == "no-store"
    print("PASS: /health/db is 200 OK with Cache-Control: no-store\n")

    # 2. /health
    print("--- 2. Testing GET /health ---")
    r_base = requests.get(f"{API_BASE}/health", timeout=60)
    print(f"Status: {r_base.status_code}, Body: {r_base.text}")
    assert r_base.status_code == 200
    assert r_base.json().get("status") == "ok"
    print("PASS: /health unchanged\n")

    # 3. Noli 1 buod chapter endpoint SHA
    print("--- 3. Testing Noli 1 buod chapter endpoint SHA ---")
    r_chap = requests.get(f"{API_BASE}/api/v1/chapters/noli/1/sanggunian?mode=buod", timeout=60)
    print(f"Status: {r_chap.status_code}")
    canonical_json = json.dumps(r_chap.json(), sort_keys=True)
    chap_sha = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    print(f"Computed SHA-256: {chap_sha}")
    expected_sha = "8f5c4830076befd1225f687fe90d1f47523b0fe7cf4959d84707b285513c25ec"
    print(f"Expected SHA-256: {expected_sha}")
    assert chap_sha == expected_sha, f"SHA mismatch! {chap_sha} != {expected_sha}"
    print("PASS: Noli 1 buod SHA is identical\n")

    # 4. 5 golden queries top-5 match
    print("--- 4. Testing 5 Golden Queries Top-5 Match ---")
    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "docs/perf/golden_snapshot_deterministic.json", "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    
    queries = ["simbahan", "korapsyon", "pag-ibig", "kamatayan", "bayan"]
    for q in queries:
        resp = requests.get(f"{API_BASE}/api/v1/search?q={q}&source_type=buod&novel=both", timeout=60)

        data = resp.json()
        noli_ids = [str(x["id"]) for x in data.get("results", {}).get("noli", [])[:5]]
        fili_ids = [str(x["id"]) for x in data.get("results", {}).get("elfili", [])[:5]]
        
        expected_noli = [str(x["id"]) for x in golden_data["queries"][q]["noli"][:5]]
        expected_fili = [str(x["id"]) for x in golden_data["queries"][q]["elfili"][:5]]


        
        print(f"Query '{q}':")
        print(f"  Noli top-5: {noli_ids} (expected: {expected_noli}) -> Match: {noli_ids == expected_noli}")
        print(f"  Fili top-5: {fili_ids} (expected: {expected_fili}) -> Match: {fili_ids == expected_fili}")
        assert noli_ids == expected_noli, f"Noli mismatch for query {q}"
        assert fili_ids == expected_fili, f"Fili mismatch for query {q}"
    print("PASS: All 5 golden queries top-5 match 100%\n")

if __name__ == "__main__":
    test_live()
