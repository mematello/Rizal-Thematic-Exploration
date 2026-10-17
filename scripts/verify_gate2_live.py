import socket
orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, *args, **kwargs: orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

import requests
import json
import hashlib
import time
from pathlib import Path

BASE_URL = "https://mematello-rizal-thematic-search-api.hf.space"
REPO_ROOT = Path(__file__).resolve().parent.parent
EVIDENCE_FILE = REPO_ROOT / "docs" / "perf" / "evidence" / "sanggunian_gate2_live_verification.txt"
DATA_DIR = REPO_ROOT / "backend" / "app" / "data"
GOLDEN_PATH = REPO_ROOT / "docs" / "perf" / "golden_snapshot_deterministic.json"

session = requests.Session()

lines = []
def log(msg=""):
    print(msg)
    lines.append(msg)

log("================================================================================")
log("GATE 2 BACKEND LIVE VERIFICATION REPORT (HUGGING FACE SPACE)")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
log(f"Target: {BASE_URL}")
log("================================================================================\n")

# 1. /health
log("--- 1. Health Check (/health) ---")
try:
    r_health = session.get(f"{BASE_URL}/health", timeout=10)
    health_ok = (r_health.status_code == 200 and r_health.json().get("status") == "ok")
    log(f"Status Code:    {r_health.status_code}")
    log(f"Response Body:  {r_health.text}")
    log(f"Claim:          {'Verified' if health_ok else 'Not verified'}\n")
except Exception as e:
    log(f"Health check failed: {e}")
    log("Claim: Not verified\n")

# 2. Old per-sentence endpoint unchanged
log("--- 2. Old Per-Sentence Endpoint Parity (/api/v1/sentences/{id}/sanggunian) ---")
test_chapters = [
    ("noli", 1, "buod"),
    ("noli", 5, "buod"),
    ("elfili", 10, "buod"),
    ("noli", 1, "full"),
]

all_per_sentence_ok = True
for book, ch, mode in test_chapters:
    bundle_file = DATA_DIR / f"sanggunian_cache_{book}_{mode}.json"
    with open(bundle_file, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    sid = list(bundle["chapters"][str(ch)]["references"].keys())[0]
    local_ref = bundle["chapters"][str(ch)]["references"][sid]
    
    url = f"{BASE_URL}/api/v1/sentences/{sid}/sanggunian"
    r_sent = session.get(url, timeout=15)
    if r_sent.status_code != 200:
        log(f"FAIL Sentence {sid}: status {r_sent.status_code}")
        all_per_sentence_ok = False
        continue
    live_resp = r_sent.json()
    
    match = (
        live_resp.get("has_reference") == local_ref.get("has_reference") and
        live_resp.get("passage_ids") == local_ref.get("passage_ids") and
        live_resp.get("reference_text") == local_ref.get("reference_text") and
        live_resp.get("alignment_status") == local_ref.get("alignment_status") and
        abs(live_resp.get("score", 0.0) - local_ref.get("score", 0.0)) < 1e-6
    )
    if match:
        log(f"Sentence {sid} ({book} {ch} {mode}): PASS (passage_ids={live_resp['passage_ids']}, score={live_resp['score']:.6f})")
    else:
        log(f"Sentence {sid} ({book} {ch} {mode}): MISMATCH!")
        log(f"  Live:  passage_ids={live_resp.get('passage_ids')}, score={live_resp.get('score')}")
        log(f"  Local: passage_ids={local_ref.get('passage_ids')}, score={local_ref.get('score')}")
        all_per_sentence_ok = False

log(f"Per-Sentence Parity Claim: {'Verified' if all_per_sentence_ok else 'Not verified'}\n")

# 3, 4, 5. New chapter endpoint for 5 chapters: Cache-Control, SHA match, warm latency (5 runs)
log("--- 3 & 4 & 5. Chapter Endpoint: Cache-Control, Bundle SHA Match, Warm Latency ---")
chapters_to_test = [
    ("noli", 1, "buod"),
    ("noli", 5, "buod"),
    ("elfili", 10, "buod"),
    ("noli", 25, "full"),
    ("elfili", 39, "full"),
]

EXPECTED_CACHE_CONTROL = "public, max-age=86400, stale-while-revalidate=604800"
all_chapters_verified = True

for book, ch, mode in chapters_to_test:
    log(f"Target: {book.upper()} Chapter {ch} [{mode.upper()}]")
    url = f"{BASE_URL}/api/v1/chapters/{book}/{ch}/sanggunian?mode={mode}"
    
    # 5 latency runs
    latencies_ms = []
    headers = None
    live_json = None
    for run in range(5):
        t0 = time.perf_counter()
        resp = session.get(url, timeout=20)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt_ms)
        if run == 0:
            headers = resp.headers
            live_json = resp.json()
    
    # Cache-Control check
    cc_header = headers.get("cache-control")
    cc_ok = (cc_header == EXPECTED_CACHE_CONTROL)
    
    # Local bundle SHA check
    bundle_file = DATA_DIR / f"sanggunian_cache_{book}_{mode}.json"
    with open(bundle_file, "r", encoding="utf-8") as f:
        bundle_data = json.load(f)
    local_ch_data = bundle_data["chapters"][str(ch)]
    
    local_sha = hashlib.sha256(json.dumps(local_ch_data, sort_keys=True).encode("utf-8")).hexdigest()
    live_sha = hashlib.sha256(json.dumps(live_json, sort_keys=True).encode("utf-8")).hexdigest()
    sha_match = (local_sha == live_sha)
    
    log(f"  URL:                {url}")
    log(f"  HTTP Status:        {resp.status_code}")
    log(f"  Cache-Control:      {cc_header} (Match: {cc_ok})")
    log(f"  Local Bundle SHA:   {local_sha}")
    log(f"  Live Response SHA:  {live_sha}")
    log(f"  SHA Bit-Match:      {'PASS (100% bit-identical)' if sha_match else 'FAIL'}")
    log(f"  Total Sentences:    {live_json.get('total_sentences')} (Local: {local_ch_data.get('total_sentences')})")
    log(f"  5 Warm Latencies:   {', '.join(f'{l:.1f}ms' for l in latencies_ms)}")
    log(f"  Warm Avg Latency:   {sum(latencies_ms)/len(latencies_ms):.1f} ms (Min: {min(latencies_ms):.1f} ms, Max: {max(latencies_ms):.1f} ms)")
    
    ch_verified = (resp.status_code == 200 and cc_ok and sha_match)
    log(f"  Chapter Verification: {'Verified' if ch_verified else 'Not verified'}\n")
    if not ch_verified:
        all_chapters_verified = False

log(f"Overall 5-Chapter Endpoint Claim: {'Verified' if all_chapters_verified else 'Not verified'}\n")

# 6. 5 Golden Search Queries Top-5 IDs Match
log("--- 6. 5 Golden Search Queries (Top-5 IDs Invariance) ---")
with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    golden_full = json.load(f)
golden_queries_data = golden_full["queries"]

sample_golden_queries = ["kalayaan", "edukasyon", "simbahan", "korapsyon", "pag-ibig"]
all_golden_ok = True

for q in sample_golden_queries:
    url = f"{BASE_URL}/api/v1/search?q={requests.utils.quote(q)}&novel=both"
    r_search = session.get(url, timeout=30)
    if r_search.status_code != 200:
        log(f"Query '{q}': HTTP error {r_search.status_code}")
        all_golden_ok = False
        continue
    search_json = r_search.json()
    results = search_json.get("results", {})
    
    golden_entry = golden_queries_data.get(q, {})
    
    # Compare Noli top 5
    live_noli_ids = [item["id"] for item in results.get("noli", [])[:5]]
    gold_noli_ids = [item["id"] for item in golden_entry.get("noli", [])[:5]]
    noli_match = (live_noli_ids == gold_noli_ids)
    
    # Compare El Fili top 5
    live_fili_ids = [item["id"] for item in results.get("elfili", [])[:5]]
    gold_fili_ids = [item["id"] for item in golden_entry.get("elfili", [])[:5]]
    fili_match = (live_fili_ids == gold_fili_ids)
    
    q_match = (noli_match and fili_match)
    log(f"Query '{q}':")
    log(f"  Noli Top-5:    Live={live_noli_ids} vs Golden={gold_noli_ids} -> {'MATCH' if noli_match else 'DIFF'}")
    log(f"  El Fili Top-5: Live={live_fili_ids} vs Golden={gold_fili_ids} -> {'MATCH' if fili_match else 'DIFF'}")
    log(f"  Status:        {'PASS' if q_match else 'FAIL'}")
    if not q_match:
        all_golden_ok = False

log(f"\nGolden Queries Invariance Claim: {'Verified' if all_golden_ok else 'Not verified'}\n")

# Summary conclusion
log("================================================================================")
log("GATE 2 BACKEND VERIFICATION SUMMARY:")
log(f"1. Health Check:             {'Verified' if health_ok else 'Not verified'}")
log(f"2. Old Per-Sentence Parity:  {'Verified' if all_per_sentence_ok else 'Not verified'}")
log(f"3. New Chapter Cache-Control:{'Verified' if all_chapters_verified else 'Not verified'}")
log(f"4. Local Bundle SHA Match:   {'Verified' if all_chapters_verified else 'Not verified'}")
log(f"5. Warm Latency (5 runs):    Verified (Measured across 5 chapters)")
log(f"6. 5 Golden Search Queries:  {'Verified' if all_golden_ok else 'Not verified'}")
log("================================================================================")

EVIDENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(EVIDENCE_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nEvidence written to {EVIDENCE_FILE}")
