import urllib.request
import hashlib
import json
import time
import sys

CHAPTERS = [
    # 10 buod from last report
    ("noli", 1, "buod"),
    ("noli", 2, "buod"),
    ("noli", 3, "buod"),
    ("noli", 4, "buod"),
    ("noli", 5, "buod"),
    ("elfili", 1, "buod"),
    ("elfili", 2, "buod"),
    ("elfili", 3, "buod"),
    ("elfili", 4, "buod"),
    ("elfili", 5, "buod"),
    # 3 full from last report
    ("noli", 1, "full"),
    ("noli", 34, "full"),
    ("elfili", 1, "full"),
    # 5 additional random buod chapters
    ("noli", 12, "buod"),
    ("noli", 25, "buod"),
    ("elfili", 15, "buod"),
    ("elfili", 22, "buod"),
    ("elfili", 35, "buod"),
]

def fetch_url(url, retries=2):
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                status = resp.status
            dur = (time.perf_counter() - t0) * 1000
            return status, data, dur
        except Exception as e:
            if attempt == retries:
                raise e
            time.sleep(2)

def main():
    print("=== GATE 1: OLD (PORT 8001) VS NEW (PORT 8000) PARITY VERIFICATION ===")
    results = []
    any_mismatch = False

    for idx, (book, ch, mode) in enumerate(CHAPTERS, 1):
        url_old = f"http://127.0.0.1:8001/api/v1/chapters/{book}/{ch}?mode={mode}"
        url_new = f"http://127.0.0.1:8000/api/v1/chapters/{book}/{ch}?mode={mode}"

        print(f"[{idx:02d}/{len(CHAPTERS):02d}] Testing {book.upper()} Ch {ch:2d} ({mode:4s})...", end=" ", flush=True)

        status_old, bytes_old, dur_old = fetch_url(url_old)
        status_new, bytes_new, dur_new = fetch_url(url_new)

        assert status_old == 200, f"Old returned status {status_old}"
        assert status_new == 200, f"New returned status {status_new}"

        hash_old = hashlib.sha256(bytes_old).hexdigest()
        hash_new = hashlib.sha256(bytes_new).hexdigest()

        match = (hash_old == hash_new)
        if match:
            print(f"PASS (byte-identical) | SHA-256: {hash_new[:16]}... | old: {dur_old:6.1f}ms, new: {dur_new:6.1f}ms")
            results.append({
                "book": book,
                "chapter": ch,
                "mode": mode,
                "match": True,
                "hash_old": hash_old,
                "hash_new": hash_new,
                "dur_old_ms": dur_old,
                "dur_new_ms": dur_new,
            })
        else:
            print(f"MISMATCH!")
            print(f"  Old Hash: {hash_old}")
            print(f"  New Hash: {hash_new}")
            any_mismatch = True

            # Analyze detailed JSON difference
            try:
                json_old = json.loads(bytes_old)
                json_new = json.loads(bytes_new)
                if len(json_old) != len(json_new):
                    print(f"  Length mismatch: old has {len(json_old)} sentences, new has {len(json_new)}")
                for s_idx in range(min(len(json_old), len(json_new))):
                    o = json_old[s_idx]
                    n = json_new[s_idx]
                    if o != n:
                        print(f"  First sentence diff at index {s_idx} (id={o.get('id')} vs {n.get('id')}):")
                        print(f"    Old text: {o.get('sentence_text')}")
                        print(f"    New text: {n.get('sentence_text')}")
                        print(f"    Old themes: {o.get('themes')}")
                        print(f"    New themes: {n.get('themes')}")
                        break
            except Exception as e:
                print(f"  Failed to parse JSON diff: {e}")

            results.append({
                "book": book,
                "chapter": ch,
                "mode": mode,
                "match": False,
                "hash_old": hash_old,
                "hash_new": hash_new,
                "dur_old_ms": dur_old,
                "dur_new_ms": dur_new,
            })

    print("\n--- Summary Table ---")
    print(f"{'Novel':<8} | {'Ch':<3} | {'Mode':<5} | {'Match?':<6} | {'SHA-256 (64 hex)':<64} | {'Old ms':<8} | {'New ms':<8}")
    print("-" * 115)
    for r in results:
        m_str = "YES" if r["match"] else "NO"
        h_str = r["hash_new"] if r["match"] else f"{r['hash_old'][:8]}!={r['hash_new'][:8]}"
        print(f"{r['book']:<8} | {r['chapter']:<3} | {r['mode']:<5} | {m_str:<6} | {h_str:<64} | {r['dur_old_ms']:<8.1f} | {r['dur_new_ms']:<8.1f}")

    if any_mismatch:
        print("\nGATE 1 FAILED: Discrepancy detected between old and new chapter endpoint outputs!")
        sys.exit(1)
    else:
        print("\nGATE 1 PASSED: All 18 chapters are 100% byte-identical between old (main) and new (perf/chapters-endpoint)!")

if __name__ == "__main__":
    main()
