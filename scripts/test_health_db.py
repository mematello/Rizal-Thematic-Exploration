import time
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.models.database import engine, SessionLocal
import hashlib
from app.api.v1.content import (
    load_static_sanggunian_cache,
    verify_sanggunian_fingerprint,
    _static_sanggunian_fingerprint,
)

def _get_code_hash_lf(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def test_health_db_endpoints():
    client = TestClient(app)
    
    print("=== Test 1: Normal 200 OK Path ===")
    initial_checkedout = engine.pool.checkedout()
    print(f"Initial checked out connections: {initial_checkedout}")
    
    t0 = time.perf_counter()
    resp = client.get("/health/db")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    print(f"Status Code: {resp.status_code}")
    print(f"Response Body: {resp.json()}")
    print(f"Cache-Control Header: {resp.headers.get('cache-control')}")
    print(f"Latency: {elapsed_ms:.2f} ms")
    
    after_checkedout = engine.pool.checkedout()
    print(f"Checked out connections after request: {after_checkedout}")
    
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert resp.json() == {"status": "ok", "db": "ok"}
    assert resp.headers.get("cache-control") == "no-store"
    assert after_checkedout == initial_checkedout, f"Connection leak! {after_checkedout} != {initial_checkedout}"
    print("Test 1 Passed!")

    print("\n=== Test 2: Simulated DB Failure Path ===")
    def broken_check():
        raise Exception("Secret internal database password or stack trace that should never leak!")

    t0 = time.perf_counter()
    with patch("app.api.v1.health._check_db", side_effect=broken_check):
        resp_fail = client.get("/health/db")
    elapsed_fail_ms = (time.perf_counter() - t0) * 1000

    
    print(f"Failure Status Code: {resp_fail.status_code}")
    print(f"Failure Response Body: {resp_fail.json()}")
    print(f"Failure Cache-Control Header: {resp_fail.headers.get('cache-control')}")
    print(f"Failure Latency: {elapsed_fail_ms:.2f} ms")
    
    fail_checkedout = engine.pool.checkedout()
    print(f"Checked out connections after failure: {fail_checkedout}")
    
    assert resp_fail.status_code == 503, f"Expected 503, got {resp_fail.status_code}"
    assert resp_fail.json() == {"status": "degraded", "db": "error"}
    assert resp_fail.headers.get("cache-control") == "no-store"
    assert "Secret" not in resp_fail.text
    assert elapsed_fail_ms < 5000, f"Failure took too long: {elapsed_fail_ms} ms"
    assert fail_checkedout == initial_checkedout, "Connection leak after failure!"
    print("Test 2 Passed!")

    print("\n=== Test 3: Simulated Timeout Failure Path ===")
    def slow_check():
        time.sleep(5.5) # Exceeds 5.0s timeout
        return True

    t0 = time.perf_counter()
    with patch("app.api.v1.health._check_db", side_effect=slow_check):
        resp_timeout = client.get("/health/db")
    elapsed_timeout_ms = (time.perf_counter() - t0) * 1000

    
    print(f"Timeout Status Code: {resp_timeout.status_code}")
    print(f"Timeout Response Body: {resp_timeout.json()}")
    print(f"Timeout Latency: {elapsed_timeout_ms:.2f} ms")
    
    assert resp_timeout.status_code == 503
    assert resp_timeout.json() == {"status": "degraded", "db": "error"}
    assert 4900 <= elapsed_timeout_ms <= 6500, f"Timeout took unexpected duration: {elapsed_timeout_ms} ms"
    print("Test 3 Passed!")

    print("\n=== Test 4: Existing /health Endpoint Invariance ===")
    resp_base = client.get("/health")
    print(f"Base /health: {resp_base.status_code}, {resp_base.json()}")
    assert resp_base.status_code == 200
    assert resp_base.json().get("status") == "ok"
    print("Test 4 Passed!")

    print("\n=== Test 5: Recomputed Hashes and Bundle Fingerprint Verification ===")
    load_static_sanggunian_cache()
    fp = _static_sanggunian_fingerprint
    
    backend_dir = Path(__file__).resolve().parent.parent / "backend" / "app"
    content_path = backend_dir / "api" / "v1" / "content.py"
    aligner_path = backend_dir / "core" / "robust_aligner.py"
    
    current_content_hash = _get_code_hash_lf(content_path)
    current_aligner_hash = _get_code_hash_lf(aligner_path)
    
    print(f"Current content.py LF hash:        {current_content_hash}")
    print(f"Bundle content_alignment_hash:     {fp.get('content_alignment_hash')}")
    print(f"Current robust_aligner.py LF hash: {current_aligner_hash}")
    print(f"Bundle robust_aligner_hash:        {fp.get('robust_aligner_hash')}")
    
    assert current_content_hash == fp.get("content_alignment_hash"), "content.py hash mismatch!"
    assert current_aligner_hash == fp.get("robust_aligner_hash"), "robust_aligner.py hash mismatch!"
    
    # Run full verification against Supabase
    db = SessionLocal()
    try:
        verified = verify_sanggunian_fingerprint(db)
        print(f"Live Supabase fingerprint verification: {verified}")
        assert verified is True, "verify_sanggunian_fingerprint failed!"
    finally:
        db.close()
    
    print("Test 5 Passed! Startup line and bundle hashes verified 100%!")

if __name__ == "__main__":
    test_health_db_endpoints()
