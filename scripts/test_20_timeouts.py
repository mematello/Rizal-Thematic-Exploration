import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient
from app.main import app
from app.models.database import engine
from app.api.v1 import health

def test_20_forced_timeouts():
    client = TestClient(app)
    
    print("=== Starting 20 Consecutive Forced Timeouts Test ===")
    initial_threads = threading.active_count()
    initial_checkedout = engine.pool.checkedout()
    print(f"Before Test: Active Threads = {initial_threads}, Checked-out DB connections = {initial_checkedout}")
    
    # Simulate a hanging DB query that exceeds the 5.0s timeout
    def hanging_check():
        time.sleep(5.5)
        return True

    results = []
    t_start = time.perf_counter()
    
    with patch("app.api.v1.health._check_db", side_effect=hanging_check):
        for i in range(20):
            t0 = time.perf_counter()
            resp = client.get("/health/db")
            elapsed = (time.perf_counter() - t0) * 1000
            current_threads = threading.active_count()
            current_checkedout = engine.pool.checkedout()
            results.append((resp.status_code, elapsed, current_threads, current_checkedout))
            print(f"Run {i+1:02d}: Status={resp.status_code}, Elapsed={elapsed:.1f}ms, Threads={current_threads}, CheckedOut={current_checkedout}")
            assert resp.status_code == 503
            assert resp.headers.get("cache-control") == "no-store"
            assert resp.json() == {"status": "degraded", "db": "error"}

    total_time = time.perf_counter() - t_start
    print(f"\nCompleted 20 runs in {total_time:.2f}s")
    
    # Allow background test threads to settle
    time.sleep(6)
    
    final_threads = threading.active_count()
    final_checkedout = engine.pool.checkedout()
    print(f"After Test: Active Threads = {final_threads}, Checked-out DB connections = {final_checkedout}")
    print(f"Thread delta: {final_threads - initial_threads}")
    print(f"Connection leak delta: {final_checkedout - initial_checkedout}")
    
    assert final_checkedout == initial_checkedout, "Leaked database connection!"
    print("ALL 20 FORCED TIMEOUTS PASSED CLEANLY!")

if __name__ == "__main__":
    test_20_forced_timeouts()
