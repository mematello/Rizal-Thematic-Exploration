#!/usr/bin/env python3
"""
Tests fallback paths for chapter static cache:
  1. Fingerprint mismatch -> static cache disabled, fallback to live computation.
  2. DB connectivity failure -> static cache enabled as resilient fallback, warning logged.
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from unittest.mock import MagicMock
from sqlalchemy.exc import OperationalError
from app.api.v1 import content

print("=== TESTING FALLBACK PATHS ===")

print("\n--- Test 1: Fingerprint Mismatch Path ---")
content._fingerprint_checked = False
content._static_cache_enabled = True

mock_db = MagicMock()
mock_db.execute.return_value.fetchone.return_value = (99999, "tampered_md5", 545, "theme_md5")
res = content.verify_corpus_fingerprint(mock_db)
print(f"verify_corpus_fingerprint returned: {res}")
print(f"Static cache enabled status: {content._static_cache_enabled}")
assert res is False, "Expected False on mismatch!"
assert content._static_cache_enabled is False, "Expected cache disabled on mismatch!"
print("[PASS] Fingerprint mismatch disables bundle and triggers live computation.")

print("\n--- Test 2: DB Connectivity Failure Path ---")
content._fingerprint_checked = False
content._static_cache_enabled = False

mock_db_fail = MagicMock()
mock_db_fail.execute.side_effect = OperationalError("Connection refused: database host unreachable", None, None)
res_fail = content.verify_corpus_fingerprint(mock_db_fail)
print(f"verify_corpus_fingerprint returned: {res_fail}")
print(f"Static cache enabled status: {content._static_cache_enabled}")
assert res_fail is True, "Expected True on connectivity failure!"
assert content._static_cache_enabled is True, "Expected cache enabled on connectivity failure!"
print("[PASS] DB connectivity failure safely serves precomputed static bundle.")

print("\n=== ALL FALLBACK TESTS PASSED ===")
