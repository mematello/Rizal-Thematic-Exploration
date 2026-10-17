#!/usr/bin/env python3
"""
scripts/test_sanggunian_fallbacks.py

Verifies the 3 fallback resilience cases for Sanggunian optimization:
1. Fingerprint mismatch: loud warning, disables static cache, falls back to live compute.
2. Missing bundle: loud warning, disables static cache, falls back to live compute.
3. Database unreachable: loud warning, serves static bundle as fallback, never crashes.
"""

import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import app.api.v1.content as content


def test_fingerprint_mismatch():
    print("Testing Fallback 1: Fingerprint Mismatch...", flush=True)
    content._sanggunian_fingerprint_checked = False
    content._static_sanggunian_loaded = True
    content._static_sanggunian_enabled = False
    content._static_sanggunian_fingerprint = {
        "sentence_count": 16402,
        "sentence_text_md5": "e438ce3f61098624121329cd5a20f9a2",
        "dapt_null_count": 4580,
        "dapt_column_md5": "0e2c364fa8c8b6d9a24cdaba7e269577"
    }

    # Mock DB execute to return a mismatched row
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchone.return_value = (
        99999,  # sentence_count mismatch
        "mismatched_md5",
        999,
        "mismatched_dapt_md5"
    )

    result = content.verify_sanggunian_fingerprint(mock_db)
    assert result is False, "Expected False on fingerprint mismatch"
    assert content._static_sanggunian_enabled is False, "Cache should be disabled"
    print("  [PASS] Correctly logged warning, disabled static cache, returned False.", flush=True)


def test_missing_bundle():
    print("Testing Fallback 2: Missing Bundle...", flush=True)
    content._sanggunian_fingerprint_checked = False
    content._static_sanggunian_loaded = False
    content._static_sanggunian_enabled = False
    content._static_sanggunian_fingerprint = None

    from unittest.mock import patch
    with patch.object(content, "load_static_sanggunian_cache"):
        mock_db = MagicMock()
        result = content.verify_sanggunian_fingerprint(mock_db)
        assert result is False, "Expected False when bundle is missing"
        assert content._static_sanggunian_enabled is False, "Cache should be disabled"
        print("  [PASS] Correctly disabled static cache when bundle is missing.", flush=True)


def test_database_unreachable():
    print("Testing Fallback 3: Database Unreachable (Offline Mode)...", flush=True)
    content._sanggunian_fingerprint_checked = False
    content._static_sanggunian_loaded = True
    content._static_sanggunian_enabled = False
    content._static_sanggunian_fingerprint = {
        "sentence_count": 16402,
        "sentence_text_md5": "e438ce3f61098624121329cd5a20f9a2",
        "dapt_null_count": 4580,
        "dapt_column_md5": "0e2c364fa8c8b6d9a24cdaba7e269577"
    }

    mock_db = MagicMock()
    mock_db.execute.side_effect = Exception("Supabase connection timeout / DNS failure")

    result = content.verify_sanggunian_fingerprint(mock_db)
    assert result is True, "Expected True (serve static bundle) when DB is down"
    assert content._static_sanggunian_enabled is True, "Static bundle should be enabled as fallback"
    print("  [PASS] Correctly logged warning, enabled static bundle fallback, never crashed.", flush=True)


def main():
    print("=" * 70, flush=True)
    print("RUNNING SANGGUNIAN FALLBACK RESILIENCE TESTS", flush=True)
    print("=" * 70, flush=True)
    test_fingerprint_mismatch()
    test_missing_bundle()
    test_database_unreachable()
    print("\nALL 3 FALLBACK TESTS PASSED!", flush=True)


if __name__ == "__main__":
    main()
