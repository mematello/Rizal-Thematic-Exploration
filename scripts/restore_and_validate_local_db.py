#!/usr/bin/env python3
"""
scripts/restore_and_validate_local_db.py

Local Database Restore & Fingerprint Validation Tool for Rizal Thematic Explorer.
Safely manages the local Postgres instance (port 5433) so that developers, benchmarks,
and cache builders can run locally with zero egress against production Supabase.

Usage:
    # 1. Start local postgres container:
    docker compose up -d postgres

    # 2. Restore and validate:
    python scripts/restore_and_validate_local_db.py --restore

    # 3. Validate only:
    python scripts/restore_and_validate_local_db.py
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
BACKUP_SQL = REPO_ROOT / "rizal_db_backup.sql"

DEFAULT_LOCAL_URL = "postgresql://rizal:rizalpass123@localhost:5433/rizal_db"

# Canonical expected production metrics
CANONICAL_SENTENCE_COUNT = 16402
CANONICAL_THEME_COUNT = 545
CANONICAL_DAPT_NULL_COUNT = 4580


def restore_backup(db_url: str):
    """Restores rizal_db_backup.sql into the local database."""
    if not BACKUP_SQL.exists():
        print(f"[ERROR] Backup file not found at: {BACKUP_SQL}")
        sys.exit(1)

    backup_size_mb = BACKUP_SQL.stat().st_size / (1024 * 1024)
    print(f"[RESTORE] Found backup file: {BACKUP_SQL.name} ({backup_size_mb:.2f} MB)")

    # Check if docker is available and rizal_postgres is running
    use_docker = False
    try:
        res = subprocess.run(["docker", "ps", "--filter", "name=rizal_postgres", "--format", "{{.Names}}"],
                             capture_output=True, text=True)
        if "rizal_postgres" in res.stdout:
            use_docker = True
            print("[RESTORE] Detected running 'rizal_postgres' Docker container.")
    except Exception:
        use_docker = False

    t0 = time.time()
    if use_docker:
        print("[RESTORE] Restoring database via docker exec into rizal_postgres...")
        cmd = ["docker", "exec", "-i", "rizal_postgres", "psql", "-U", "rizal", "-d", "rizal_db"]
        with open(BACKUP_SQL, "rb") as f_in:
            proc = subprocess.run(cmd, stdin=f_in, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[ERROR] Docker restore failed (exit {proc.returncode}):\n{proc.stderr}")
            sys.exit(1)
    else:
        # Fallback to local psql command
        print(f"[RESTORE] Restoring database via psql to {db_url}...")
        cmd = ["psql", db_url, "-f", str(BACKUP_SQL)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[ERROR] psql restore failed (exit {proc.returncode}):\n{proc.stderr}")
                sys.exit(1)
        except FileNotFoundError:
            print("[ERROR] Neither Docker nor local 'psql' binary is available.")
            print("Please ensure Docker Desktop is running and execute: docker compose up -d postgres")
            sys.exit(1)

    duration = time.time() - t0
    print(f"[RESTORE] Restore completed successfully in {duration:.1f}s.")


def validate_corpus_fingerprint(db_url: str):
    """Validates that local database matches the canonical corpus fingerprint."""
    from sqlalchemy import create_engine, text

    if db_url.startswith("postgresql://"):
        try:
            import psycopg
        except ImportError:
            try:
                import psycopg2
                db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
            except ImportError:
                pass

    print(f"\n[VALIDATE] Connecting to local database: {db_url}")
    try:
        engine = create_engine(db_url, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            print("[VALIDATE] Querying corpus counts and cryptographic MD5 fingerprints...")
            t0 = time.time()
            sql = text("""
                SELECT 
                    (SELECT count(*) FROM sentences) AS sentence_count,
                    (SELECT md5(string_agg(sentence_text, '' ORDER BY id)) FROM sentences) AS sentence_text_md5,
                    (SELECT count(*) FROM themes) AS theme_count,
                    (SELECT md5(string_agg(tagalog_title || ':' || coalesce(meaning, ''), '' ORDER BY id)) FROM themes) AS theme_content_md5,
                    (SELECT count(*) FROM sentences WHERE embedding_dapt IS NULL) AS dapt_null_count
            """)
            row = conn.execute(sql).fetchone()
            dur = time.time() - t0
    except Exception as e:
        print(f"\n[ERROR] Unable to connect to local database: {e}")
        print("\nTroubleshooting:")
        print("  1. Verify Docker container is running: docker compose up -d postgres")
        print("  2. Verify port 5433 is accessible: docker ps")
        print("  3. To restore database: python scripts/restore_and_validate_local_db.py --restore")
        sys.exit(1)

    sentence_count = int(row[0])
    sentence_text_md5 = str(row[1])
    theme_count = int(row[2])
    theme_content_md5 = str(row[3])
    dapt_null_count = int(row[4])

    print(f"[VALIDATE] Queried in {dur:.2f}s:")
    print(f"  - Sentences:        {sentence_count:,} (Expected: {CANONICAL_SENTENCE_COUNT:,})")
    print(f"  - Sentence MD5:     {sentence_text_md5}")
    print(f"  - Themes:           {theme_count:,} (Expected: {CANONICAL_THEME_COUNT:,})")
    print(f"  - Theme MD5:        {theme_content_md5}")
    print(f"  - DAPT NULL Count:  {dapt_null_count:,} (Expected: {CANONICAL_DAPT_NULL_COUNT:,})")

    # Assertions
    mismatches = []
    if sentence_count != CANONICAL_SENTENCE_COUNT:
        mismatches.append(f"sentence_count mismatch: got {sentence_count}, expected {CANONICAL_SENTENCE_COUNT}")
    if theme_count != CANONICAL_THEME_COUNT:
        mismatches.append(f"theme_count mismatch: got {theme_count}, expected {CANONICAL_THEME_COUNT}")
    if dapt_null_count != CANONICAL_DAPT_NULL_COUNT:
        mismatches.append(f"dapt_null_count mismatch: got {dapt_null_count}, expected {CANONICAL_DAPT_NULL_COUNT}")

    if mismatches:
        print("\n" + "=" * 80)
        print("[FAIL] LOCAL DATABASE CORPUS MISMATCH!")
        for m in mismatches:
            print(f"  - {m}")
        print("\nPlease run with '--restore' to re-import rizal_db_backup.sql:")
        print("  python scripts/restore_and_validate_local_db.py --restore")
        print("=" * 80)
        sys.exit(1)
    else:
        print("\n" + "=" * 80)
        print("[OK] Local database matches production corpus fingerprint (16,402 sentences, 545 themes).")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Restore and validate local PostgreSQL database.")
    parser.add_argument("--db-url", default=DEFAULT_LOCAL_URL, help=f"Local database URL (default: {DEFAULT_LOCAL_URL})")
    parser.add_argument("--restore", action="store_true", help="Restore rizal_db_backup.sql before validating")
    args = parser.parse_args()

    # Guard: ensure we NEVER restore into production
    if "supabase" in args.db_url.lower():
        print("[ERROR] Refusing to restore into production Supabase host!")
        sys.exit(1)

    if args.restore:
        restore_backup(args.db_url)

    validate_corpus_fingerprint(args.db_url)


if __name__ == "__main__":
    main()
