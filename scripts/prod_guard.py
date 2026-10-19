#!/usr/bin/env python3
"""
scripts/prod_guard.py
Safeguard module to prevent accidental egress consumption against production Supabase.
Refuses execution if DATABASE_URL points to a Supabase host unless --allow-prod is passed.
"""

import os
import sys
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

def get_target_db_url() -> str:
    """Resolves DATABASE_URL from environment or backend/.env."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        backend_env = Path(__file__).resolve().parent.parent / "backend" / ".env"
        if backend_env.exists():
            load_dotenv(backend_env)
            db_url = os.getenv("DATABASE_URL")
    return db_url or ""

def verify_not_production_db(db_url: str = None, allow_prod: bool = False, caller_name: str = ""):
    """
    Checks if db_url points to a Supabase host.
    If yes and allow_prod is False: prints error message and exits with code 1.
    If safe or allowed: logs target host and returns target host.
    """
    if "--allow-prod" in sys.argv:
        allow_prod = True
    if os.getenv("ALLOW_PROD") == "1":
        allow_prod = True

    resolved_url = db_url or get_target_db_url()
    if not resolved_url:
        print("[PROD GUARD] Warning: DATABASE_URL is not set.")
        return "unknown"

    # Clean quotes and prefixes if present
    clean_url = resolved_url.strip()
    if clean_url.startswith("DATABASE_URL="):
        clean_url = clean_url[len("DATABASE_URL="):].strip()
    if (clean_url.startswith('"') and clean_url.endswith('"')) or (clean_url.startswith("'") and clean_url.endswith("'")):
        clean_url = clean_url[1:-1].strip()

    try:
        parsed = urlparse(clean_url)
        host = parsed.hostname or "unknown"
        port = parsed.port or "default"
        database = parsed.path.lstrip("/") or "unknown"
    except Exception:
        host = "unknown"
        port = "unknown"
        database = "unknown"

    script_label = f" for '{caller_name}'" if caller_name else ""

    if "supabase" in host.lower():
        if not allow_prod:
            sys.stderr.write(
                "\n" + "=" * 80 + "\n"
                f"[PROD GUARD ERROR] REFUSING TO RUN{script_label} AGAINST PRODUCTION SUPABASE!\n"
                f"Target Host: {host} (Database: {database})\n\n"
                "Running dev, verification, cache building, or benchmark scripts against Supabase\n"
                "consumes free-tier egress quota (~17.6 KB per row transfer).\n\n"
                "Safe Local Workflow:\n"
                "  1. Start local Postgres container:\n"
                "     docker compose up -d postgres\n"
                "  2. Restore and validate local DB:\n"
                "     python scripts/restore_and_validate_local_db.py --restore\n"
                "  3. Set local DATABASE_URL:\n"
                "     Windows PowerShell: $env:DATABASE_URL=\"postgresql://rizal:rizalpass123@localhost:5433/rizal_db\"\n"
                "     Linux / macOS:      export DATABASE_URL=\"postgresql://rizal:rizalpass123@localhost:5433/rizal_db\"\n\n"
                "To override and explicitly allow running against production Supabase:\n"
                "  Pass the '--allow-prod' command-line flag or set ALLOW_PROD=1.\n"
                + "=" * 80 + "\n\n"
            )
            sys.stderr.flush()
            sys.exit(1)
        else:
            print(f"[PROD GUARD] [WARNING] Explicit override: Allowing execution against PRODUCTION Supabase ({host}).")
    else:
        print(f"[TARGET DB] Host: {host} | Port: {port} | Database: {database}")

    return host

def verify_not_production_api(api_url: str = "", allow_prod: bool = False, caller_name: str = ""):
    """
    Checks if api_url points to Hugging Face production Space or Vercel production UI.
    If yes and allow_prod is False: prints error message and exits with code 1.
    """
    if "--allow-prod" in sys.argv or os.getenv("ALLOW_PROD") == "1":
        allow_prod = True

    if not api_url:
        return "unknown"

    script_label = f" for '{caller_name}'" if caller_name else ""
    is_prod_hf = "hf.space" in api_url.lower()
    is_prod_vercel = "vercel.app" in api_url.lower()

    if is_prod_hf or is_prod_vercel:
        target_type = "HuggingFace Production API" if is_prod_hf else "Vercel Production UI"
        if not allow_prod:
            sys.stderr.write(
                "\n" + "=" * 80 + "\n"
                f"[PROD GUARD ERROR] REFUSING TO RUN{script_label} AGAINST {target_type.upper()}!\n"
                f"Target URL: {api_url}\n\n"
                "To run against local server:\n"
                "  API: http://127.0.0.1:8000\n"
                "  UI:  http://localhost:3000\n\n"
                "To override and explicitly allow running against production:\n"
                "  Pass the '--allow-prod' command-line flag or set ALLOW_PROD=1.\n"
                + "=" * 80 + "\n\n"
            )
            sys.stderr.flush()
            sys.exit(1)
        else:
            print(f"[PROD GUARD] [WARNING] Explicit override: Allowing execution against {target_type} ({api_url}).")
    else:
        print(f"[TARGET API] URL: {api_url}")

    return api_url

if __name__ == "__main__":
    verify_not_production_db()
