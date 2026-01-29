#!/usr/bin/env python3
"""
Entry point for Rizal Exploration Engine.
"""
import sys
import os

# Ensure src is in path so we can import 'rizal' even if not installed via pip yet (for dev convenience)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rizal.main import run_cli

if __name__ == "__main__":
    run_cli()
