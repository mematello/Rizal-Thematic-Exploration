import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

print("Testing imports...")
try:
    from rizal.config import BOOKS_CONFIG
    print("✓ config")
    from rizal.utils import extract_words
    print("✓ utils")
    from rizal.errors import RizalError
    print("✓ errors")
    from rizal.ui import RizalUI
    print("✓ ui")
    from rizal.query_analyzer import QueryAnalyzer
    print("✓ query_analyzer")
    from rizal.loader import DataLoader
    print("✓ loader")
    from rizal.engine import RizalEngine
    print("✓ engine")
    print("\nAll modules imported successfully.")
except Exception as e:
    print(f"\nImport failed: {e}")
    sys.exit(1)
