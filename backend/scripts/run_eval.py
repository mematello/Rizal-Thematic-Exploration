import json
import os
import sys
import numpy as np

# Configure path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("DEBUG_SEARCH", None)  # Ensure debug prints are off for clean eval output

from app.models.database import SessionLocal
from app.core.engine import get_engine

def load_queries():
    with open(os.path.join(os.path.dirname(__file__), 'eval_queries.json'), 'r') as f:
        return json.load(f)

def run_evaluation(engine, db, queries, book, mode="full"):
    print(f"\n{'='*50}")
    print(f" EVALUATION RUN: Book={book.upper()} | Mode={mode.upper()}")
    print(f"{'='*50}")

    total_results = 0
    total_filtered = 0

    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Engine searches both books, we just isolate the one we want
        response = engine.search(db, query, top_k=20, source_type=mode)
        results = response.get("results", {}).get(book, [])
        meta = response.get("metadata", {})
        
        result_mode = meta.get("result_mode", "unknown")
        reason = meta.get("reason", "unknown")
        
        count = len(results)
        total_results += count
        
        print(f"  Mode: {result_mode} | Reason: {reason}")
        print(f"  Returned: {count} results")
        
        if count > 0:
            avg_sem = np.mean([r['scores'].get('semantic', 0) for r in results])
            avg_lex = np.mean([r['scores'].get('lexical', 0) for r in results])
            print(f"  Avg Semantic Score: {avg_sem:.2f}")
            print(f"  Avg Lexical Score: {avg_lex:.2f}")
            
            # Print top 3 snippet previews
            print(f"  Top 3 Snippets:")
            for i, r in enumerate(results[:3], 1):
                print(f"    {i}. {r['sentence_text'][:80]}...")
        else:
            if reason in ["validation_failed", "filtered_by_ranker"]:
                total_filtered += 1

def main():
    queries = load_queries()
    db = SessionLocal()
    
    try:
        engine = get_engine()
        print(f"Loaded {len(queries)} evaluation queries.")
        
        run_evaluation(engine, db, queries, "noli", "full")
        run_evaluation(engine, db, queries, "elfili", "full")
        
    finally:
        db.close()

if __name__ == "__main__":
    main()
