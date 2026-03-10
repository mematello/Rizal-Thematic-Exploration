import argparse
import os
import sys

# Configure path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DEBUG_SEARCH"] = "1"

from app.models.database import SessionLocal
from app.core.engine import get_engine

def print_result(rank, result_dict, mode):
    print("-" * 60)
    word_count = len(result_dict['sentence_text'].split())
    print(f"Rank {rank} | ID: {result_dict['id']} | Length: {word_count} words")
    print(f"Chapter: {result_dict['chapter_title']}")
    print(f"Text: {result_dict['sentence_text']}")
    print(f"Scores -> Final: {result_dict['scores'].get('final', 0)}, " 
          f"Semantic: {result_dict['scores'].get('semantic', 0)}, "
          f"Lexical: {result_dict['scores'].get('lexical', 0)}")
    
    if result_dict.get('themes'):
        print(f"Themes: {[t['label'] for t in result_dict['themes']]}")

def main():
    parser = argparse.ArgumentParser(description="Debug search retrieval for the Rizal engine.")
    parser.add_argument("--book", type=str, choices=["noli", "elfili"], required=True, help="Book to search in (noli or elfili)")
    parser.add_argument("--mode", type=str, choices=["buod", "full"], required=True, help="Source text mode")
    parser.add_argument("--query", type=str, required=True, help="Query string")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top results to retrieve")
    args = parser.parse_args()

    print(f"\n======================================")
    print(f" DEBUGGING SEARCH: '{args.query}'")
    print(f" Book: {args.book} | Mode: {args.mode} | Top K: {args.top_k}")
    print(f"======================================\n")

    db = SessionLocal()
    try:
        engine = get_engine()
        response = engine.search(db, args.query, top_k=args.top_k, source_type=args.mode)
        
        meta = response.get("metadata", {})
        print(f"\n--- METADATA ---")
        print(f"Result Mode: {meta.get('result_mode')}")
        print(f"Reason: {meta.get('reason')}")
        print(f"----------------\n")
        
        book_key = args.book
        results = response.get("results", {}).get(book_key, [])
        
        if not results:
            print(f"No results returned for {book_key}.")
            return
            
        print(f"Found {len(results)} results in {book_key}:")
        for i, res in enumerate(results, 1):
            print_result(i, res, meta.get('result_mode'))
            
    finally:
        db.close()

if __name__ == "__main__":
    main()
