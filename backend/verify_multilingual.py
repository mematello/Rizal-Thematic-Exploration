import os
import sys
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine
import json

def verify_multilingual():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    queries = [
        "education",
        "justice",
        "oppression",
        "religion",
        "education ni Ibarra",
        "justice for Basilio",
        "oppression ng simbahan",
        "revolution against the prayle"
    ]
    
    print("Initializing Engine...")
    engine = RizalEngine()
    db: Session = SessionLocal()
    
    results_report = []

    try:
        for q in queries:
            print(f"\n{'='*50}")
            print(f"Testing Query: {q}")
            print(f"{'='*50}")
            
            # Capture output
            original_stdout = sys.stdout
            from io import StringIO
            captured = StringIO()
            sys.stdout = captured
            
            error_str = None
            try:
                res = engine.search(db, q, top_k=3, source_type="full")
            except Exception as e:
                res = {"error": str(e)}
                error_str = str(e)
                
            sys.stdout = original_stdout
            output_log = captured.getvalue()
            
            if error_str:
                print(f"Exception triggered: {error_str}")
            
            # Parse debug values
            is_cross_lingual = False
            native_tokens = "[]"
            foreign_tokens = "[]"
            bridge_tokens = "None"
            discarded_words = "[]"
            
            for line in output_log.split('\n'):
                if "[DEBUG] is_cross_lingual:" in line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        is_cross_lingual = "True" in parts[0]
                        native_tokens = parts[1].split(":", 1)[1].strip() if "Native:" in parts[1] else "[]"
                        foreign_tokens = parts[2].split(":", 1)[1].strip() if "Foreign:" in parts[2] else "[]"
                        bridge_tokens = parts[3].split(":", 1)[1].strip() if "Bridge Tokens:" in parts[3] else parts[3]
                        discarded_words = parts[4].split(":", 1)[1].strip() if "Discarded:" in parts[4] else "[]"
            
            noli_results = res.get('results', {}).get('noli', []) if isinstance(res, dict) and 'results' in res else (res.get('noli', []) if isinstance(res, dict) else [])
            fili_results = res.get('results', {}).get('elfili', []) if isinstance(res, dict) and 'results' in res else (res.get('elfili', []) if isinstance(res, dict) else [])
            
            total_results = len(noli_results) + len(fili_results)
            
            report = {
                "Query": q,
                "CrossLingualDetected": is_cross_lingual,
                "NativeTokens": native_tokens,
                "ForeignTokens": foreign_tokens,
                "BridgeTokens": bridge_tokens,
                "DiscardedWords": discarded_words,
                "TotalResults": total_results,
                "TopMatches": []
            }
            
            # Print to terminal
            print(f"Mode Detected Cross-Lingual: {is_cross_lingual}")
            print(f"Discarded Low-Info Words: {discarded_words}")
            print(f"Native Tokens: {native_tokens}")
            print(f"Foreign Tokens: {foreign_tokens}")
            print(f"Bridge Tokens Used: {bridge_tokens}")
            print(f"Total Results: {total_results}")
            
            match_pool = noli_results + fili_results
            match_pool.sort(key=lambda x: x.get('scores', {}).get('final', 0), reverse=True)
            
            for rank, r in enumerate(match_pool[:2]):
                snip = r.get('sentence_text', '')[:100]
                book = 'Noli' if 'noli' in str(r.get('sent_obj', '')).lower() else ('Fili' if 'fili' in str(r.get('sent_obj', '')).lower() else 'Unknown')
                score = float(r.get('scores', {}).get('final', 0)) / 100.0 if 'scores' in r else 0.0
                reason = r.get('reason', '')
                print(f"  [{rank+1}] ({book}) [Score: {score:.3f}] [Reason: {reason}] {snip}...")
                report["TopMatches"].append(f"({book}) {snip}")
                
            results_report.append(report)
            
    finally:
        db.close()
        
    # Write report to file
    with open("multilingual_report.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=4)
        
    print("\nVerification Complete.")
    
if __name__ == "__main__":
    verify_multilingual()
