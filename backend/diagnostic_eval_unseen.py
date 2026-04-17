import os
import sys
import json
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def run_unseen_diagnostics():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Concepts NOT in the engine dictionary
    unseen_concepts = [
        "power",
        "freedom",
        "fear",
        "poverty",
        "truth",
        "wealth",
        "disease"
    ]
    
    mixed_concepts = [
        "power of the church",
        "fear ng mga indio",
        "lack of freedom"
    ]
    
    queries = unseen_concepts + mixed_concepts

    print("Initializing Engine...")
    engine = RizalEngine()
    db: Session = SessionLocal()
    
    results_report = []

    try:
        for idx, q in enumerate(queries):
            print(f"[{idx+1}/{len(queries)}] Testing Unseen Concept: {q}")
            
            original_stdout = sys.stdout
            from io import StringIO
            captured = StringIO()
            sys.stdout = captured
            
            error_str = None
            try:
                res = engine.search(db, q, top_k=3, source_type="full")
            except Exception as e:
                res = {"error": str(e), "metadata": {}}
                error_str = str(e)
                
            sys.stdout = original_stdout
            output_log = captured.getvalue()
            
            native_tokens = "[]"
            foreign_tokens = "[]"
            bridge_tokens = "None"
            enrichment_anchor = "None"
            closest_theme_score = "None"
            raw_candidates = "Unknown"
            
            for line in output_log.split('\n'):
                if "[DEBUG] is_cross_lingual:" in line:
                    parts = line.split("|")
                    if len(parts) >= 6:
                        native_tokens = parts[1].split(":", 1)[1].strip() if "Native:" in parts[1] else "[]"
                        foreign_tokens = parts[2].split(":", 1)[1].strip() if "Foreign:" in parts[2] else "[]"
                        bridge_tokens = parts[3].split(":", 1)[1].strip() if "Bridge Tokens:" in parts[3] else parts[3]
                        enrichment_anchor = parts[5].split(":", 1)[1].strip() if "Enriched:" in parts[5] else "None"
                if "[DEBUG] Theme Anchor Score:" in line:
                    import re
                    m = re.search(r"Theme Anchor Score: ([\d\.]+)", line)
                    if m: closest_theme_score = m.group(1)
                if "[DEBUG] Raw semantic candidates retrieved:" in line:
                    import re
                    m = re.search(r"retrieved: (\d+)", line)
                    if m: raw_candidates = m.group(1)
            
            meta = res.get('metadata', {})
            result_mode = meta.get('result_mode', 'error')
            suggestions = meta.get('suggestions', [])
            
            if result_mode == "lexical" and raw_candidates == "Unknown":
                raw_candidates = ">20 (Lexical Trigger)"
                
            noli_results = res.get('results', {}).get('noli', []) if isinstance(res, dict) and 'results' in res else []
            fili_results = res.get('results', {}).get('elfili', []) if isinstance(res, dict) and 'results' in res else []
            
            match_pool = noli_results + fili_results
            match_pool.sort(key=lambda x: x.get('scores', {}).get('final', 0), reverse=True)
            
            total_results = len(match_pool)
            
            report = {
                "Query": q,
                "BridgeTokens": bridge_tokens,
                "EnrichmentAnchor": enrichment_anchor,
                "Stage_A_Candidates": raw_candidates,
                "ResultMode": result_mode,
                "ThemeScore": closest_theme_score,
                "TotalResults": total_results,
                "Suggestions": suggestions
            }
            results_report.append(report)
            
    finally:
        db.close()
        
    print(f"{'Query':<25} | {'Found':<5} | {'Mode':<28} | {'Suggestions'}")
    print("-" * 110)
    for d in results_report:
        print(f"{d['Query']:<25} | {d['TotalResults']:<5} | {d['ResultMode']:<28} | {d['Suggestions']}")
        
if __name__ == "__main__":
    run_unseen_diagnostics()
