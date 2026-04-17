import os
import sys
import json
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def run_regression_eval():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    test_queries = [
        "education",
        "religion",
        "oppression ng simbahan",
        "revolution against the prayle",
        "death",
        "love",
        "church",
        "night",
        "evening",
        "power"
    ]
    
    engine = RizalEngine()
    db = SessionLocal()
    
    results = []
    
    for q in test_queries:
        original_stdout = sys.stdout
        from io import StringIO
        captured = StringIO()
        sys.stdout = captured
        
        try:
            res = engine.search(db, q, top_k=6, source_type="full")
        except Exception as e:
            res = {"metadata": {}}
            
        sys.stdout = original_stdout
        output_log = captured.getvalue()
        
        native_tokens = "[]"
        foreign_tokens = "[]"
        bridge_tokens = "None"
        enrichment_anchor = "None"
        raw_candidates = "Unknown"
        precision_filtered = "Unknown"
        
        for line in output_log.split('\n'):
            if "[DEBUG] is_cross_lingual:" in line:
                parts = line.split("|")
                if len(parts) >= 6:
                    native_tokens = parts[1].split(":", 1)[1].strip() if "Native:" in parts[1] else "[]"
                    foreign_tokens = parts[2].split(":", 1)[1].strip() if "Foreign:" in parts[2] else "[]"
                    bridge_tokens = parts[3].split(":", 1)[1].strip() if "Bridge Tokens:" in parts[3] else parts[3]
                    enrichment_anchor = parts[5].split(":", 1)[1].strip() if "Enriched:" in parts[5] else "None"
            if "[DEBUG] Raw semantic candidates retrieved:" in line:
                import re
                m = re.search(r"retrieved: (\d+)", line)
                if m: raw_candidates = m.group(1)
        
        meta = res.get('metadata', {})
        result_mode = meta.get('result_mode', 'error')
        reason = meta.get('reason', '')
        suggestions = meta.get('suggestions', [])
        
        total_results = 0
        if isinstance(res, dict) and 'results' in res:
             noli = res['results'].get('noli', [])
             fili = res['results'].get('elfili', [])
             total_results = len(noli) + len(fili)
             
        results.append({
            "Query": q,
            "NativeToken": native_tokens,
            "ForeignToken": foreign_tokens,
            "BridgeToken": bridge_tokens,
            "Enriched": enrichment_anchor,
            "Stage_A": raw_candidates,
            "Mode": result_mode,
            "Reason": reason,
            "FallbackTriggered": "Yes" if result_mode == "semantic_fallback" else "No",
            "Hits": total_results
        })
        
    db.close()
    
    print(f"{'Query':<30} | {'Hits':<5} | {'Enriched':<15} | {'Mode':<15} | {'Reason'}")
    print("-" * 100)
    for r in results:
        print(f"{r['Query']:<30} | {r['Hits']:<5} | {r['Enriched']:<15} | {r['Mode']:<15} | {r['Reason']}")
        
if __name__ == "__main__":
    run_regression_eval()
