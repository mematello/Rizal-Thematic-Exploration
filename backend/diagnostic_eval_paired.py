import os
import sys
import json
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def run_paired_diagnostics():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    query_groups = [
        [
            "oppression",
            "oppression ng simbahan",
            "oppression ni Padre Damaso",
            "oppression of the friars"
        ],
        [
            "religion",
            "religion ng mga prayle"
        ],
        [
            "love",
            "love ni Maria Clara"
        ],
        [
            "death",
            "death of Elias"
        ],
        [
            "justice",
            "justice for Basilio"
        ],
        [
            "corruption",
            "corruption ng pamahalaan"
        ],
        [
            "suffering",
            "suffering ni Sisa"
        ]
    ]
    
    queries = []
    for g in query_groups:
        concept = g[0]
        for q in g:
            queries.append({"q": q, "concept": concept})

    print("Initializing Engine...")
    engine = RizalEngine()
    db: Session = SessionLocal()
    
    results_report = []

    try:
        for idx, item in enumerate(queries):
            q = item["q"]
            print(f"[{idx+1}/{len(queries)}] Testing Query: {q}")
            
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
            
            # If ResultMode is lexical, the number of raw_candidates isn't printed by fallback. 
            # We can infer it's > 20 since it didn't trigger fallback.
            if result_mode == "lexical" and raw_candidates == "Unknown":
                raw_candidates = ">20 (Lexical Trigger)"
                
            noli_results = res.get('results', {}).get('noli', []) if isinstance(res, dict) and 'results' in res else []
            fili_results = res.get('results', {}).get('elfili', []) if isinstance(res, dict) and 'results' in res else []
            
            match_pool = noli_results + fili_results
            match_pool.sort(key=lambda x: x.get('scores', {}).get('final', 0), reverse=True)
            
            total_results = len(match_pool)
            
            report = {
                "Query": q,
                "Concept": item["concept"],
                "NativeTokens": native_tokens,
                "ForeignTokens": foreign_tokens,
                "BridgeTokens": bridge_tokens,
                "EnrichmentAnchor": enrichment_anchor,
                "Stage_A_Candidates": raw_candidates,
                "ResultMode": result_mode,
                "ThemeScore": closest_theme_score,
                "TotalResults": total_results,
                "TopMatches": []
            }
            
            for rank, r in enumerate(match_pool[:2]):
                snip = r.get('sentence_text', '')[:100]
                book = 'Noli' if 'noli' in str(r.get('sent_obj', '')).lower() else ('Fili' if 'fili' in str(r.get('sent_obj', '')).lower() else 'Unknown')
                score = float(r.get('scores', {}).get('final', 0)) / 100.0 if 'scores' in r else 0.0
                report["TopMatches"].append(f"[{score:.3f}] {snip}")
                
            results_report.append(report)
            
    finally:
        db.close()
        
    with open("paired_diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=4)
        
    print("\nDiagnostic Complete.")

if __name__ == "__main__":
    run_paired_diagnostics()
