import os
import sys
import json
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def run_diagnostics():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # 12 pure concepts x 4 languages (Tagalog, English, Spanish, Italian)
    pure_concepts = [
        ("education", ["edukasyon", "education", "educación", "educazione"]),
        ("justice", ["katarungan", "justice", "justicia", "giustizia"]),
        ("oppression", ["pang-aapi", "oppression", "opresión", "oppressione"]),
        ("religion", ["relihiyon", "religion", "religión", "religione"]),
        ("revolution", ["rebolusyon", "revolution", "revolución", "rivoluzione"]),
        ("corruption", ["korapsyon", "corruption", "corrupción", "corruzione"]),
        ("death", ["kamatayan", "death", "muerte", "morte"]),
        ("love", ["pag-ibig", "love", "amor", "amore"]),
        ("sacrifice", ["sakripisyo", "sacrifice", "sacrificio", "sacrificio"]),
        ("suffering", ["pagdurusa", "suffering", "sufrimiento", "sofferenza"]),
        ("family", ["pamilya", "family", "familia", "famiglia"]),
        ("honor", ["karangalan", "honor", "honor", "onore"]),
    ]
    
    mixed_concepts = [
        "love ni Ibarra",
        "death of Elias",
        "justice for Basilio",
        "suffering ni Sisa",
        "familia de Ibarra"
    ]
    
    queries = []
    for concept, langs in pure_concepts:
        for q in langs:
            queries.append({"q": q, "category": f"pure_{concept}"})
            
    for q in mixed_concepts:
        queries.append({"q": q, "category": "mixed_entity"})

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
            
            is_cross_lingual = False
            native_tokens = "[]"
            foreign_tokens = "[]"
            bridge_tokens = "None"
            discarded_words = "[]"
            closest_theme_score = "None"
            
            for line in output_log.split('\n'):
                if "[DEBUG] is_cross_lingual:" in line:
                    parts = line.split("|")
                    if len(parts) >= 5:
                        is_cross_lingual = "True" in parts[0]
                        native_tokens = parts[1].split(":", 1)[1].strip() if "Native:" in parts[1] else "[]"
                        foreign_tokens = parts[2].split(":", 1)[1].strip() if "Foreign:" in parts[2] else "[]"
                        bridge_tokens = parts[3].split(":", 1)[1].strip() if "Bridge Tokens:" in parts[3] else parts[3]
                        discarded_words = parts[4].split(":", 1)[1].strip() if "Discarded:" in parts[4] else "[]"
                if "[DEBUG] Theme Anchor Score:" in line:
                    import re
                    m = re.search(r"Theme Anchor Score: ([\d\.]+)", line)
                    if m: closest_theme_score = m.group(1)
            
            meta = res.get('metadata', {})
            result_mode = meta.get('result_mode', 'error')
            
            noli_results = res.get('results', {}).get('noli', []) if isinstance(res, dict) and 'results' in res else []
            fili_results = res.get('results', {}).get('elfili', []) if isinstance(res, dict) and 'results' in res else []
            
            match_pool = noli_results + fili_results
            match_pool.sort(key=lambda x: x.get('scores', {}).get('final', 0), reverse=True)
            
            total_results = len(match_pool)
            top_score = match_pool[0].get('scores', {}).get('final', 0) / 100.0 if total_results > 0 else 0.0
            
            report = {
                "Query": q,
                "Category": item["category"],
                "CrossLingualDetected": is_cross_lingual,
                "NativeTokens": native_tokens,
                "ForeignTokens": foreign_tokens,
                "BridgeTokens": bridge_tokens,
                "DiscardedWords": discarded_words,
                "ThemeScore": closest_theme_score,
                "ResultMode": result_mode,
                "TotalResults": total_results,
                "TopScore": top_score,
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
        
    with open("diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=4)
        
    print("\nDiagnostic Complete.")

if __name__ == "__main__":
    run_diagnostics()
