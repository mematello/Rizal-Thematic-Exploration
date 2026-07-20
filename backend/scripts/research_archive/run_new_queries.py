import os
import sys
import json
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def execute_query(engine, db, q):
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
    
    closest_theme_score = "None"
    raw_candidates = "Unknown"
    
    for line in output_log.split('\n'):
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
    
    if result_mode == "lexical" and raw_candidates == "Unknown":
        raw_candidates = ">20"
        
    noli_results = res.get('results', {}).get('noli', []) if isinstance(res, dict) and 'results' in res else []
    fili_results = res.get('results', {}).get('elfili', []) if isinstance(res, dict) and 'results' in res else []
    
    match_pool = noli_results + fili_results
    match_pool.sort(key=lambda x: x.get('scores', {}).get('final', 0), reverse=True)
    
    total_results = len(match_pool)
    
    fail_stage = "N/A (Succeeded)"
    if total_results == 0:
        if result_mode == "none" and closest_theme_score != "None" and float(closest_theme_score) < 0.50:
            fail_stage = "Semantic Fallback Gate"
        elif result_mode == "semantic_fallback":
            fail_stage = "Final Ranking/Precision Boundary"
        elif result_mode == "lexical":
            fail_stage = "Lexical (0 strict hits)"
        else:
            fail_stage = "Unknown"

    return {
        "Query": q,
        "TotalResults": total_results,
        "PassFail": "Pass" if total_results > 0 else "Fail",
        "FailStage": fail_stage if total_results == 0 else "N/A",
        "MaxThemeScore": closest_theme_score
    }

def run_custom_queries():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    test_queries = {
        "Level 1 - Single word, clearly in-domain Filipino": [
            "prayle",
            "kura"
        ],
        "Level 2 - Two to three word Filipino concept": [
            "pag-ibig sa bayan",
            "utang na loob"
        ],
        "Level 3 - Multi-word cross-lingual mixed query": [
            "corruption ng mga prayle",
            "freedom para sa Pilipinas"
        ],
        "Level 4 - Complex full Filipino sentence, multi-concept": [
            "paano ipinaglaban ni ibarra ang edukasyon ng mga kabataan",
            "ang lihim na yaman ni simoun at ang kanyang paghihiganti"
        ],
        "Level 5 - Should be correctly rejected": [
            "cellphone at internet connection",
            "spaceship navigation system",
            "artificial intelligence algorithms",
            "bitcoin blockchain technology"
        ]
    }
    
    experiments = [
        ("Base", "0.50", "0.45", "0.60"),
        ("Lower OOV Gate", "0.40", "0.45", "0.60"),
        ("Lower Sentence Rank", "0.50", "0.35", "0.60"),
        ("Lower Precision", "0.50", "0.45", "0.40"),
        ("All Lowered", "0.40", "0.35", "0.40")
    ]
    
    print("Initializing Engine...")
    engine = RizalEngine()
    db = SessionLocal()
    
    final_results = {}
    
    for level, queries in test_queries.items():
        final_results[level] = {}
        for q in queries:
            final_results[level][q] = {}
            for exp_name, gate_oov, min_multi, p_thresh in experiments:
                os.environ["TEST_GATE_THRESHOLD_OOV"] = gate_oov
                os.environ["TEST_MIN_THRESHOLD_MULTI"] = min_multi
                os.environ["TEST_PRECISION_THRESHOLD"] = p_thresh
                
                print(f"Running '{q}' with config '{exp_name}'...")
                res = execute_query(engine, db, q)
                final_results[level][q][exp_name] = res
                
    db.close()
    
    with open("custom_queries_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)
        
    print("\nDone. Saved to custom_queries_results.json")

if __name__ == "__main__":
    run_custom_queries()
