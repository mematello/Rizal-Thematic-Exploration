import os
import sys
import json
from sqlalchemy.orm import Session
from app.models.database import SessionLocal
from app.core.engine import RizalEngine

def execute_query(engine, db, q, is_experiment=False):
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
    bridge_tokens = "[]"
    closest_theme_score = "None"
    raw_candidates = "Unknown"
    
    for line in output_log.split('\n'):
        if "[DEBUG] is_cross_lingual:" in line:
            parts = line.split("|")
            if len(parts) >= 5:
                native_tokens = parts[1].split(":", 1)[1].strip() if "Native:" in parts[1] else "[]"
                foreign_tokens = parts[2].split(":", 1)[1].strip() if "Foreign:" in parts[2] else "[]"
                bridge_tokens = parts[3].split(":", 1)[1].strip() if "Bridge Tokens:" in parts[3] else parts[3]
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
    top_score = match_pool[0].get('scores', {}).get('final', 0) if total_results > 0 else 0
    top_prec = match_pool[0].get('scores', {}).get('precision', 0) if total_results > 0 else 0
    
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
            
    # For experimental modes, just return the count and top match snippet
    if is_experiment:
        snippet = match_pool[0].get('sentence_text', '')[:60] if total_results > 0 else ""
        return total_results, snippet

    report = {
        "Query": q,
        "NativeTokens": native_tokens,
        "ForeignTokens": foreign_tokens,
        "BridgeTokens": bridge_tokens,
        "StageA_Cands": raw_candidates,
        "ResultMode": result_mode,
        "ThemeScore": closest_theme_score,
        "TotalResults": total_results,
        "TopScore": f"{top_score} (Prec: {top_prec})",
        "FailStage": fail_stage
    }
    return report

def run_diagnostics():
    os.environ["DEBUG_SEARCH"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # 1. Baseline testing
    failing_multilingual = [
        "night",
        "evening",
        "church",
        "love for the church",
        "joy in the church"
    ]
    
    filipino_controls = [
        "gabi",
        "hapon", # (alternative for evening/afternoon)
        "simbahan",
        "pagmamahal sa simbahan",
        "kasiyahan sa simbahan"
    ]
    
    working_multilingual_controls = [
        "education",
        "religion",
        "oppression ng simbahan",
        "revolution against the prayle"
    ]
    
    print("Initializing Engine...")
    engine = RizalEngine()
    db: Session = SessionLocal()
    
    baseline_results = []
    
    all_baseline = failing_multilingual + filipino_controls + working_multilingual_controls
    
    print("\n--- RUNNING BASELINE ---")
    for q in all_baseline:
        print(f"Testing: {q}")
        r = execute_query(engine, db, q, False)
        baseline_results.append(r)
        
    print("\n--- RUNNING MULTILINGUAL THRESHOLD SENSITIVITY EXPERIMENTS ---")
    
    experiments = [
        ("Base", "0.50", "0.45", "0.60"),
        ("Lower OOV Gate (0.40)", "0.40", "0.45", "0.60"),
        ("Lower Multi Rank (0.35)", "0.50", "0.35", "0.60"),
        ("Lower Multi Precision (0.40)", "0.50", "0.45", "0.40"),
        ("All Lowered (0.40/0.35/0.40)", "0.40", "0.35", "0.40")
    ]
    
    experiment_results = {}
    
    for exp_name, gate_oov, min_multi, p_thresh in experiments:
        os.environ["TEST_GATE_THRESHOLD_OOV"] = gate_oov
        os.environ["TEST_MIN_THRESHOLD_MULTI"] = min_multi
        os.environ["TEST_PRECISION_THRESHOLD"] = p_thresh
        
        experiment_results[exp_name] = []
        
        for q in failing_multilingual:
            cnt, snippet = execute_query(engine, db, q, True)
            experiment_results[exp_name].append({"query": q, "count": cnt, "snippet": snippet})
            
    db.close()
    
    # Save output
    output = {
        "baseline": baseline_results,
        "experiments": experiment_results
    }
    with open("threshold_experiment.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
        
    print("\nExperiments complete. Result saved to threshold_experiment.json.")

if __name__ == "__main__":
    run_diagnostics()
