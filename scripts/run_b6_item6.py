import os
import sys
import json
import hashlib
import subprocess
import numpy as np
from pathlib import Path

out_lines = []
out_lines.append("================================================================================")
out_lines.append("ITEM 6: BRIDGE TOKENS, BRIDGE TEXT & EMBEDDING EQUALITY (PRE-E2 vs HEAD)")
out_lines.append("================================================================================\n")

dict_items = {
    "death": ["kamatayan", "patay"],
    "love": ["pag-ibig", "pagmamahal"],
    "corruption": ["korapsyon", "katiwalian"],
    "justice": ["katarungan", "hustisya"],
    "oppression": ["pang-aapi", "kalupitan"],
    "church": ["simbahan"],
}

out_lines.append("--------------------------------------------------------------------------------")
out_lines.append("1. EN_TL_DICTIONARY ENTRIES AND TOKEN PROPERTIES (> 3 CHARACTERS)")
out_lines.append("--------------------------------------------------------------------------------")
for term, tokens in dict_items.items():
    valid_tokens = [bt for bt in tokens if len(bt) > 3]
    out_lines.append(f"Term: {term:<12} | EN_TL_DICTIONARY: {tokens} | Tokens with len > 3: {valid_tokens}")

# Run subprocesses to evaluate exact bridge_text and embeddings under seeds 1 and 2
script_code = '''
import os, sys, json
import numpy as np
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['DATABASE_URL'] = 'postgresql://rizal:rizalpass123@127.0.0.1:5433/rizal_db'
os.environ['REDIS_URL'] = 'redis://localhost:6379'
sys.path.insert(0, 'backend')

from app.core.engine import get_engine
engine = get_engine()

dict_items = {
    "death": ["kamatayan", "patay"],
    "love": ["pag-ibig", "pagmamahal"],
    "corruption": ["korapsyon", "katiwalian"],
    "justice": ["katarungan", "hustisya"],
    "oppression": ["pang-aapi", "kalupitan"],
    "church": ["simbahan"],
}

res = {}
for t, tokens in dict_items.items():
    bt_set = set(tokens)
    bt_pre = " ".join([bt for bt in bt_set if len(bt) > 3])
    bt_head = " ".join(sorted([bt for bt in bt_set if len(bt) > 3]))

    emb_pre = engine.base_model.encode(bt_pre, show_progress_bar=False)
    emb_head = engine.base_model.encode(bt_head, show_progress_bar=False)

    res[t] = {
        "set_order": list(bt_set),
        "bt_pre": bt_pre,
        "bt_head": bt_head,
        "emb_pre": emb_pre.tolist(),
        "emb_head": emb_head.tolist()
    }

print("JSON_START" + json.dumps(res))
'''

seed_results = {}
for s in ["1", "2"]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = s
    p = subprocess.run([sys.executable, "-c", script_code], env=env, capture_output=True, text=True)
    out = p.stdout
    if "JSON_START" in out:
        seed_results[s] = json.loads(out.split("JSON_START")[1].strip())
    else:
        print(f"Error under seed {s}:", p.stderr)

out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("2. BRIDGE TEXT & np.array_equal COMPARISON (SEED 1 vs SEED 2)")
out_lines.append("--------------------------------------------------------------------------------")

for t in ["death", "love", "corruption", "justice", "oppression", "church"]:
    d1 = seed_results["1"][t]
    d2 = seed_results["2"][t]

    txt_eq_pre = (d1["bt_pre"] == d2["bt_pre"])
    txt_eq_head = (d1["bt_head"] == d2["bt_head"])

    arr_pre_1 = np.array(d1["emb_pre"])
    arr_pre_2 = np.array(d2["emb_pre"])
    arr_head_1 = np.array(d1["emb_head"])
    arr_head_2 = np.array(d2["emb_head"])

    arr_eq_pre = np.array_equal(arr_pre_1, arr_pre_2)
    arr_eq_head = np.array_equal(arr_head_1, arr_head_2)

    cos_sim_pre = float(np.dot(arr_pre_1, arr_pre_2) / (np.linalg.norm(arr_pre_1) * np.linalg.norm(arr_pre_2)))

    out_lines.append(f"Term: '{t}'")
    out_lines.append(f"  Commit 17896f3 pre-E2 (raw set iteration order):")
    out_lines.append(f"    Seed 1 bridge_text: '{d1['bt_pre']}'")
    out_lines.append(f"    Seed 2 bridge_text: '{d2['bt_pre']}'")
    out_lines.append(f"    Text identical:     {txt_eq_pre}")
    out_lines.append(f"    np.array_equal:     {arr_eq_pre} (cosine similarity: {cos_sim_pre:.6f})")
    out_lines.append(f"  HEAD (sorted(bridge_tokens)):")
    out_lines.append(f"    Seed 1 bridge_text: '{d1['bt_head']}'")
    out_lines.append(f"    Seed 2 bridge_text: '{d2['bt_head']}'")
    out_lines.append(f"    Text identical:     {txt_eq_head}")
    out_lines.append(f"    np.array_equal:     {arr_eq_head}")
    out_lines.append("-" * 80)

# Check execution of bridge_text in engine.py
out_lines.append("\n--------------------------------------------------------------------------------")
out_lines.append("3. EXECUTION DISPOSITION IN engine.py & S0 NON-DIVERGENCE EXPLANATION")
out_lines.append("--------------------------------------------------------------------------------")

with open("docs/perf/golden_snapshot_seed0.json", "r", encoding="utf-8") as f:
    snap = json.load(f)["queries"]

for t in ["death", "love", "corruption", "justice", "oppression", "church"]:
    meta = snap[t]["metadata"]
    mode = meta.get("result_mode")
    lex_cands = meta.get("engine_timings", {}).get("lexical_candidates", 0)
    needs_fallback = (lex_cands < 20)

    out_lines.append(f"Query: '{t:<12}' | result_mode: {mode:<18} | lexical_cands: {lex_cands:>2} | needs_fallback: {needs_fallback}")

out_lines.append("\nDetailed Explanation of Why S0 Shows No Divergence Across Seeds:")
out_lines.append("1. Gate Location:")
out_lines.append("   In engine.py, `bridge_text` is generated at line 1070 inside the block `if needs_fallback:`:")
out_lines.append("     needs_fallback = len(candidates) < top_k * 2  (i.e. candidates < 20)")
out_lines.append("2. Queries Where the Line NEVER Executes (death, church):")
out_lines.append("   - 'death' produces 48 lexical candidates from Noli and El Fili.")
out_lines.append("   - 'church' produces 36 lexical candidates.")
out_lines.append("   Because 48 >= 20 and 36 >= 20, `needs_fallback` is False.")
out_lines.append("   Therefore, the block containing `bridge_text` NEVER executes for 'death' or 'church'.")
out_lines.append("3. Queries Where the Line Executes but Does NOT Reach Output (love, corruption, justice, oppression):")
out_lines.append("   - 'love' (4 candidates), 'corruption' (0 candidates), 'justice' (7 candidates), 'oppression' (1 candidate)")
out_lines.append("     have `needs_fallback = True`.")
out_lines.append("   - In lines 1070-1082, `bridge_embedding` is computed and blended into `q_vec = (q_vec + bridge_embedding) / 2.0`.")
out_lines.append("   - `q_vec` is a LOCAL variable used EXCLUSIVELY to compute `max_theme_score = float(np.max(theme_scores))`")
out_lines.append("     for the in-domain similarity gate.")
out_lines.append("   - Because these queries have valid bridge tokens, `is_safely_mapped` is True, so they never fail the gate.")
out_lines.append("   - Crucially, `q_vec` is DISCARDED immediately after line 1095:")
out_lines.append("     - Candidate semantic scoring uses `query_embedding` (the unblended base embedding of query text).")
out_lines.append("     - `query_vector_sha256` is computed on `query_embedding` bytes, NOT `q_vec`.")
out_lines.append("     - `bridge_tokens` metadata output in Commit D was already sorted: `sorted(list(bridge_tokens))`.")
out_lines.append("   Therefore, the word order inversion in `bridge_text` ('pagmamahal pag-ibig' vs 'pag-ibig pagmamahal')")
out_lines.append("   has 0 effect on candidate scoring, rank order, query vector hashes, or metadata.")
out_lines.append("   Commit E2 (`sorted(bridge_tokens)` in `bridge_text`) is a defensive guarantee of cross-process consistency,")
out_lines.append("   even though the local gate variable never reached search rankings.")

with open("docs/perf/evidence/b6_6.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("Saved docs/perf/evidence/b6_6.txt successfully.")
