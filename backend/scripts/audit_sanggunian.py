"""
Audit script: Run real alignment cases and output runtime scores
for academic documentation Section 4.10.2
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from app.core.robust_aligner import RobustAligner

# Load character alias map
aliases_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'character_aliases.json')
with open(aliases_path, 'r', encoding='utf-8') as f:
    aliases_raw = json.load(f)

tauhan_map = {}
for entry in aliases_raw:
    canonical = entry.get('canonical') or entry.get('name', '')
    for alias in entry.get('aliases', [canonical]):
        tauhan_map[alias.lower()] = canonical

aligner = RobustAligner(tauhan_map=tauhan_map)

# ─── CASE 1: Character-assisted (Tauhan contributes significantly) ───────────
# Ibarra is present in buod and passage
c1_buod = [
    "Pinaghandaan ni Ibarra ang kaniyang pagtatanghal sa harap ng mga panauhin.",
]
c1_full = [
    "Ang lahat ay abala sa pagdating ng okasyon.",
    "Si Ibarra ay nagbihis nang maingat at inihanda ang kaniyang sarili para sa mahalagang sandali.",
    "Maraming panauhin ang dumating upang manood.",
    "Naghanda siya ng matagal para sa pagkakataong ito.",
    "Ang buong bayan ay interesado sa magiging resulta.",
    "Kaya naman nag-ingat si Ibarra sa bawat detalye.",
    "Nais niyang mapanatili ang dignidad ng kanyang pamilya.",
]
c1_full_short = [False] * len(c1_full)

# ─── CASE 2: Semantic-only (no named characters) ──────────────────────────────
c2_buod = [
    "Ang kaluluwa ng tao ay naghahanap ng katarungan sa gitna ng kawalan ng pag-asa.",
]
c2_full = [
    "Ang buhay ay puno ng pagsubok na walang katapusan.",
    "Sa dilim ng mundo, nagsisikap pa rin ang tao na mahanap ang kaliwanagan.",
    "Walang pag-asa na hatid ang lipunan sa mga dukha.",
    "Sa harap ng kawalang-katarungan, nananatili ang puso ng tao.",
    "Ang kaluluwa ay laging naghahanap ng katotohanan kahit sa pinakamalim na dilim.",
    "Hindi matutumbok ng anumang parusa ang pagnanais ng tao para sa katuwiran.",
    "Ito ang katotohanan na nararamdaman ng bawat isa.",
]
c2_full_short = [False] * len(c2_full)

# ─── CASE 3: Lexical-driven (keyword-overlap dominant) ────────────────────────
c3_buod = [
    "Ang punong-bayan ay nagpulong kasama ang mga pinuno ng simbahan tungkol sa paaralan.",
]
c3_full = [
    "Dumating ang mga kapitan mula sa iba't ibang lugar.",
    "Iniulat ng gobernadorcillo ang kalagayan ng bayan sa kapwa opisyal.",
    "Ang punong-bayan ay nagtipon ng mga pinuno upang talakayin ang pagtatayo ng paaralan.",
    "Maraming bumabangga sa ideya ng pagtatayo ng bagong paaralan.",
    "Ang mga pare ay may sariling kuro-kuro tungkol dito.",
    "Nagkakaisa ang lahat sa kahalagahan ng edukasyon para sa mga bata.",
    "Kaya naman ipinulong ito ng punong opisyal ng bayan.",
]
c3_full_short = [False] * len(c3_full)

# ─── CASE 4: High-confidence (semantic + lexical + tauhan all strong) ─────────
c4_buod = [
    "Sinabi ni Padre Salvi kay Maria Clara na dapat niyang sundin ang utos ng simbahan.",
]
c4_full = [
    "Nang dumating si Padre Salvi ay tiningnan niya si Maria Clara nang may kakaibang tingin.",
    "Sinabi niya: 'Dapat mong sundin ang kalooban ng Diyos at ng Simbahan.'",
    "Tumango si Maria Clara nang tahimik at hindi makatingin sa pari.",
    "Alam ng lahat na may kapangyarihan ang pari sa kanyang buhay.",
    "Walang makakatanggi sa utos ng isang kaparian.",
    "Nagpatuloy si Padre Salvi sa kaniyang sermon ng walang pahintulot.",
]
c4_full_short = [False] * len(c4_full)

def make_embeddings(texts, dim=384):
    """Use random but consistent embeddings for audit purposes."""
    rng = np.random.default_rng(42)
    return rng.random((len(texts), dim)).astype(np.float32)

cases = [
    ("Case 1 – Character-Assisted", c1_buod, c1_full, c1_full_short),
    ("Case 2 – Semantic-Driven (No Characters)", c2_buod, c2_full, c2_full_short),
    ("Case 3 – Lexical-Driven / Keyword Overlap", c3_buod, c3_full, c3_full_short),
    ("Case 4 – High-Confidence Multi-Signal", c4_buod, c4_full, c4_full_short),
]

for label, buod, full, full_short in cases:
    b_embs = make_embeddings(buod)
    f_embs = make_embeddings(full)
    results, debug = aligner.align(buod, full, b_embs, f_embs, full_short, return_debug=True)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Buod: {buod[0][:80]}...")
    print(f"  Matched window full sentences [{results[0].best_window_start}:{results[0].best_window_end}]:")
    for idx in range(results[0].best_window_start, results[0].best_window_end+1):
        print(f"    [{idx}] {full[idx][:75]}")
    print(f"  Lexical Score:  {results[0].lexical_score:.4f}")
    print(f"  Semantic Score: {results[0].semantic_score:.4f}")
    print(f"  Tauhan Score:   {results[0].tauhan_score:.4f}")
    print(f"  Position Score: {results[0].position_score:.4f}")
    print(f"  FINAL SCORE:    {results[0].final_score:.4f}")
    print(f"  Matched chars:  {results[0].matched_characters}")
