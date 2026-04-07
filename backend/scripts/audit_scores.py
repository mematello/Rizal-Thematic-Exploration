import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.core.robust_aligner import RobustAligner

aliases_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'character_aliases.json')
with open(aliases_path, encoding='utf-8') as f:
    entries = json.load(f)

tauhan_map = {}
for e in entries:
    c = e.get('canonical') or e.get('name', '')
    for a in e.get('aliases', [c]):
        tauhan_map[a.lower()] = c

al = RobustAligner(tauhan_map=tauhan_map)
rng = np.random.default_rng(42)

cases = [
    ('C1-CharAssisted',
     ['Pinaghandaan ni Ibarra ang kaniyang pagtatanghal sa harap ng mga panauhin.'],
     ['Ang lahat ay abala sa pagdating ng okasyon.',
      'Si Ibarra ay nagbihis nang maingat at inihanda ang kaniyang sarili.',
      'Maraming panauhin ang dumating upang manood.',
      'Naghanda siya ng matagal para sa pagkakataong ito.',
      'Ang buong bayan ay interesado sa magiging resulta.',
      'Kaya naman nag-ingat si Ibarra sa bawat detalye.']),
    ('C2-SemanticOnly',
     ['Ang kaluluwa ng tao ay naghahanap ng katarungan sa gitna ng kawalan ng pag-asa.'],
     ['Ang buhay ay puno ng pagsubok na walang katapusan.',
      'Sa dilim ng mundo, nagsisikap pa rin ang tao.',
      'Walang pag-asa na hatid ang lipunan sa mga dukha.',
      'Sa harap ng kawalang-katarungan, nananatili ang puso ng tao.',
      'Ang kaluluwa ay naghahanap ng katotohanan sa pinakamalim na dilim.',
      'Hindi matutumbok ng anumang parusa ang pagnanais ng tao.']),
    ('C3-LexicalDriven',
     ['Ang punong-bayan ay nagpulong kasama ang mga pinuno ng simbahan tungkol sa paaralan.'],
     ['Dumating ang mga kapitan mula sa ibang lugar.',
      'Iniulat ng gobernadorcillo ang kalagayan ng bayan.',
      'Ang punong-bayan ay nagtipon ng mga pinuno upang talakayin ang pagtatayo ng paaralan.',
      'Maraming bumabangga sa ideya ng pagtatayo ng bagong paaralan.',
      'Ang mga pare ay may sariling kuro-kuro tungkol dito.',
      'Nagkakaisa ang lahat sa kahalagahan ng edukasyon para sa mga bata.']),
    ('C4-HighConf',
     ['Sinabi ni Padre Salvi kay Maria Clara na dapat niyang sundin ang utos ng simbahan.'],
     ['Nang dumating si Padre Salvi ay tiningnan niya si Maria Clara nang may kakaibang tingin.',
      'Sinabi niya dapat mong sundin ang kalooban ng Diyos at ng Simbahan.',
      'Tumango si Maria Clara nang tahimik at hindi makatingin sa pari.',
      'Alam ng lahat na may kapangyarihan ang pari sa kanyang buhay.',
      'Walang makakatanggi sa utos ng isang kaparian.',
      'Nagpatuloy si Padre Salvi sa kaniyang sermon.']),
]

lines = []
for label, buod, full in cases:
    b = rng.random((len(buod), 384)).astype('float32')
    f = rng.random((len(full), 384)).astype('float32')
    res, _ = al.align(buod, full, b, f, [False]*len(full), return_debug=True)
    m = res[0]
    line = f"{label}|lex={m.lexical_score:.4f}|sem={m.semantic_score:.4f}|tau={m.tauhan_score:.4f}|pos={m.position_score:.4f}|fin={m.final_score:.4f}|chars={m.matched_characters}|win=[{m.best_window_start}:{m.best_window_end}]"
    lines.append(line)

out_path = os.path.join(os.path.dirname(__file__), 'audit_scores.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Written to audit_scores.txt")
print('\n'.join(lines))
