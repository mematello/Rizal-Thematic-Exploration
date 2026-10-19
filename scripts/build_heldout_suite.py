
import sys
from pathlib import Path
_repo_root = Path(__file__).resolve()
while _repo_root.parent != _repo_root and not (_repo_root / "scripts" / "prod_guard.py").exists():
    _repo_root = _repo_root.parent
if str(_repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(_repo_root / "scripts"))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
import json
import random
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from app.core.engine import get_engine, extract_words

engine = get_engine()

# 1. All EN_TL_DICTIONARY keys
en_tl_keys = sorted(list(engine.EN_TL_DICTIONARY.keys()))

# 2. 100 Tagalog vocabulary words sampled across frequency bands
# Let's inspect engine.theme_word_df or word frequency in vocab
vocab = sorted(list(engine.vocabulary))
# Sort vocab by df if available, else sample across alphabetical/length bands
random.seed(42)
# Sample across length and alphabetical distribution
short_words = [w for w in vocab if 4 <= len(w) <= 6 and w not in engine.LOW_INFO_WORDS]
mid_words = [w for w in vocab if 7 <= len(w) <= 9 and w not in engine.LOW_INFO_WORDS]
long_words = [w for w in vocab if len(w) >= 10 and w not in engine.LOW_INFO_WORDS]

random.shuffle(short_words)
random.shuffle(mid_words)
random.shuffle(long_words)

tagalog_vocab = sorted(short_words[:35] + mid_words[:35] + long_words[:30])

# 3. 50 two- to three-word phrases
phrases = [
    "sigaw ng bayan", "lihim ng simbahan", "aral sa paaralan", "tinig ng katotohanan",
    "anak ng bayan", "lupit ng prayle", "pag-asa ng kabataan", "dangal ng pamilya",
    "buhay sa piitan", "kasalanan at kapatawaran", "yaman at kapangyarihan", "hirap ng magsasaka",
    "liwanag at dilim", "pananampalataya at pag-ibig", "pagbabalik ni ibarra", "hustisya para sa dukha",
    "sumpa ng bayan", "luha ng ina", "dunong at karunungan", "lihim na pag-ibig",
    "dusa ng bilanggo", "katapusan ng kasamaan", "paglilitis kay basilio", "panaginip ni sisa",
    "tinig sa kadiliman", "sigaw ng katarungan", "pag-aalsa sa gabi", "habilin ng guro",
    "pangako sa kasintahan", "sumpa ng nakaraan", "laban sa korapsyon", "dasal ng maralita",
    "dugo ng bayani", "kamatayan sa lawa", "kasunduan ng makapangyarihan", "banta ng digmaan",
    "lihim ng kumbento", "yapak sa lansangan", "tanglaw sa dilim", "sukat ng pagtitiis",
    "pighati ng puso", "dunong ng matanda", "panata sa simbahan", "anghel ng tahanan",
    "hapis at hilahil", "sibat ng katotohanan", "bukas na aklat", "sigwa ng panahon",
    "tinig ng kunsensya", "ganti ng inapi"
]

# 4. 15 names
names = [
    "Crisostomo Ibarra", "Maria Clara", "Padre Damaso", "Elias", "Kapitan Tiago",
    "Sisa", "Basilio", "Crispin", "Pilosopo Tasyo", "Donya Victorina",
    "Simoun", "Isagani", "Paulita Gomez", "Kabesang Tales", "Juli"
]

# 5. 10 queries containing a period (route-level aggregation / multi-sentence queries)
period_queries = [
    "Noli Me Tangere. El Filibusterismo.",
    "Bayan ko. Mahal ko.",
    "Katarungan. Kalayaan.",
    "Edukasyon ang lunas. Paaralan ang pag-asa.",
    "Simbahan at pamahalaan. Prayle at heneral.",
    "Buhay ni Simoun. Kamatayan ni Ibarra.",
    "Dasal ng ina. Luha ng anak.",
    "Yaman ng bayan. Hirap ng mamamayan.",
    "Gabi ng paghihimagsik. Araw ng kalayaan.",
    "Sakit ng lipunan. Kanser ng bayan."
]

suite = {
    "total_queries": len(en_tl_keys) + len(tagalog_vocab) + len(phrases) + len(names) + len(period_queries),
    "categories": {
        "en_tl_dictionary": en_tl_keys,
        "tagalog_vocabulary": tagalog_vocab,
        "phrases": phrases,
        "names": names,
        "period_queries": period_queries
    }
}

suite_path = Path(__file__).resolve().parent / "heldout_queries.json"
with open(suite_path, "w", encoding="utf-8") as f:
    json.dump(suite, f, indent=2, ensure_ascii=False)

print(f"Built heldout query suite with {suite['total_queries']} queries at {suite_path}")
for cat, qs in suite["categories"].items():
    print(f"  {cat:<20}: {len(qs)} queries")
