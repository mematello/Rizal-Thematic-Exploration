
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prod_guard import verify_not_production_db
verify_not_production_db(caller_name=Path(__file__).name)

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

base_dir = Path(__file__).resolve().parent.parent
backend_dir = base_dir / "backend"
sys.path.insert(0, str(backend_dir))
load_dotenv(backend_dir / ".env")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from sqlalchemy import select, func
from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine

db = SessionLocal()
engine = get_engine()

queries = ["bayan", "simbahan", "paaralan"]

print("=" * 85)
print("STAGE ROW COUNTS FOR BAYAN, SIMBAHAN, PAARALAN")
print("=" * 85)

for q in queries:
    print(f"\n==========================================")
    print(f"QUERY: '{q}'")
    print(f"==========================================")

    stopword_ratio = engine.query_analyzer.get_stopword_ratio(q)
    query_analysis = engine.query_analyzer.analyze_query_words(q)
    is_single_word = len(q.split()) == 1
    result_mode = "lexical"

    # Query vector
    q_emb = engine.base_model.encode(q, show_progress_bar=False)
    q_emb_norm = q_emb / (q_emb ** 2).sum() ** 0.5

    for b_key, b_name in [("noli", "noli"), ("elfili", "elfili")]:
        # 1. Total rows in DB matching ILIKE without limit
        cnt_all = db.scalar(
            select(func.count(Sentence.id)).filter(
                Sentence.book == b_name,
                Sentence.source_type == "summary",
                Sentence.sentence_text.ilike(f"%{q}%")
            )
        )

        # 2. SQL query executed by engine (ORDER BY Sentence.id LIMIT 200)
        stmt = (
            select(Sentence)
            .filter(Sentence.book == b_name, Sentence.source_type == "summary")
            .filter(Sentence.sentence_text.ilike(f"%{q}%"))
            .order_by(Sentence.id)
            .limit(200)
        )
        sql_rows = db.scalars(stmt).all()
        sql_count = len(sql_rows)

        # 3. Post-validation and thresholding
        dropped_strict_lex = []
        dropped_thresh = []
        post_val = []

        for sent in sql_rows:
            lex_score = engine._compute_lexical_score(q, sent.sentence_text, query_analysis, stopword_ratio)

            # Line 1241 strict lex check
            if is_single_word and result_mode == "lexical" and lex_score < 0.1:
                dropped_strict_lex.append(sent.id)
                continue

            v_sent = sent.embedding / (sent.embedding ** 2).sum() ** 0.5
            sem_score = float(q_emb_norm @ v_sent)
            final_score = sem_score * 0.4 + lex_score * 0.6
            precision_score = lex_score

            # Line 1469 threshold check
            min_threshold = 0.45 if not is_single_word else 0.05
            p_thresh = 0.60
            if final_score < min_threshold and precision_score < p_thresh:
                dropped_thresh.append(sent.id)
                continue

            post_val.append({
                "id": sent.id,
                "text": sent.sentence_text,
                "scores": {
                    "final": round(final_score * 100),
                    "precision": round(precision_score * 100)
                }
            })

        post_val_count = len(post_val)

        # 4. Seen text deduplication (Line 1527-1533)
        post_val.sort(key=lambda x: (x["scores"]["final"] + x["scores"]["precision"] * 0.5), reverse=True)
        seen_text = set()
        unique = []
        dropped_dedupe = []
        for itm in post_val:
            txt = itm["text"].strip()
            if txt not in seen_text:
                seen_text.add(txt)
                unique.append(itm)
            else:
                dropped_dedupe.append(itm["id"])

        post_dedupe_count = len(unique)
        final_top10 = unique[:10]
        final_count = len(final_top10)

        print(f"\n  [{b_name.upper()}]:")
        print(f"    Stage 1 - SQL Rows in DB (SELECT count(*)): {cnt_all}")
        print(f"    Stage 2 - SQL Rows retrieved (LIMIT 200):  {sql_count}")
        print(f"              (Truncated by LIMIT 200: {cnt_all - sql_count} rows dropped at line 963)")
        print(f"    Stage 3 - Strict lex filter (<0.1, line 1241): dropped {len(dropped_strict_lex)} rows")
        print(f"    Stage 4 - Min threshold (<0.05, line 1469):   dropped {len(dropped_thresh)} rows")
        print(f"    Stage 5 - Post-validation/threshold count: {post_val_count}")
        print(f"    Stage 6 - Post-dedupe (seen_text, line 1531): {post_dedupe_count} (dropped {len(dropped_dedupe)} duplicate text rows)")
        print(f"    Stage 7 - Final pool (top-10 cut, line 1549): {final_count}")
