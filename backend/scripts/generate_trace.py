import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.database import SessionLocal, Sentence
from app.core.engine import get_engine
from app.core.analyzer import QueryAnalyzer, extract_words
from sqlalchemy import select

def print_trace():
    db = SessionLocal()
    engine = get_engine()
    query = "artificial intelligence at makinarya ng kolonyal na pamahalaan"
    source_type = "summary"
    
    print("\n==================================")
    print("        FULL RETRIEVAL TRACE      ")
    print("==================================")
    
    # 1. Validation & Preprocessing
    engine.query_analyzer.STOPWORDS = engine.query_analyzer._load_official_stopwords()
    
    print("\n--- 1. Query Validation Output ---")
    pass_blocklist = not any(term in query.lower() for term in engine.MODERN_TERMS)
    print(f"Passed modern-term blocklist check? {'Yes' if pass_blocklist else 'No'}")
    
    query_words = extract_words(query.lower())
    analysis = engine.query_analyzer.analyze_query_words(query)
    stopword_ratio = engine.query_analyzer.get_stopword_ratio(query)
    
    print("\nLinguistic Validation (WordFreq & Valid Ratio):")
    for a in analysis:
        word_type = "stopword" if a['is_stopword'] else "content"
        print(f" - {a['word']:<15} : freq={a['frequency']:.6f}, {word_type}")
    print(f"Valid word ratio: {1.0 - stopword_ratio:.2f}")

    q_vec_base = engine.base_model.encode(query, show_progress_bar=False)
    q_vec_base_norm = q_vec_base / np.linalg.norm(q_vec_base) if np.linalg.norm(q_vec_base) > 0 else q_vec_base
    
    engine._ensure_themes_loaded(db)
    theme_scores = np.dot(engine.theme_matrix, q_vec_base_norm)
    max_theme_score = float(np.max(theme_scores))
    print(f"\nDomainAlignment Score (Max Theme Score): {max_theme_score:.4f}")
    
    is_oov = any(w not in engine.vocabulary for w in query_words)
    threshold = 0.50 if is_oov else 0.40
    print(f"ThemeProximity Gate Threshold: {threshold:.2f} (OOV={is_oov})")
    
    validation_pass = pass_blocklist and len(query_words) > 0
    print(f"Did it pass or fail validation? {'Pass' if validation_pass else 'Fail'} (Blocklist pass & length > 0)")


    print("\n--- 2. Query Preprocessing Output ---")
    sig_words = [item['word'].lower() for item in analysis if not item['is_stopword']]
    total_query_weight = sum(a['semantic_weight'] for a in analysis)
    
    print("Word-by-word Semantic Weight breakdown:")
    for a in analysis:
        w_type = "stopword" if a['is_stopword'] else "content word"
        print(f" - {a['word']:<15} : {w_type}, Weight={a['semantic_weight']:.2f}")
    print(f"Total query weight: {total_query_weight:.2f}")
    print(f"Query embedding generated? Yes")


    print("\n--- 3. Stage A — Lexical Retrieval ---")
    # Simulate DB Stage A logic
    books = {'noli': 'noli', 'elfili': 'elfili'}
    candidates_A = []
    for key, book_name in books.items():
        stmt = select(Sentence).filter(Sentence.book == book_name).filter(Sentence.source_type == source_type)
        if len(sig_words) >= 2:
            for w in sig_words:
                stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{w}%"))
        else:
            lex_word = sig_words[0] if sig_words else query_words[0]
            stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{lex_word}%"))
        res = db.scalars(stmt.limit(20)).all()
        candidates_A.extend(res)
    
    top_k = 10  # default top_k=10
    needs_fallback = len(candidates_A) < top_k * 2
    
    print(f"How many candidates retrieved via ILIKE? {len(candidates_A)}")
    print(f"Did Stage A return >= 20 candidates? {'Yes' if len(candidates_A) >= 20 else 'No'}")
    print(f"Was Stage B triggered? {'Yes' if needs_fallback else 'No'}")
    if len(candidates_A) > 0:
        print("List of top Lexical candidates before scoring:")
        for c in candidates_A[:3]:
            print(f" - [ID: {c.id}] {c.sentence_text[:80]}...")


    print("\n--- 4. Stage B — Semantic Fallback ---")
    candidates_B = []
    if needs_fallback:
        print("Stage B activated? Yes, because lexical candidates < 20.")
        print(f"OOV domain gate threshold results: max_score={max_theme_score:.4f} vs threshold={threshold:.2f}")
        for key, book_name in books.items():
            sem_res = db.scalars(
                select(Sentence)
                .filter(Sentence.book == book_name)
                .filter(Sentence.source_type == source_type)
                .order_by(Sentence.embedding.cosine_distance(q_vec_base))
                .limit(top_k * 15)
            ).all()
            # Only add to candidate count if not already in A
            seen_ids_A = [cand.id for cand in candidates_A]
            sem_new = [c for c in sem_res if c.id not in seen_ids_A]
            candidates_B.extend(sem_new)
        print(f"How many candidates retrieved via pgvector (excluding Stage A)? {len(candidates_B)}")
    else:
        print("Stage B activated? No")

    
    print("\n--- Running engine.search() to get Final Results ---")
    response = engine.search(db, query, top_k=10, source_type=source_type)
    results = response['results']['noli'] + response['results']['elfili']
    results.sort(key=lambda x: x['scores']['final'], reverse=True)
    
    meta = response['metadata']
    result_mode = meta['result_mode']
    components = meta.get('components', engine._get_query_components(query))
    component_vecs = [engine.base_model.encode(comp, show_progress_bar=False) for comp in components]
    component_vecs = [v / np.linalg.norm(v) if np.linalg.norm(v)>0 else v for v in component_vecs]

    print("\n--- 5. Formula 1 Scoring — Top 3 Results ---")
    for idx, item in enumerate(results[:3]):
        print(f"\nResult #{idx+1}:")
        print(f"  Sentence: {item['sentence_text']}")
        print(f"  Chapter: {item['chapter_number']} - {item.get('chapter_title', 'Unknown')}")
        print(f"  Novel: {'Noli Me Tangere' if 'Noli' in item.get('book', item.get('chapter_title', '')) or idx<len(response['results']['noli']) else 'El Filibusterismo'} (Warning exact logic varies)")
        
        words_len = len(extract_words(item['sentence_text'].lower()))
        print(f"  Passage word count: {words_len}")
        
        # Recalculate weights and intermediate scores
        lam_lex, lam_sem = engine._compute_dynamic_weights(words_len, len(sig_words))
        print(f"  Dynamic weights: lam_lex={lam_lex:.2f}, lam_sem={lam_sem:.2f}")
        
        lex_score = engine._compute_lexical_score(query, item['sentence_text'], analysis, stopword_ratio)
        v_query = q_vec_base_norm
        v_sent = np.array(item['embedding']) if 'embedding' in item else db.query(Sentence).filter_by(id=item['id']).first().embedding
        v_sent = np.array(v_sent)
        v_sent = v_sent / np.linalg.norm(v_sent) if np.linalg.norm(v_sent)>0 else v_sent
        sem_score = float(np.dot(v_query, v_sent))
        
        struct_info = engine.parser.structured_string(query)
        dapt_q = engine.dapt_model.encode(f"{query} [SEP] {struct_info}", show_progress_bar=False)
        hybrid_q = (q_vec_base + dapt_q) / 2
        hybrid_q_norm = hybrid_q / np.linalg.norm(hybrid_q)
        sem_score_base = sem_score
        sem_score_dapt = float(np.dot(dapt_q / np.linalg.norm(dapt_q), v_sent))
        sem_score_hybrid = max(sem_score_base, sem_score_dapt)
        
        # Penalties logic
        penalty_text = []
        is_exclamation = "!" in item['sentence_text'] and words_len < 8
        if words_len < 5 or is_exclamation:
            penalty_text.append("Short Sentence/Exclamation Penalty (sem_score *= 0.5)")
        
        # Coverage
        sent_words_set = set(extract_words(item['sentence_text'].lower()))
        coverage_count = 0
        threshold_cov = 0.55
        sig_vecs = [engine.base_model.encode(sw, show_progress_bar=False) for sw in sig_words]
        sig_vecs = [v / np.linalg.norm(v) if np.linalg.norm(v)>0 else v for v in sig_vecs]
        
        for i, sw in enumerate(sig_words):
            if sw in sent_words_set or sw in item['sentence_text'].lower():
                coverage_count += 1
            else:
                if float(np.dot(sig_vecs[i], v_sent)) >= threshold_cov:
                    coverage_count += 1
        cvg = coverage_count / len(sig_words) if sig_words else 1.0
        if cvg < 1.0:
            penalty = 1.0 - cvg
            penalty_text.append(f"Coverage Penalty ({cvg:.2f})")
            
        print(f"  S_sem (semantic score - check base vs hybrid): base={sem_score_base:.4f}, max={sem_score_hybrid:.4f}")
        
        matched_weight = 0
        total_weight = sum(a['semantic_weight'] for a in analysis)
        match_count = 0
        for a in analysis:
            if not a['is_stopword'] and a['word'].lower() in item['sentence_text'].lower(): match_count += 1
            if a['word'].lower() in sent_words_set: matched_weight += a['semantic_weight']
            elif a['word'].lower() in item['sentence_text'].lower(): matched_weight += a['semantic_weight'] * 0.8
        
        density = match_count / max(1, words_len)
        coverage_val = matched_weight / total_weight if total_weight > 0 else 0
        print(f"  S_lex (lexical): raw={lex_score:.4f} (coverage={coverage_val:.2f}, density={density:.2f})")
        
        if len(penalty_text) > 0:
            print(f"  Penalties applied: {', '.join(penalty_text)}")
        else:
            print(f"  Penalties applied: None")
            
        print(f"  Score_final: {item['scores']['final']}%")
        print(f"  Match type classification: {item.get('concept_match_type', 'unknown')}")
        print(f"  Geometric Mean Precision score: {item['scores'].get('precision', 0)}%")
        
        if idx == 0:
            top_sent = db.query(Sentence).filter_by(id=item['id']).first()
    
    print("\n--- 6. Formula 2 — Context Expansion (Top Result Only) ---")
    if len(results) > 0:
        range_start, range_end = top_sent.sentence_index - 3, top_sent.sentence_index + 3
        neighbors = db.scalars(select(Sentence).filter(
            Sentence.book == top_sent.book,
            Sentence.chapter_number == top_sent.chapter_number,
            Sentence.source_type == top_sent.source_type,
            Sentence.sentence_index >= range_start,
            Sentence.sentence_index <= range_end
        )).all()
        
        for nn in neighbors:
            if nn.id == top_sent.id: continue
            v1, v2 = np.array(top_sent.embedding), np.array(nn.embedding)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            sem = np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0
            w1, w2 = set(extract_words(top_sent.sentence_text.lower())), set(extract_words(nn.sentence_text.lower()))
            lex = len(w1 & w2) / len(w1 | w2) if w1 | w2 else 0
            score_nb = (0.6 * sem) + (0.4 * lex)
            passed = score_nb >= engine.NEIGHBOR_RELEVANCE_THRESHOLD
            print(f"  Neighbor [{nn.sentence_index}] {nn.sentence_text[:30]}...")
            print(f"   -> S_sem: {sem:.4f}, LexicalSim: {lex:.4f}")
            print(f"   -> Score_neighbor: {score_nb:.4f} | Included? {'Yes' if passed else 'No'}")
    else:
        print("  *(No top result available for context expansion)*")

    print("\n--- 7. Thematic Classification ---")
    if len(results) > 0:
        themes = engine._classify_themes(db, top_sent, query, query_vec=q_vec_base_norm)
        print(f"Assigned themes to top result:")
        for t in themes:
            print(f" - {t['label']} (score: {t['score']:.4f})")
        
        themes_full = []
        # get cosine similarity for all
        v_top = np.array(top_sent.embedding)
        v_top = v_top / np.linalg.norm(v_top)
        all_sims = np.dot(engine.theme_matrix, v_top)
        for t in themes:
            # find index
            for i, x in enumerate(engine.theme_cache):
                if str(x['id']) == str(t['id']):
                    print(f"   Cosine base purely on vector: {all_sims[i]:.4f}")
        
        if len(themes) > 0:
            print("Classification was treated as Thematic Analysis + Hybrid Semantic (due to theme assignments crossing thresholds).")
    else:
        print("  *(No top result available for thematic classification)*")


    print("\n--- 8. Final Output ---")
    print("Final ranked list of results:")
    for i, res in enumerate(results[:5]):
        print(f"{i+1}. [FINAL: {res['scores']['final']}% | PREC: {res['scores'].get('precision',0)}%] {res['sentence_text'][:80]}")
    
    print(f"Result mode: {meta['result_mode']}")
    print(f"has_lexical_hits flag: {meta['has_lexical_hits']}")
    print(f"retrieval_stage (reason): {meta['reason']}")

    db.close()

if __name__ == '__main__':
    print_trace()
