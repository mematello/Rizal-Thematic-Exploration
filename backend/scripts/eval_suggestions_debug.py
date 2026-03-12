import sys
import os
import re
import math
from collections import Counter
from app.models.database import SessionLocal
from app.core.engine import get_engine
from app.core.analyzer import extract_words
from app.services.suggestions import DynamicSuggestionGenerator

def debug_suggestions():
    queries = [
        "edukasyon",
        "bakit mahalaga ang edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "elias",
        "maria clara",
        "prayle sa pilipinas",
        "pag-aaral ng kabataan",
        "umupo sa silya",
        "tiktok"
    ]
    
    db = SessionLocal()
    engine = get_engine()
    
    print("\n================ EVALUATING SUGGESTION RANDOMNESS ================\n")
    for q in queries:
        print(f"\n#################################################################")
        print(f"--- Query: '{q}' ---")
        
        # Suppress prints from the internal logic for clean reporting
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            resp = engine.search(db, q, source_type="full")
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout

        # Extract context
        results_noli = resp.get("results", {}).get("noli", [])
        results_fili = resp.get("results", {}).get("elfili", [])
        top_combined = sorted(results_noli + results_fili, key=lambda x: x['scores']['final'], reverse=True)[:5]
        
        meta = resp.get("metadata", {})
        matched_themes = meta.get("matched_themes", [])
        anchor_score = meta.get("anchor_score", 0.0)
        
        print(f"\n[ENGINE CONTEXT]")
        print(f"Top Combined Results: {len(top_combined)}")
        print(f"Theme Anchor Score: {anchor_score:.3f}")
        print(f"Matched Themes: {matched_themes}")
        
        # Re-run suggestion logic verbosely
        if not top_combined:
            print(">>> Suppression Condition 1: Zero results")
            continue
            
        query_cleaned = q.lower().strip()
        if len(query_cleaned) < 3:
            print(">>> Suppression Condition: Query too short")
            continue
            
        query_words = set(extract_words(query_cleaned))
        candidates = []
        layer_a_triggered = False
        
        is_strong_theme = anchor_score > 0.40
        if is_strong_theme and matched_themes:
            primary_theme = matched_themes[0].lower()
            for key, curations in DynamicSuggestionGenerator.THEMATIC_SUGGESTIONS.items():
                if key in primary_theme or any(w in primary_theme for w in key.split()):
                    candidates.extend(curations)
                    layer_a_triggered = True
                    print(f"\n[LAYER A: THEMATIC]")
                    print(f"Matched Key: '{key}'")
                    print(f"Curations Added: {curations}")
                    break
        else:
            print("\n[LAYER A: THEMATIC] Not triggered.")

        print(f"\n[LAYER B: TF-IDF]")
        doc_texts = []
        for i, res in enumerate(top_combined[:3]):
            text = res.get('context_text') or res.get('sentence_text', '')
            text = re.sub(r'<[^>]+>', ' ', text).lower()
            doc_texts.append(text)
            print(f"  Passage {i+1}: {text[:100]}...")
            
        if doc_texts:
            term_freqs = Counter()
            doc_freqs = Counter()
            for doc in doc_texts:
                words = extract_words(doc)
                valid_words = [w for w in words if len(w) > 3 and w not in DynamicSuggestionGenerator.STOPWORDS and w not in query_words]
                term_freqs.update(valid_words)
                doc_freqs.update(set(valid_words))
            
            num_docs = len(doc_texts)
            tfidf_scores = {}
            for term, tf in term_freqs.items():
                df = doc_freqs[term]
                idf = math.log(num_docs / df) + 1.0 if df > 0 else 0
                tfidf_scores[term] = tf * idf
                
            sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top 10 Ranked TF-IDF Terms:")
            for t, s in sorted_terms[:10]:
                print(f"    - {t}: {s:.2f}")
                
            top_tf_candidates = []
            for term, score in sorted_terms:
                if score < 1.5:
                    continue
                top_tf_candidates.append(term)
                if len(top_tf_candidates) >= 3:
                    break
                    
            print(f"  Tokens Passing Entropy Filter (<1.5 cutoff): {top_tf_candidates}")
            layer_b_cands = []
            for bw in top_tf_candidates:
                c1 = f"{query_cleaned} at {bw}"
                if len(query_words) == 1:
                    c2 = f"mga {bw}"
                    candidates.append(c2)
                    layer_b_cands.append(c2)
                candidates.append(c1)
                layer_b_cands.append(c1)
            print(f"  Layer B Candidates Added: {layer_b_cands}")
            
        print(f"\n[POST-PROCESSING]")
        print(f"  All Intermediate Candidates: {candidates}")
        final_suggestions = []
        seen = set([query_cleaned])
        for cand in candidates:
            c_clean = cand.lower().strip()
            words_in_c = extract_words(c_clean)
            if c_clean in seen:
                continue
            if all(w in DynamicSuggestionGenerator.STOPWORDS for w in words_in_c):
                continue
            seen.add(c_clean)
            final_suggestions.append(c_clean)
            if len(final_suggestions) == 5:
                break
                
        if len(final_suggestions) < 2:
            print(">>> Suppression Condition 2: < 2 valid suggestions")
            final_suggestions = []
            
        print(f"\n[FINAL SUGGESTIONS]")
        for i, s in enumerate(final_suggestions, 1):
            source = "Layer A" if layer_a_triggered and i <= len(DynamicSuggestionGenerator.THEMATIC_SUGGESTIONS.get(matched_themes[0].split()[0] if matched_themes else "", [])) else "Layer B"
            if layer_a_triggered:
                # approximation since we extended A then B
                is_a = any(s in v for v in DynamicSuggestionGenerator.THEMATIC_SUGGESTIONS.values())
                source = "Layer A" if is_a else "Layer B"
            print(f"  {i}. {s} ({source})")

if __name__ == "__main__":
    debug_suggestions()
