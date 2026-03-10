
import sys
import os
import numpy as np
import pandas as pd
from sqlalchemy import select
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.engine import get_engine, extract_words
from app.models.database import SessionLocal, Sentence

# Load .env
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env')))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def visualize_scoring(query, target_text=None):
    print(f"\nInitializing Rizal Engine...")
    engine = get_engine()
    db = SessionLocal()

    # 1. Get Target Sentence
    sentence = None
    if target_text:
        # Try to find exact match in DB first to get its embedding
        sentence = db.scalar(select(Sentence).filter(Sentence.sentence_text == target_text).limit(1))
        if not sentence:
            print(f"Target sentence not found in DB. Using raw text for simulation (no pre-computed embedding).")
    else:
        # Perform a search to find the best match
        print(f"Searching for top result for '{query}'...")
        results = engine.search(db, query, top_k=1)
        # Flatten results
        all_res = results.get('noli', []) + results.get('elfili', [])
        if not all_res:
            print("No results found.")
            return
        best = all_res[0]
        print(f"Top Result Found: \"{best['sentence_text']}\"")
        target_text = best['sentence_text']
        sentence = db.scalar(select(Sentence).filter(Sentence.id == best['id']).limit(1))

    # 2. Replicate Engine Logic for Visualization
    print("\n" + "="*60)
    print(f" ANALYSIS: '{query}'  vs  '{target_text[:50]}...'")
    print("="*60)

    # A. Query Processing
    query_words = extract_words(query.lower())
    query_analysis = engine.query_analyzer.analyze_query_words(query)
    stopword_ratio = engine.query_analyzer.get_stopword_ratio(query)
    sig_items = [item for item in query_analysis if not item['is_stopword']]
    sig_words = [item['word'].lower() for item in sig_items]
    is_single_word = len(sig_words) < 2

    print(f"\n[1] Query Analysis:")
    print(f"    - Significant Words: {sig_words}")
    print(f"    - Is Single Word Mode: {is_single_word}")
    
    # Query Expansion (Visualizing the logic inside engine.search)
    expanded_queries = [query]
    if is_single_word:
        synonyms = {
            'edukasyon': ['pag-aaral', 'paaaralan', 'estudyante', 'guro', 'karunungan'],
            'pag-aaral': ['edukasyon', 'paaralan', 'estudyante', 'leksyon', 'karunungan'],
            'kamatayan': ['namatay', 'patay', 'bangkay', 'libing'],
            'paglilitis': ['hukuman', 'litis', 'pari', 'sentensya', 'kasalanan'],
            'kababata': ['kaibigan', 'kalaro', 'bata']
        }
        if query.lower() in synonyms:
            expanded = synonyms[query.lower()]
            print(f"    - Expanded Synonyms: {expanded}")
            expanded_queries.extend(expanded)

    # B. Embeddings
    print(f"\n[2] Embedding Generation:")
    # Base Model
    base_emb = engine.base_model.encode(query, show_progress_bar=False)
    
    # DAPT Logic (Replicated)
    dapt_emb = None
    if engine.has_dapt:
        dapt_emb_single = engine.dapt_model.encode(query, show_progress_bar=False)
        structured_info = engine.parser.structured_string(query)
        combined_query = f"{query} [SEP] {structured_info}"
        dapt_emb_ctx = engine.dapt_model.encode(combined_query, show_progress_bar=False)
        
        if is_single_word:
            query_embedding = (base_emb * 0.4) + (dapt_emb_single * 0.6)
            dapt_query_embedding = dapt_emb_single
            print("    - Strategy: Single Word Mix (40% Base + 60% DAPT)")
        else:
            query_embedding = (base_emb + dapt_emb_ctx) / 2
            dapt_query_embedding = dapt_emb_ctx
            print("    - Strategy: Multi Word Hybrid (Base + DAPT Context)")
    else:
        query_embedding = base_emb
        dapt_query_embedding = None
        print("    - Strategy: Base Model Only (DAPT not found)")

    # Normalize
    v_query_base = base_emb / np.linalg.norm(base_emb)
    v_query_final = query_embedding / np.linalg.norm(query_embedding)
    v_query_dapt = dapt_query_embedding / np.linalg.norm(dapt_query_embedding) if dapt_query_embedding is not None else None

    # Sentence Embedding
    if sentence and sentence.embedding is not None:
        v_sent = np.array(sentence.embedding)
    else:
        v_sent = engine.base_model.encode(target_text)
    
    norm_sent = np.linalg.norm(v_sent)
    v_sent = v_sent / norm_sent if norm_sent > 0 else v_sent

    # C. Scoring Breakdown
    print(f"\n[3] Scoring Breakdown:")
    
    # Lexical
    lex_score = engine._compute_lexical_score(query, target_text, query_analysis, stopword_ratio)
    print(f"    A. Lexical Score: {lex_score:.4f}")

    # Semantic (Sentence Level)
    sem_score_base_full = float(np.dot(v_query_base, v_sent))
    print(f"    B. Semantic (Sentence-Level Base): {sem_score_base_full:.4f}")

    # Semantic (Word Level - MaxSim)
    # Re-encode words to show the matrix
    t_words = target_text.split()
    # We use the tokenizer's tokens for accuracy in the engine, but split() for visualization
    # Let's perform the matrix calc using the engine's token method for accuracy
    candidate_tokens = engine.base_model.encode([target_text], output_value='token_embeddings', show_progress_bar=False)[0]
    # Normalize tokens
    norms = np.linalg.norm(candidate_tokens, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    c_tokens_norm = candidate_tokens / norms
    
    word_sims = np.dot(c_tokens_norm, v_query_base)
    sem_score_base_word = float(np.max(word_sims))
    print(f"    C. Semantic (MaxSim Word-Level): {sem_score_base_word:.4f}")

    # DAPT Score
    sem_score_dapt = 0.0
    if v_query_dapt is not None:
        sem_score_dapt_full = float(np.dot(v_query_dapt, v_sent))
        
        # DAPT Word Level
        c_dapt_tokens = engine.dapt_model.encode([target_text], output_value='token_embeddings', show_progress_bar=False)[0]
        d_norms = np.linalg.norm(c_dapt_tokens, axis=1, keepdims=True)
        d_norms[d_norms == 0] = 1.0
        c_dapt_tokens_norm = c_dapt_tokens / d_norms
        
        dapt_word_sims = np.dot(c_dapt_tokens_norm, v_query_dapt)
        sem_score_dapt_word = float(np.max(dapt_word_sims))
        
        sem_score_dapt = max(sem_score_dapt_full, sem_score_dapt_word)
        print(f"    D. DAPT Score (Max of Full {sem_score_dapt_full:.4f} / Word {sem_score_dapt_word:.4f}): {sem_score_dapt:.4f}")

    # Combined Semantic
    if is_single_word:
         sem_score_base = max(sem_score_base_full * 0.8, sem_score_base_word)
         sem_score = max(sem_score_base, sem_score_dapt) if v_query_dapt is not None else sem_score_base
    else:
         sem_score_base = max(sem_score_base_full, sem_score_base_word)
         sem_score = max(sem_score_base, sem_score_dapt) if v_query_dapt is not None else sem_score_base
    
    print(f"    E. Combined Semantic Score: {sem_score:.4f}")

    # Final Calculation
    text_len = len(target_text.split())
    query_sig_len = len(sig_words)
    
    if is_single_word:
        print(f"    F. Weighting (Single Word Mode):")
        if lex_score > 0.5:
            print("       - Lexical Match Found (High Confidence)")
            final_score = (lex_score * 0.6) + (sem_score * 0.4)
            print(f"       - Formula: (Lex * 0.6) + (Sem * 0.4)")
        else:
            print("       - Semantic Focus (Synonym Search)")
            final_score = sem_score * 1.5
            print(f"       - Formula: Sem * 1.5 (Boosted)")
    else:
        print(f"    F. Weighting (Multi-Word Dynamic):")
        lam_lex, lam_sem = engine._compute_dynamic_weights(text_len, query_sig_len)
        print(f"       - Dynamic Weights: Lex {lam_lex:.2f} / Sem {lam_sem:.2f}")
        final_score = engine._calculate_clear_score(sem_score, lex_score, lam_lex, lam_sem, text_len)

    # Clip
    final_score = max(0.0, min(1.0, final_score))
    print(f"\n[4] FINAL SCORE: {final_score:.4f} ({round(final_score*100)}%)")

    # D. VISUALIZATION MATRIX
    print(f"\n[5] Visualization Matrix (Query vs Sentence Tokens)")
    print("(Comparison using Base Model Embeddings)")
    
    # We map the token similarities back to words roughly for display
    # Note: Tokens != Words, but we'll try to align them or just show the token interactions
    # Since we can't easily get the tokenizer's string map here without importing the tokenizer explicitly,
    # we will use the user-friendly approach: compare explicit words from split()
    
    q_words_display = query.split()
    s_words_display = target_text.split()
    
    q_vecs = engine.base_model.encode(q_words_display)
    s_vecs = engine.base_model.encode(s_words_display) # Embed words individually for matrix
    
    # Normalize
    q_vecs = q_vecs / np.linalg.norm(q_vecs, axis=1, keepdims=True)
    s_vecs = s_vecs / np.linalg.norm(s_vecs, axis=1, keepdims=True)
    
    # Header
    # Print first 10 words of sentence across top
    display_limit = 10
    s_display = s_words_display[:display_limit]
    header = " " * 15 + "".join([f"{w[:10]:>12}" for w in s_display])
    if len(s_words_display) > display_limit:
        header += " ..."
    print(header)
    print("-" * len(header))
    
    for i, qw in enumerate(q_words_display):
        row = f"{qw:<15}|"
        for j, sw in enumerate(s_display):
            sim = np.dot(q_vecs[i], s_vecs[j])
            
            # Color coding (ANSI)
            val_str = f"{sim:.2f}"
            if sim > 0.6:
                val_str = f"\033[92m{val_str}\033[0m" # Green
            elif sim > 0.4:
                val_str = f"\033[93m{val_str}\033[0m" # Yellow
            elif sim < 0.2:
                val_str = f"\033[90m{val_str}\033[0m" # Gray
                
            row += f" {val_str:>10}" # Use 10 char width for alignment (codes don't count visually but do for format)
            # Alignment with colors is tricky in python f-strings without library, doing manual pad
            padding = 12 - len(f"{sim:.2f}")
            # Simplified: just print standard
        print(row)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_search_logic.py \"<query>\" [\"<optional_target_sentence>\"]")
        print("Example: python visualize_search_logic.py \"edukasyon\" \"Isinantabi niya ang kanyang pag-aaral.\"")
        sys.exit(1)
        
    q = sys.argv[1]
    t = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_scoring(q, t)
