import numpy as np
import re
import os
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import select, or_
from pgvector.sqlalchemy import Vector

from app.core.config import get_settings
from app.core.analyzer import QueryAnalyzer, extract_words
from app.models.database import Sentence
from app.core.tagalog_parser import TagalogRoleParser
from app.services.suggestions import DynamicSuggestionGenerator

settings = get_settings()

class RizalEngine:
    def __init__(self):
        print("Loading SentenceTransformer models...")
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dapt_path = os.path.join(base_path, 'models', 'rizal-xlm-r-dapt')
        
        # Load Base Model (Always)
        self.base_model = SentenceTransformer(settings.BERT_MODEL_NAME)
        
        # Load DAPT Model if available, else fallback to base
        if os.path.exists(dapt_path):
            print(f"Using DAPT model from {dapt_path}")
            self.dapt_model = SentenceTransformer(dapt_path)
            self.has_dapt = True
        else:
            print(f"DAPT model not found at {dapt_path}. Using base model as fallback.")
            self.dapt_model = self.base_model
            self.has_dapt = False
            
        self.query_analyzer = QueryAnalyzer()
        self.parser = TagalogRoleParser()
        
        # Tuning parameters
        self.SHORT_SENTENCE_THRESHOLD = 10 
        self.SHORT_SENTENCE_PENALTY = 0.25 
        self.HIGH_STOPWORD_RATIO = 0.6
        self.STOPWORD_PENALTY_FACTOR = 0.5
        
        self.MAX_CONTEXT_EXPANSION = 3
        self.NEIGHBOR_RELEVANCE_THRESHOLD = 0.40
        
        self.vocabulary = self._load_vocabulary()
        
        # Hardcoded blocklist for modern terms that might trigger semantic matches
        self.MODERN_TERMS = {
            'tiktok', 'bitcoin', 'crypto', 'facebook', 'instagram', 'twitter', 
            'cellphone', 'internet', 'wifi', 'computer', 'laptop', 'online',
            'covid', 'pandemic', 'app', 'website', 'netflix', 'youtube'
        }
        
        self.LOW_INFO_WORDS = {
            "hindi", "lahat", "walang", "ang", "mga", "ng", "sa", "at", "ay", "ito", "iyan", "iyon", 
            "niya", "nila", "mo", "ko", "siya", "sila", "tayo", "kami", "kayo", "namin", "ano", "sino", 
            "ni", "si", "sina", "nina", "kay", "kina",
            "bakit", "paano", "kailan", "saan", "may", "wala", "din", "rin", "pa", "na", "ba", "nga", 
            "kaya", "pati", "para", "upang", "dahil", "isang", "kanilang", "kanyang", "naging", 
            "masyado", "marami", "siyang", "akong", "aking", "iyong", "ating", "inyong", "kanila", 
            "kanya", "ganyan", "ganito", "ganoon", "naman", "daw", "raw", "muli", "hanggang", "lamang",
            "kaya", "sana", "kapag", "kung", "habang", "kahit", "agad", "tuwing", "tiyak", "ilang",
            "katulad", "mukhang", "lalong", "panahong", "talagang", "bagay", "katwiran", "marahil",
            "upang", "gayon", "ngunit", "datapwat", "subalit", "bagaman", "palibhasa", "sapagkat"
        }

    
    def _get_passage_context(self, db: Session, center: Sentence, window: int = 8) -> str:
        range_start, range_end = center.sentence_index - window, center.sentence_index + window
        candidates = db.scalars(select(Sentence).filter(
            Sentence.book == center.book, 
            Sentence.chapter_number == center.chapter_number, 
            Sentence.source_type == center.source_type,
            Sentence.sentence_index >= range_start, 
            Sentence.sentence_index <= range_end
        ).order_by(Sentence.sentence_index)).all()
        return " ".join([s.sentence_text for s in candidates])

    def _expand_query_with_themes(self, query_vec):
        """
        Finds the most relevant Themes for a query vector and returns 
        (expanded_text, set_of_valid_tokens).
        """
        if self.theme_matrix is None:
            return "", set()

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        else:
            return "", set()
            
        # Compute cosine similarity with all themes
        scores = np.dot(self.theme_matrix, query_vec)
        
        # Get top 3 matching themes
        top_indices = np.argsort(scores)[-3:][::-1]
        
        candidates = set()
        best_title = ""
        target_theme_key = ""
        
        for i, idx in enumerate(top_indices):
            score = scores[idx]
            if score < 0.45: continue
            
            theme = self.theme_cache[idx]
            if i == 0: 
                best_title = theme['tagalog_title']
                target_theme_key = best_title.strip().lower()
            
            # Extract keywords from the meaning
            meaning_words = extract_words(theme['meaning'].lower())
            candidates.update(meaning_words)

        if not candidates or not target_theme_key:
            return "", set()
            
        # Filter candidates based on Specificity
        # P(Theme | Word) = Count(Word in Theme) / Count(Word in All Themes)
        
        final_tokens = []
        
        for w in candidates:
            if len(w) <= 3: continue
            
            # Global DF check (still useful for super common words)
            df = self.theme_word_df.get(w, 0)
            if df < 1: continue 
            
            # Specificity calculation
            theme_counts = self.word_theme_map.get(w, {})
            count_in_target = theme_counts.get(target_theme_key, 0)
            
            if count_in_target == 0: continue
            
            # Specificity score
            specificity = count_in_target / df
            
            # Specificity threshold
            # word must be primarily associated with this theme
            # Lowered to 0.30 to capture terms like "paaralan" which might appear in related themes (Reform, Town)
            # but still filter generic terms like "indio" (low specificity)
            if specificity >= 0.30: 
                final_tokens.append(w)
        
        # Sort by specificity (descending)
        final_tokens.sort(key=lambda w: self.word_theme_map.get(w, {}).get(target_theme_key, 0) / self.theme_word_df.get(w, 1), reverse=True)
        top_tokens = final_tokens[:12]
        
        if not top_tokens:
            return "", set()

        # Construct expansion text
        expansion_text = f"{best_title} {' '.join(top_tokens)}"
        
        return expansion_text, set(top_tokens)

    def _load_vocabulary(self):
        vocab = set()
        import pandas as pd
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) 
        csv_dir = os.path.join(base_dir, 'csvFiles')
        files = ['noli_chapters.csv', 'elfili_chapters.csv', 'fullversion_noli.csv', 'fullversion_elfili.csv']
        
        for f in files:
            path = os.path.join(csv_dir, f)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    if 'sentence_text' in df.columns:
                        for text in df['sentence_text'].dropna():
                            words = extract_words(str(text).lower())
                            vocab.update(words)
                except Exception as e:
                    pass
        return vocab

    def search(self, db: Session, query: str, top_k: int = 10, source_type: str = "summary"):
        if not self._validate_query_semantics(db, query):
             return {
                 'results': {'noli': [], 'elfili': []},
                 'metadata': {'result_mode': 'none', 'reason': 'invalid_query'}
             }
             
        query_words = extract_words(query.lower())
        
        # Query Analysis for dynamic behavior
        query_analysis = self.query_analyzer.analyze_query_words(query)
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        sig_items = [item for item in query_analysis if not item['is_stopword']]
        sig_words = [item['word'].lower() for item in sig_items]
        
        # If there are 2 or more significant words, it's NOT a single-word query
        is_single_word = len(sig_words) < 2
        
        # 1. Embedding Generation
        query_expansion_text = ""
        theme_tokens = set()
        
        # Always attempt to expand query using Themes (both single and multi-word)
        self._ensure_themes_loaded(db)
        # Use base model for initial concept embedding
        query_vec_initial = self.base_model.encode(query, show_progress_bar=False)
        
        expanded_text, extracted_tokens = self._expand_query_with_themes(query_vec_initial)
        if expanded_text:
            if os.getenv("DEBUG_SEARCH"):
                print(f"[DEBUG] Expanding query '{query}' with: {expanded_text[:50]}...")
            
            # Re-embed with expansion
            # We combine query + expansion to form a "Concept Vector"
            combined_text = f"{query} {expanded_text}"
            query_embedding_base = self.base_model.encode(combined_text, show_progress_bar=False)
            theme_tokens = extracted_tokens
        else:
            query_embedding_base = query_vec_initial
            
        if is_single_word:
            query_embedding = query_embedding_base
            dapt_query_embedding = None 
        else:
            # Dynamic Mix for multi-word
            # Use structured info + DAPT for adaptive mix
            structured_info = self.parser.structured_string(query)
            combined_query = f"{query} [SEP] {structured_info}"
            dapt_query_embedding = self.dapt_model.encode(combined_query, show_progress_bar=False)
            
            # Hybrid selection embedding (mix base concept with DAPT syntax)
            query_embedding = (query_embedding_base + dapt_query_embedding) / 2

        query_list = query_embedding.tolist()

        # 2. Candidate Selection (Stage A: Lexical First)
        candidates = []
        books = {
            'noli': 'Noli Me Tangere' if source_type == 'full' else 'noli',
            'elfili': 'El Filibusterismo' if source_type == 'full' else 'elfili'
        }

        lex_word = sig_words[0] if sig_words else query_words[0]
        for key, book_name in books.items():
            stmt = select(Sentence).filter(Sentence.book == book_name).filter(Sentence.source_type == source_type)
            
            if not is_single_word and sig_words:
                # Require all significant words to appear for a true lexical phrase match
                for w in sig_words:
                    stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{w}%"))
            else:
                stmt = stmt.filter(Sentence.sentence_text.ilike(f"%{lex_word}%"))
                
            lex_results = db.scalars(stmt.limit(top_k * 20)).all()
            
            seen = set([c.id for c in candidates])
            for c in lex_results:
                if c.id not in seen:
                    candidates.append(c)
                    seen.add(c.id)

        result_mode = "lexical"
        reason = "matches_found"
        valid_tokens = []
        self.word_sim_cache = {} # Cache for dynamic semantic validation

        # Stage B: Semantic Fallback
        # Run fallback if we have very few lexical candidates overall (e.g. one book had none)
        # or if it's a multi-word query that might need semantic padding due to strict lexical limits.
        needs_fallback = len(candidates) < top_k * 2
        
        if needs_fallback:
            # --- In-Domain Gate ---
            
            # 1. Blocklist Check
            if any(term in query.lower() for term in self.MODERN_TERMS):
                 return {
                        'results': {'noli': [], 'elfili': []},
                        'metadata': {'result_mode': 'none', 'reason': 'out_of_domain'}
                    }

            # 2. Similarity Gate
            # Check if query is anchorable to any theme in the corpus
            self._ensure_themes_loaded(db)
            if self.theme_matrix is not None:
                # Reuse query_list (embedding)
                q_vec = np.array(query_list)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm
                    
                # Compute sim with all themes
                theme_scores = np.dot(self.theme_matrix, q_vec)
                max_theme_score = float(np.max(theme_scores))
                
                # Gate Threshold
                query_tokens = extract_words(query.lower())
                is_oov = any(token not in self.vocabulary for token in query_tokens)
                threshold = 0.50 if is_oov else 0.40
                
                if os.getenv("DEBUG_SEARCH"):
                    print(f"[DEBUG] Theme Anchor Score: {max_theme_score:.3f} | Threshold: {threshold:.2f} | OOV: {is_oov}")
                
                if max_theme_score < threshold:
                    return {
                        'results': {'noli': [], 'elfili': []},
                        'metadata': {'result_mode': 'none', 'reason': 'out_of_domain'}
                    }

            result_mode = "semantic_fallback"
            
            # Build valid tokens using significant query words (no stopwords) + theme tokens
            raw_valid_tokens = set(sig_words) | theme_tokens
            valid_tokens = {t for t in raw_valid_tokens if t.lower() not in self.LOW_INFO_WORDS}
            
            # Pre-compute vectors for each individual significant word to avoid phrase-inflation drift
            sig_word_vecs = []
            for sw in sig_words:
                vec = self.base_model.encode(sw, show_progress_bar=False)
                if np.linalg.norm(vec) > 0:
                    vec = vec / np.linalg.norm(vec)
                sig_word_vecs.append((sw, vec))
            
            if os.getenv("DEBUG_SEARCH"):
                print(f"[DEBUG] Valid tokens for validation: {list(valid_tokens)[:10]}")

            for key, book_name in books.items():
                sem_results = db.scalars(
                    select(Sentence)
                    .filter(Sentence.book == book_name)
                    .filter(Sentence.source_type == source_type)
                    .order_by(Sentence.embedding.cosine_distance(query_list))
                    .limit(top_k * 15)
                ).all()
                
                # print(f"DEBUG: Selected {len(sem_results)} semantic candidates for {book_name}")
                seen = set([c.id for c in candidates])
                for c in sem_results:
                    if c.id not in seen:
                        candidates.append(c)
                        seen.add(c.id)

            if os.getenv("DEBUG_SEARCH"):
                print(f"[DEBUG] Raw semantic candidates retrieved: {len(candidates)}")
                
            if not candidates:
                reason = "no_match"

        # 3. Hybrid Re-ranking
        results = {'noli': [], 'elfili': []}
        
        if sig_words:
            sig_vecs = self.base_model.encode(sig_words, show_progress_bar=False)
            norm_sig_vecs = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in sig_vecs]
        else:
            norm_sig_vecs = []

        for sent in candidates:
            v_sent = np.array(sent.embedding)
            v_sent = v_sent / np.linalg.norm(v_sent) if np.linalg.norm(v_sent) > 0 else v_sent
            
            # Lexical Score Calculation
            lex_score = self._compute_lexical_score(query, sent.sentence_text, query_analysis, stopword_ratio)
            
            # STRICT FILTER: For single words in lexical mode, must have lexical match
            if is_single_word and result_mode == "lexical" and lex_score < 0.1:
                continue

            # Semantic scores
            v_query_base = query_embedding / np.linalg.norm(query_embedding)
            sem_score_base = float(np.dot(v_query_base, v_sent))
            
            if not is_single_word and dapt_query_embedding is not None:
                v_query_dapt = dapt_query_embedding / np.linalg.norm(dapt_query_embedding)
                sem_score_dapt = float(np.dot(v_query_dapt, v_sent))
                sem_score = max(sem_score_base, sem_score_dapt)
            else:
                sem_score = sem_score_base

            # Semantic Fallback Guardrails
            if result_mode == "semantic_fallback":
                # 1. Penalize very short sentences or exclamations
                word_count = len(sent.sentence_text.split())
                is_exclamation = "!" in sent.sentence_text and word_count < 8
                
                if word_count < 5 or is_exclamation:
                    sem_score *= 0.5
                
                # Lexical/Synonym Validation
                text_lower = sent.sentence_text.lower()
                has_validation = False
                matched_token = None
                
                # Check 1: Theme-based tokens
                for token in valid_tokens:
                    if len(token) >= 5:
                        if token in text_lower:
                            has_validation = True
                            matched_token = token
                            break
                    else:
                        if re.search(r'\b' + re.escape(token) + r'[a-z]*', text_lower):
                            has_validation = True
                            matched_token = token
                            break
                
                # Check 2: Dynamic Semantic Validation (for strong synonyms missing from themes)
                # STRICTER: Only allow dynamic validation if at least one significant query word is in the corpus vocabulary.
                # This prevents OOV modern terms (like 'tiktok') from using this loose check.
                vocab_check_passed = any(w in self.vocabulary for w in sig_words)
                if not has_validation and vocab_check_passed:
                    sent_words = set(extract_words(text_lower))
                    # Filter short words and low-info words immediately
                    cand_words = [w for w in sent_words if len(w) > 4 and w not in self.LOW_INFO_WORDS]
                    
                    for w in cand_words:
                        if w in valid_tokens: continue # Already checked
                        
                        # Vocabulary Guard: Only allow words that exist in the corpus vocabulary
                        if w not in self.vocabulary:
                            continue

                        # Check cache
                        if w not in self.word_sim_cache:
                            # Compute similarity on the fly
                            w_vec = self.base_model.encode(w, show_progress_bar=False)
                            if np.linalg.norm(w_vec) > 0:
                                w_vec = w_vec / np.linalg.norm(w_vec)
                            
                            # Use max similarity against any individual significant word
                            max_sim = 0.0
                            if sig_word_vecs:
                                for _, s_vec in sig_word_vecs:
                                    sim = float(np.dot(s_vec, w_vec))
                                    if sim > max_sim:
                                        max_sim = sim
                            else:
                                # Fallback for edge cases where sig_words is empty
                                q_vec = self.base_model.encode(query, show_progress_bar=False)
                                if np.linalg.norm(q_vec) > 0: q_vec = q_vec / np.linalg.norm(q_vec)
                                max_sim = float(np.dot(q_vec, w_vec))
                                
                            self.word_sim_cache[w] = max_sim
                        
                        if self.word_sim_cache[w] > 0.55: # Strict single-word synonym threshold
                            has_validation = True
                            matched_token = f"{w} (dynamic)"
                            # Add to valid_tokens to speed up future checks
                            valid_tokens.add(w) 
                            break
                
                if os.getenv("DEBUG_SEARCH"):
                    print(f"  [DEBUG] ID: {sent.id} | Sem Score: {sem_score:.3f} | Thresh: 0.30")

                if sem_score < 0.30:
                    continue
                
                if not has_validation:
                    if os.getenv("DEBUG_SEARCH"):
                        print(f"  [DEBUG] Filtered by Validation: ID: {sent.id} | Context: {text_lower[:50]}...")
                    reason = "validation_failed"
                    continue
                else:
                    # Semantic Reranking Boost
                    # If it passes validation (contains a concept keyword), we trust it more
                    sem_score += 0.1
                    lex_score = max(lex_score, 0.5) # Ensure it survives the lexical mix
                    
                    if os.getenv("DEBUG_SEARCH"):
                        print(f"  [DEBUG] Passed Validation: ID: {sent.id} | Trigger Token: {matched_token} | Boosted Sem: {sem_score:.3f}")

            # Coverage & Noise Penalty
            sent_words_set = set(extract_words(sent.sentence_text.lower()))
            
            # Multi-word Coverage: Ensure all significant words are represented
            if not is_single_word and sig_words:
                coverage_count = 0
                # Increased threshold to 0.55 to avoid vague semantic matches
                threshold = 0.55 if source_type == 'summary' else 0.60
                
                # STRICT: track lexical and semantic separately
                lexical_matches = 0
                sent_text_lower = sent.sentence_text.lower()
                for i, sw in enumerate(sig_words):
                    if sw in sent_words_set or sw in sent_text_lower:
                        coverage_count += 1
                        lexical_matches += 1
                    else:
                        # Semantic representation check for missing lexical word
                        word_sim = float(np.dot(norm_sig_vecs[i], v_sent))
                        if word_sim >= threshold:
                            coverage_count += 1
                
                coverage_ratio = coverage_count / len(sig_words)
                
                if result_mode != "semantic_fallback":
                    # STRICT FILTER for short queries (2-3 words):
                    # Must have 100% coverage (at least semantically)
                    if len(sig_words) >= 2 and coverage_ratio < 1.0:
                        sem_score = 0.0
                        lex_score = 0.0
                    elif coverage_ratio < 1.0:
                        # General penalty for partial coverage
                        penalty = 1.0 - coverage_ratio
                        sem_score *= (1.0 - (penalty * 0.95))
                        lex_score *= (1.0 - (penalty * 0.98))
                    
                    # Heavy penalty if missing lexical matches for core words in short queries
                    if len(sig_words) >= 2 and lexical_matches < len(sig_words):
                        # If one of the core words is not literal, it must be a VERY strong semantic match
                        sem_score *= 0.5
                        lex_score *= 0.3
                    
                    # Heavy penalty if NO lexical matches exist for any core word
                    if lexical_matches == 0 and len(sig_words) > 0:
                        sem_score *= 0.1
                        lex_score *= 0.05
            
            # Stopword dominance check
            if result_mode == "semantic_fallback":
                match_sig = True  # We already did strict synonym validation
            else:
                match_sig = any(w in sent_words_set for w in sig_words)
                
            if not match_sig and sig_words:
                sem_score *= 0.2
                lex_score *= 0.05

            # Dynamic Weights
            text_len = len(sent.sentence_text.split())
            query_sig_len = len(sig_words)
            
            if is_single_word:
                if result_mode == "semantic_fallback":
                    # Allow semantic score to drive the final score since lexical failed
                    lam_lex, lam_sem = 0.2, 0.8
                    final_score = (lex_score * lam_lex) + (sem_score * lam_sem)
                else:
                    # Pure lexical focus
                    lam_lex, lam_sem = 1.0, 0.0
                    final_score = (lex_score * 0.95) + (sem_score * 0.05)
            else:
                lam_lex, lam_sem = self._compute_dynamic_weights(text_len, query_sig_len)
                final_score = self._calculate_clear_score(sem_score, lex_score, lam_lex, lam_sem, text_len)
            
            # Safety clamp and noise floor
            final_score = max(0.0, min(1.0, final_score))
            
            # Increased noise floor for multi-word queries to prune unrelated results
            min_threshold = 0.45 if not is_single_word else 0.05
            
            if final_score < min_threshold: continue

            context_text = self._expand_context(db, sent)
            result_item = {
                'id': sent.id,
                'chapter_number': sent.chapter_number,
                'chapter_title': sent.chapter_title,
                'sentence_text': sent.sentence_text,
                'context_text': context_text,
                'scores': {
                    'semantic': round(sem_score * 100),
                    'lexical': round(lex_score * 100),
                    'final': round(final_score * 100)
                }
            }
            result_item['themes'] = self._classify_themes(db, sent, query)
            
            if 'noli' in sent.book.lower():
                results['noli'].append(result_item)
            else:
                results['elfili'].append(result_item)
        
        def finalize(lst, label):
            lst.sort(key=lambda x: x['scores']['final'], reverse=True)
            seen_text = set()
            unique = []
            for itm in lst:
                txt = itm['sentence_text'].strip()
                if txt not in seen_text:
                    seen_text.add(txt)
                    unique.append(itm)
            if result_mode == "semantic_fallback":
                return unique[:20] # Take top 20 candidates for reranking
            else:
                return unique[:top_k]

        if result_mode == "semantic_fallback":
            results['noli'] = self._rerank_candidates(query, finalize(results['noli'], 'noli'))[:top_k]
            results['elfili'] = self._rerank_candidates(query, finalize(results['elfili'], 'elfili'))[:top_k]
        else:
            results['noli'] = finalize(results['noli'], 'noli')
            results['elfili'] = finalize(results['elfili'], 'elfili')
        
        if reason == "matches_found" and not (results['noli'] or results['elfili']):
            reason = "filtered_by_ranker"
        elif reason == "validation_failed" and (results['noli'] or results['elfili']):
            reason = "validation_passed"
        elif not (results['noli'] or results['elfili']):
            if result_mode == "semantic_fallback" and reason == "matches_found":
                reason = "below_threshold"

        if os.getenv("DEBUG_SEARCH"):
            print(f"[DEBUG] Final Results -> Noli: {len(results['noli'])}, Fili: {len(results['elfili'])}")

        # 4. Generate Dynamic Search Suggestions ("Kaugnay na Paghahanap")
        matched_theme_string = ""
        anchor_score = 0.0
        
        if getattr(self, 'theme_matrix', None) is not None and getattr(self, 'theme_cache', None):
            q_vec = np.array(query_vec_initial)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
                scores = np.dot(self.theme_matrix, q_vec)
                max_score = float(np.max(scores))
                if max_score > 0:
                    best_idx = int(np.argmax(scores))
                    anchor_score = max_score
                    matched_theme_string = self.theme_cache[best_idx].get('tagalog_title', '').lower()

        theme_metadata = {
            "matched_themes": [matched_theme_string] if matched_theme_string else [],
            "anchor_score": anchor_score
        }
        top_combined = sorted(results['noli'] + results['elfili'], key=lambda x: x['scores']['final'], reverse=True)[:5]
        suggestions = DynamicSuggestionGenerator.generate_suggestions(query, top_combined, theme_metadata)

        return {
            "results": results,
            "metadata": {
                "result_mode": result_mode,
                "reason": reason,
                "suggestions": suggestions
            }
        }

    def _compute_lexical_score(self, query, sentence_text, query_analysis, stopword_ratio):
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        sentence_words = set(extract_words(sentence_lower))

        if query_lower == sentence_lower: return 1.0
        
        query_pattern = r'\b' + re.escape(query_lower) + r'\b'
        exact_phrase_match = re.search(query_pattern, sentence_lower)

        matched_weight = 0.0
        total_weight = sum(item['semantic_weight'] for item in query_analysis)
        if total_weight == 0: return 0.0
            
        for item in query_analysis:
            w = item['word'].lower()
            if w in sentence_words:
                matched_weight += item['semantic_weight']
            elif w in sentence_lower: # Substring match for affixes
                matched_weight += item['semantic_weight'] * 0.8
        coverage = matched_weight / total_weight
        # Stopwords should NOT contribute to match density
        match_count = sum(1 for item in query_analysis if not item['is_stopword'] and item['word'].lower() in sentence_lower)
        density = match_count / max(1, len(sentence_words))
        score = (coverage * 0.8) + (density * 0.2)

        if stopword_ratio > self.HIGH_STOPWORD_RATIO:
            score *= (1.0 - (stopword_ratio - self.HIGH_STOPWORD_RATIO) * self.STOPWORD_PENALTY_FACTOR)
            
        if exact_phrase_match:
            exact_bonus = min(1.0, 0.90 + (len(query_lower) / max(1, len(sentence_lower))))
            return max(score, exact_bonus)

        return score

    def _compute_dynamic_weights(self, text_length, query_length):
        # Base weights on sentence length
        if text_length <= 8: 
            l_lex, l_sem = 0.90, 0.10
        elif text_length <= 15: 
            l_lex, l_sem = 0.70, 0.30
        elif text_length <= 25: 
            l_lex, l_sem = 0.50, 0.50
        else:
            l_lex, l_sem = 0.30, 0.70
            
        # Boost lexical if query is short (2-3 words)
        if query_length <= 2:
            l_lex = max(l_lex, 0.80)
            l_sem = 1.0 - l_lex
        elif query_length <= 4:
            l_lex = max(l_lex, 0.60)
            l_sem = 1.0 - l_lex
            
        return l_lex, l_sem

    def _rerank_candidates(self, query: str, candidates: list) -> list:
        if not candidates:
            return []
            
        # compute embedding for the full query
        query_emb = self.base_model.encode(query, show_progress_bar=False)
        query_norm = np.linalg.norm(query_emb)
        if query_norm > 0:
            query_emb = query_emb / query_norm
            
        for i, cand in enumerate(candidates):
            text = cand['sentence_text']
            # compute embedding for each candidate passage
            text_emb = self.base_model.encode(text, show_progress_bar=False)
            text_norm = np.linalg.norm(text_emb)
            if text_norm > 0:
                text_emb = text_emb / text_norm
                
            # compute cosine similarity
            rerank_score = float(np.dot(query_emb, text_emb))
            
            # Record scores for debugging
            cand['scores']['rerank'] = round(rerank_score * 100)
            cand['scores']['initial_rank'] = i + 1
            
        # rerank candidates based on this score
        candidates.sort(key=lambda x: x['scores']['rerank'], reverse=True)
        return candidates

    def _calculate_clear_score(self, sem, lex, lam_lex, lam_sem, length):
        score = (lam_sem * sem) + (lam_lex * lex)
        if length < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (self.SHORT_SENTENCE_THRESHOLD - length) / self.SHORT_SENTENCE_THRESHOLD
            score -= penalty
        return max(0.0, min(1.0, score))

    def _ensure_themes_loaded(self, db: Session):
        if not hasattr(self, 'theme_cache'):
            from app.models.database import Theme
            themes = db.query(Theme).all()
            self.theme_cache = []
            self.theme_matrix = []
            
            # For IDF and Specificity calculation
            word_doc_freq = {}
            word_theme_map = {} # word -> {theme_title: count}
            total_themes = 0
            
            for t in themes:
                if t.embedding is not None and len(t.embedding) > 0:
                    emb = np.array(t.embedding)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                        self.theme_cache.append({
                            'id': t.id, 'tagalog_title': t.tagalog_title, 'meaning': t.meaning,
                            'meaning_len': len(t.meaning.split())
                        })
                        self.theme_matrix.append(emb)
                        
                        # Count distinct words in this theme for IDF and Specificity
                        words = set(extract_words(t.meaning.lower()))
                        title_key = t.tagalog_title.strip().lower()
                        
                        for w in words:
                            word_doc_freq[w] = word_doc_freq.get(w, 0) + 1
                            
                            if w not in word_theme_map:
                                word_theme_map[w] = {}
                            word_theme_map[w][title_key] = word_theme_map[w].get(title_key, 0) + 1
                            
                        total_themes += 1
            
            if self.theme_matrix:
                self.theme_matrix = np.array(self.theme_matrix)
            else:
                self.theme_matrix = None
                
            self.theme_word_df = word_doc_freq
            self.word_theme_map = word_theme_map
            self.total_themes = total_themes

    def _validate_query_semantics(self, db: Session, query: str) -> bool:
        words = extract_words(query.lower())
        return len(words) > 0
        
    
    def _generate_hybrid_theme_pool(self, query: str, target_pool_size: int = 5) -> list[int]:
        pool_indices = set()
        q_lower = query.lower()
        
        lexical_aliases = {
            "kalayaan": ["Kalayaan at Pagmamahal sa Bayan"],
            "bayan": ["Kalayaan at Pagmamahal sa Bayan"],
            "edukasyon": ["Edukasyon", "Edukasyon at Kalayaan"],
            "paaralan": ["Edukasyon"],
            "simbahan": ["Katiwalian", "Kolonyal na Kaisipan at Paghahangad Umasenso", "Kapangyarihan at Kawalang-Katarungan"],
            "prayle": ["Katiwalian", "Ipokrasya at Pang-aaping Kolonyal"],
            "pang-aapi": ["Kawalang-Katarungan at Katarungan", "Kapangyarihan at Kawalang-Katarungan", "Pang-aapi sa Kababaihan"],
            "kababaihan": ["Pang-aapi sa Kababaihan"],
            "pag-ibig": ["Pag-ibig, Kadalisayan, at Katapatan"]
        }
        
        q_emb_raw = self.base_model.encode(query, show_progress_bar=False)
        if isinstance(q_emb_raw, list): q_emb_raw = np.array(q_emb_raw)
        
        q_norm = np.linalg.norm(q_emb_raw)
        query_vec = q_emb_raw / q_norm if q_norm > 0 else q_emb_raw
        
        q_sims = np.dot(self.theme_matrix, query_vec)
        ranked_indices = np.argsort(q_sims)[::-1]
        
        for i, theme in enumerate(self.theme_cache):
            title_tag = theme['tagalog_title']
            title_lower = title_tag.lower()
            if q_lower in lexical_aliases:
                if title_tag in lexical_aliases[q_lower]:
                    pool_indices.add(i)
            elif title_lower == q_lower or q_lower in title_lower.split():
                 pool_indices.add(i)
                    
        for idx in ranked_indices:
            if len(pool_indices) >= target_pool_size:
                break
            pool_indices.add(idx)
            
        return list(pool_indices) if len(pool_indices) > 0 else list(range(len(self.theme_cache)))

    def _classify_themes(self, db: Session, center_sent: Sentence, query: str = ""):
        self._ensure_themes_loaded(db)
        if self.theme_matrix is None or center_sent.embedding is None:
            return []
            
        q_type = self._determine_query_type(query) if query else "Abstract Phrase"
        pool_indices = self._generate_hybrid_theme_pool(query, target_pool_size=5) if query else list(range(len(self.theme_cache)))
            
        import json
        emb_data = center_sent.embedding
        if isinstance(emb_data, str):
            try:
                emb_data = json.loads(emb_data)
            except:
                return []
        sent_vec = np.array(emb_data, dtype=float)
        if len(sent_vec.shape) > 1:
            sent_vec = sent_vec.flatten()
        sent_norm = np.linalg.norm(sent_vec)
        if sent_norm > 0: sent_vec = sent_vec / sent_norm
        else: return []

        passage_text = self._get_passage_context(db, center_sent, window=8)
        p_emb_raw = self.base_model.encode(passage_text, show_progress_bar=False)
        if isinstance(p_emb_raw, list): p_emb_raw = np.array(p_emb_raw)
        p_norm = np.linalg.norm(p_emb_raw)
        passage_emb = p_emb_raw / p_norm if p_norm > 0 else p_emb_raw
        
        sent_sims = np.dot(self.theme_matrix, sent_vec)
        pass_sims = np.dot(self.theme_matrix, passage_emb)
        
        candidates = []
        for tidx in pool_indices:
            theme = self.theme_cache[tidx]
            meaning = theme.get('meaning', '')
            
            s_score = max(0.0, float(sent_sims[tidx]))
            p_score = max(0.0, float(pass_sims[tidx]))
            lex_score = self._compute_simple_lexical(passage_text, meaning)
            sent_lex = self._compute_simple_lexical(center_sent.sentence_text, meaning)
            
            final_s_score = (0.5 * s_score) + (0.5 * sent_lex)
            final_p_score = (0.5 * p_score) + (0.5 * lex_score)
            
            score = 0
            if q_type == "Character": score = final_p_score
            elif q_type == "Theme Keyword": score = (0.7 * final_s_score) + (0.3 * final_p_score)
            else: score = (0.4 * final_s_score) + (0.6 * final_p_score)
                
            candidates.append({'id': str(theme['id']), 'label': theme['tagalog_title'], 'score': score})
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # UI DISPLAY THRESHOLD (Phase 36 Approval)
        if candidates:
            best_theme = candidates[0]
            
            # Explicit Override Rule: If query explicitly contains the theme keyword,
            # bypass the threshold suppression to avoid confusing UX.
            literal_override = False
            if query and q_type == "Theme Keyword":
                query_words = set([w.lower().strip() for w in query.split()])
                # Split theme label into words. e.g "Edukasyon at Pag-aaral" -> "Edukasyon", "Pag", "aaral"
                theme_words = set([w.lower().strip(' ,.!?"\'') for w in best_theme['label'].split()])
                
                # Check for direct word overlap
                if len(query_words.intersection(theme_words)) > 0:
                    literal_override = True

            # Standard threshold is 0.55 unless explicitly matched directly by string
            if best_theme['score'] >= 0.55 or literal_override:
                return [best_theme]
            
        return []

    def batch_classify_themes(self, db: Session, sentences):
        """
        Classify themes for a batch of sentences (e.g., a whole chapter)
        using optimized matrix operations.
        """
        self._ensure_themes_loaded(db)
        if self.theme_matrix is None or not sentences:
            return [[] for _ in sentences]
            
        # 1. Prepare sentence matrix
        sent_embeddings = []
        valid_mask = []
        
        for s in sentences:
            if s.embedding is not None:
                sent_vec = np.array(s.embedding)
                if len(sent_vec.shape) > 1:
                    sent_vec = sent_vec.flatten()
                norm = np.linalg.norm(sent_vec)
                if norm > 0:
                    sent_embeddings.append(sent_vec / norm)
                    valid_mask.append(True)
                    continue
            valid_mask.append(False)
            
        if not sent_embeddings:
            return [[] for _ in sentences]
            
        sent_matrix = np.array(sent_embeddings) # Shape: (NumValid, 768)
        
        # 2. Compute all similarities at once
        # theme_matrix shape: (NumThemes, 768)
        # Result shape: (NumValid, NumThemes)
        all_sims = np.dot(sent_matrix, self.theme_matrix.T)
        
        # 3. Process results back to requested format
        results = []
        valid_idx = 0
        for i, is_valid in enumerate(valid_mask):
            if not is_valid:
                results.append([])
                continue
                
            sentence_text = sentences[i].sentence_text
            sentence_sims = all_sims[valid_idx]
            valid_idx += 1
            
            candidates = []
            for j, theme in enumerate(self.theme_cache):
                sem_sim = max(0.0, min(1.0, float(sentence_sims[j])))
                lex_score = self._compute_simple_lexical(sentence_text, theme['meaning'])
                theme_score = (0.5 * sem_sim) + (0.5 * lex_score)
                candidates.append({
                    'id': str(theme['id']), 
                    'label': theme['tagalog_title'], 
                    'score': theme_score,
                    'explanation': theme['meaning'][:100] + "..." # Added for the UI
                })
            
            candidates.sort(key=lambda x: x['score'], reverse=True)
            results.append(candidates[:1])
            
        return results

    
    def _determine_query_type(self, query: str) -> str:
        chars = ["elias", "maria clara", "ibarra", "simoun", "crisostomo", "padre", "sisa", "basilio", "crispin"]
        if query.lower() in chars or any(c in query.lower() for c in chars):
            return "Character"
        exact_themes = ["edukasyon", "kalayaan", "simbahan", "bayan", "prayle", "paaralan", "pag-ibig"]
        if query.lower() in exact_themes:
            return "Theme Keyword"
        return "Abstract Phrase"

    def _compute_simple_lexical(self, text1, text2):
        w1, w2 = set(extract_words(text1.lower())), set(extract_words(text2.lower()))
        return len(w1 & w2) / len(w2) if w2 else 0.0

    def _expand_context(self, db: Session, center_sentence: Sentence) -> str:
        max_dist = self.MAX_CONTEXT_EXPANSION
        range_start, range_end = center_sentence.sentence_index - max_dist, center_sentence.sentence_index + max_dist
        candidates = db.scalars(select(Sentence).filter(Sentence.book == center_sentence.book, Sentence.chapter_number == center_sentence.chapter_number, Sentence.sentence_index >= range_start, Sentence.sentence_index <= range_end)).all()
        sent_map = {s.sentence_index: s for s in candidates}
        
        prefix, suffix = [], []
        for i in range(1, max_dist + 1):
            idx = center_sentence.sentence_index - i
            if idx in sent_map and self._compute_neighbor_score(center_sentence, sent_map[idx]) >= self.NEIGHBOR_RELEVANCE_THRESHOLD: prefix.append(sent_map[idx].sentence_text)
            else: break
        for i in range(1, max_dist + 1):
            idx = center_sentence.sentence_index + i
            if idx in sent_map and self._compute_neighbor_score(center_sentence, sent_map[idx]) >= self.NEIGHBOR_RELEVANCE_THRESHOLD: suffix.append(sent_map[idx].sentence_text)
            else: break
        return " ".join(reversed(prefix)) + f" <strong>{center_sentence.sentence_text}</strong> " + " ".join(suffix)

    def _compute_neighbor_score(self, center, neighbor):
        v1, v2 = np.array(center.embedding), np.array(neighbor.embedding)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        sem = np.dot(v1, v2) / (n1 * n2) if n1 > 0 and n2 > 0 else 0
        w1, w2 = set(extract_words(center.sentence_text.lower())), set(extract_words(neighbor.sentence_text.lower()))
        lex = len(w1 & w2) / len(w1 | w2) if w1 | w2 else 0
        return (0.6 * sem) + (0.4 * lex)

@lru_cache()
def get_engine():
    return RizalEngine()
