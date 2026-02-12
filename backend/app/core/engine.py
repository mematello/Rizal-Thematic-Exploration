import numpy as np
import re
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from sqlalchemy import select
from pgvector.sqlalchemy import Vector

from app.core.config import get_settings
from app.core.analyzer import QueryAnalyzer, extract_words
from app.models.database import Sentence

settings = get_settings()

class RizalEngine:
    def __init__(self):
        print("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(settings.BERT_MODEL_NAME)
        self.query_analyzer = QueryAnalyzer()
        
        # Tuning parameters (from vbest.py)
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.08
        self.HIGH_STOPWORD_RATIO = 0.6
        self.STOPWORD_PENALTY_FACTOR = 0.5
        
        # Context expansion settings
        self.MAX_CONTEXT_EXPANSION = 3
        self.NEIGHBOR_RELEVANCE_THRESHOLD = 0.40
        
        # Load vocabulary for validation
        self.vocabulary = self._load_vocabulary()

    def _load_vocabulary(self):
        vocab = set()
        import os
        import pandas as pd
        
        # Define paths relative to backend/app/core/
        # __file__ = backend/app/core/engine.py
        # dirname = backend/app/core
        # ../ = backend/app
        # ../../ = backend
        # ../../../ = root (Rizal-Thematic-Exploration)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) 
        csv_dir = os.path.join(base_dir, 'csvFiles')
        
        files = ['noli_chapters.csv', 'elfili_chapters.csv']
        
        print(f"Loading vocabulary from: {csv_dir}")
        print(f"Files to check: {files}")
        for f in files:
            path = os.path.join(csv_dir, f)
            if os.path.exists(path):
                print(f"Processing {f}...")
                try:
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip() # Strip whitespace from headers
                    print(f"Columns in {f}: {df.columns.tolist()}")
                    # Assuming column 'sentence_text' exists
                    if 'sentence_text' in df.columns:
                        count = 0
                        for text in df['sentence_text'].dropna():
                            words = extract_words(str(text).lower())
                            vocab.update(words)
                            count += 1
                        print(f"Added words from {count} rows in {f}")
                    else:
                        print(f"WARNING: 'sentence_text' column not found in {f}")
                except Exception as e:
                    print(f"Error loading vocab from {f}: {e}")
            else:
                print(f"File not found: {path}")
                    
        print(f"Vocabulary loaded: {len(vocab)} unique words.")
        return vocab

    def search(self, db: Session, query: str, top_k: int = 10):
        # 0. Validate Query (Gibberish Filter & Semantic Consistency)
        if not self._validate_query_semantics(db, query):
             print(f"Query '{query}' rejected: Semantically inconsistent.")
             return {'noli': [], 'elfili': []}
             
        query_words = extract_words(query.lower())
        valid_words = [w for w in query_words if w in self.vocabulary]
        
        # If no words from the query exist in our corpus, return empty results
        if not valid_words:
            print(f"Query '{query}' rejected: No words found in corpus.")
            return {'noli': [], 'elfili': []}

        # 1. Generate Embedding
        query_embedding = self.model.encode(query, show_progress_bar=False)
        
        # 2. Vector Search (Semantic Candidates)
        # Using cosine distance (<=> operator in pgvector)
        # Convert numpy array to list for SQLAlchemy
        query_list = query_embedding.tolist()
        
        # Fetch candidates for Noli
        noli_candidates = db.scalars(
            select(Sentence)
            .filter(Sentence.book == 'noli')
            .order_by(Sentence.embedding.cosine_distance(query_list))
            .limit(top_k * 2)
        ).all()

        # Fetch candidates for Fili
        fili_candidates = db.scalars(
            select(Sentence)
            .filter(Sentence.book == 'elfili')
            .order_by(Sentence.embedding.cosine_distance(query_list))
            .limit(top_k * 2)
        ).all()
        
        candidates = list(noli_candidates) + list(fili_candidates)
        
        # 3. Hybrid Re-ranking
        results = {'noli': [], 'elfili': []}
        
        query_analysis = self.query_analyzer.analyze_query_words(query)
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        
        # Coverage Check Preparation: Identify significant words and their embeddings
        sig_items = [item for item in query_analysis if not item['is_stopword']]
        sig_words = [item['word'].lower() for item in sig_items]
        
        if sig_words:
            sig_vecs = self.model.encode(sig_words, show_progress_bar=False)
            # Normalize sig vecs
            norm_sig_vecs = []
            for vec in sig_vecs:
                n = np.linalg.norm(vec)
                norm_sig_vecs.append(vec / n if n > 0 else vec)
        else:
            norm_sig_vecs = []

        for sent in candidates:
            # Normalize vectors to ensure dot product = cosine similarity
            v_query = query_embedding / np.linalg.norm(query_embedding)
            v_sent = np.array(sent.embedding)
            v_sent = v_sent / np.linalg.norm(v_sent)
            
            # --- COVERAGE CHECK ---
            # Ensure every significant query word is represented
            fully_covered = True
            sent_words_set = set(extract_words(sent.sentence_text.lower()))
            
            for i, sw in enumerate(sig_words):
                # 1. Lexical Match (Exact)
                if sw in sent_words_set:
                    continue
                
                # 2. Semantic Match (Word vs Sentence Vector)
                # If the sentence is highly relevant to the concept of the word
                # Threshold: 0.30 - strict enough to imply relevance, loose enough for long sentences
                word_sim = float(np.dot(norm_sig_vecs[i], v_sent))
                if word_sim >= 0.30:
                    continue
                    
                # If neither, the word is missing
                fully_covered = False
                break
            
            if not fully_covered:
                continue
            # ----------------------

            # Re-compute cosine similarity
            sem_score = float(np.dot(v_query, v_sent))
            
            # Lexical Score
            lex_score = self._compute_lexical_score(query, sent.sentence_text, query_analysis, stopword_ratio)
            
            # Dynamic Weights
            text_len = len(sent.sentence_text.split())
            lambda_lex, lambda_sem = self._compute_dynamic_weights(text_len)
            
            # Final Score
            final_score = self._calculate_clear_score(sem_score, lex_score, lambda_lex, lambda_sem, text_len)
            
            # Fetch Context (Dynamic Expansion)
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

            # Thematic Classification
            themes = self._classify_themes(db, sent.embedding, sent.sentence_text)
            result_item['themes'] = themes
            
            if sent.book == 'noli':
                results['noli'].append(result_item)
            else:
                results['elfili'].append(result_item)
        
        # Sort by final score
        # Helper to deduplicate list of dicts by sentence_text
        def deduplicate_results(result_list):
            seen = set()
            unique = []
            for item in result_list:
                text = item['sentence_text'].strip()
                if text not in seen:
                    seen.add(text)
                    unique.append(item)
            return unique

        # Sort by final score
        # Note: Deduplication should happen BEFORE or AFTER sorting? 
        # If we sort first, we keep the highest scoring version of the duplicate.
        # Although theoretically duplicates should have same score if context is same.
        
        noli_sorted = sorted(results['noli'], key=lambda x: x['scores']['final'], reverse=True)
        results['noli'] = deduplicate_results(noli_sorted)[:top_k]
        
        elfili_sorted = sorted(results['elfili'], key=lambda x: x['scores']['final'], reverse=True)
        results['elfili'] = deduplicate_results(elfili_sorted)[:top_k]
        
        return results

    def _compute_lexical_score(self, query, sentence_text, query_analysis, stopword_ratio):
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        sentence_words = set(extract_words(sentence_lower))

        if query_lower == sentence_lower:
            return 1.0

        # 1. Exact Phrase Match (Regex Word Boundary) - from vbest.py
        # Prioritize exact phrase matches with boundaries over scattered words
        query_pattern = r'\b' + re.escape(query_lower) + r'\b'
        if re.search(query_pattern, sentence_lower):
            # If phrase is found, use length ratio logic from vbest
            # This ensures "Jose Rizal" matches better than just "Jose ... Rizal"
            return min(1.0, len(query_lower) / len(sentence_lower) * 2)

        matched_weight = 0.0
        total_weight = sum(item['semantic_weight'] for item in query_analysis)
        
        if total_weight == 0:
            return 0.0
            
        for item in query_analysis:
            w = item['word'].lower()
            # Strict word matching only (equivalent to regex boundary \b)
            if w in sentence_words:
                matched_weight += item['semantic_weight']
            # Removed unsafe substring check (e.g. "bus" in "abuso") to reduce noise
        
        # Metric 1: Query Coverage
        coverage_score = matched_weight / total_weight
        
        # Metric 2: Sentence Density
        match_count = 0
        for item in query_analysis:
            if item['word'].lower() in sentence_words:
                match_count += 1
        
        density_score = match_count / max(1, len(sentence_words))
        
        # Final Lexical Score = Average (giving more weight to coverage usually)
        # Let's say 70% Coverage, 30% Density to allow variance based on length
        score = (coverage_score * 0.7) + (density_score * 0.3)
        
        if stopword_ratio > self.HIGH_STOPWORD_RATIO:
            penalty = (stopword_ratio - self.HIGH_STOPWORD_RATIO) * self.STOPWORD_PENALTY_FACTOR
            score *= (1.0 - penalty)
            
        return score

    def _compute_dynamic_weights(self, text_length):
        if text_length <= 5: return 0.75, 0.25
        elif text_length <= 10: return 0.65, 0.35
        elif text_length <= 15: return 0.55, 0.45
        elif text_length <= 20: return 0.45, 0.55
        else: return 0.35, 0.65

    def _calculate_clear_score(self, sem, lex, lam_lex, lam_sem, length):
        score = (lam_sem * sem) + (lam_lex * lex)
        if length < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (self.SHORT_SENTENCE_THRESHOLD - length) / self.SHORT_SENTENCE_THRESHOLD
            score -= penalty
        return max(0.0, min(1.0, score))



    def _ensure_themes_loaded(self, db: Session):
        if not hasattr(self, 'theme_cache'):
            # Lazy load themes
            from app.models.database import Theme
            themes = db.query(Theme).all()
            self.theme_cache = [
                {
                    'id': t.id,
                    'tagalog_title': t.tagalog_title,
                    'meaning': t.meaning,
                    'embedding': np.array(t.embedding) / np.linalg.norm(np.array(t.embedding)),
                    'meaning_len': len(t.meaning.split())
                }
                for t in themes
            ]

    def _validate_query_semantics(self, db: Session, query: str) -> bool:
        """
        Validates the query using strict lexical checking.
        Rejects queries if ANY word is not found in the vocabulary.
        """
        words = extract_words(query.lower())
        if not words: return False

        unknown_words = [w for w in words if w not in self.vocabulary]
        
        if unknown_words:
            print(f"Rejecting query '{query}': Unknown words found: {unknown_words}")
            return False
            
        return True
        
    def _classify_themes(self, db: Session, sentence_embedding, sentence_text):
        self._ensure_themes_loaded(db)
        matches = []
        sent_vec = np.array(sentence_embedding)
        norm_sent = np.linalg.norm(sent_vec)
        if norm_sent > 0:
            sent_vec = sent_vec / norm_sent
            
        sent_len = len(sentence_text.split())

        candidates = []
        for theme in self.theme_cache:
            # Semantic Sim (now true Cosine Sim)
            sem_sim = float(np.dot(sent_vec, theme['embedding']))
            sem_sim = max(0.0, min(1.0, sem_sim))
            
            # Use same query analysis logic for text? Or just word overlap?
            # vbest.py uses simple lexical overlap:
            lex_score = self._compute_simple_lexical(sentence_text, theme['meaning'])
            
            # Dynamic weights for themes
            lambda_lex, lambda_sem = self._compute_dynamic_weights_by_length(theme['meaning_len'], sent_len)
            
            theme_score = (lambda_sem * sem_sim) + (lambda_lex * lex_score)
            
            candidates.append({
                'id': str(theme['id']),
                'label': theme['tagalog_title'],
                'score': theme_score,
                'explanation': theme['meaning']
            })
        
        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Pick best one (Top 1)
        unique_matches = []
        seen_labels = set()
        
        # 1. Try to find the best match above threshold
        for m in candidates:
            if m['score'] >= 0.45:
                if m['label'] not in seen_labels:
                    unique_matches.append(m)
                    seen_labels.add(m['label'])
                    break # Limit to 1
        
        # 2. Fallback: If no theme met the threshold, force the absolute best candidate
        if not unique_matches and candidates:
            best = candidates[0]
            unique_matches.append(best)

        return unique_matches

    def _compute_simple_lexical(self, text1, text2):
        w1 = set(extract_words(text1.lower()))
        w2 = set(extract_words(text2.lower()))
        if not w2: return 0.0
        intersection = w1.intersection(w2)
        return len(intersection) / len(w2)

    def _compute_dynamic_weights_by_length(self, meaning_len, sent_len):
        ratio = meaning_len / max(1, sent_len)
        # Logic from vbest.py roughly:
        # If meaning is much longer, rely more on semantic?
        # vbest logs:
        # lambda_lex = 0.4 if ratio < 0.5 else 0.6
        # lambda_sem = 1 - lambda_lex
        if ratio < 0.5:
            return 0.4, 0.6
        return 0.6, 0.4

    def _expand_context(self, db: Session, center_sentence: Sentence) -> str:
        # 1. Fetch range
        max_dist = self.MAX_CONTEXT_EXPANSION
        range_start = center_sentence.sentence_index - max_dist
        range_end = center_sentence.sentence_index + max_dist
        
        candidates = db.scalars(
            select(Sentence)
            .filter(
                Sentence.book == center_sentence.book,
                Sentence.chapter_number == center_sentence.chapter_number,
                Sentence.sentence_index >= range_start,
                Sentence.sentence_index <= range_end
            )
        ).all()
        
        # Organize by index
        sent_map = {s.sentence_index: s for s in candidates}
        
        # Expand backward
        prefix = []
        for i in range(1, max_dist + 1):
            idx = center_sentence.sentence_index - i
            if idx not in sent_map: break
            
            neighbor = sent_map[idx]
            score = self._compute_neighbor_score(center_sentence, neighbor)
            
            if score >= self.NEIGHBOR_RELEVANCE_THRESHOLD:
                prefix.append(neighbor.sentence_text)
            else:
                break
        
        # Expand forward
        suffix = []
        for i in range(1, max_dist + 1):
            idx = center_sentence.sentence_index + i
            if idx not in sent_map: break
            
            neighbor = sent_map[idx]
            score = self._compute_neighbor_score(center_sentence, neighbor)
            
            if score >= self.NEIGHBOR_RELEVANCE_THRESHOLD:
                suffix.append(neighbor.sentence_text)
            else:
                break
                
        # Join (reverse prefix because we appended 1-away, 2-away...)
        return " ".join(reversed(prefix)) + f" <strong>{center_sentence.sentence_text}</strong> " + " ".join(suffix)

    def _compute_neighbor_score(self, center, neighbor):
        # Semantic
        v_center = np.array(center.embedding)
        v_neigh = np.array(neighbor.embedding)
        # Normalize
        norm_c = np.linalg.norm(v_center)
        norm_n = np.linalg.norm(v_neigh)
        if norm_c == 0 or norm_n == 0: return 0.0
        
        score_sem = np.dot(v_center, v_neigh) / (norm_c * norm_n)
        
        # Lexical (Inline Jaccard)
        w1 = set(extract_words(center.sentence_text.lower()))
        w2 = set(extract_words(neighbor.sentence_text.lower()))
        if not w1 or not w2:
            score_lex = 0.0
        else:
            score_lex = len(w1.intersection(w2)) / len(w1.union(w2))
        
        # Balanced weights (simplified from vbest)
        return (0.6 * score_sem) + (0.4 * score_lex)

@lru_cache()
def get_engine():
    return RizalEngine()
