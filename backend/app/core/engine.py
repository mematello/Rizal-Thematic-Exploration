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
             return {'noli': [], 'elfili': []}
             
        query_words = extract_words(query.lower())
        valid_words = [w for w in query_words if w in self.vocabulary]
        
        if not valid_words:
            return {'noli': [], 'elfili': []}

        # Query Analysis for dynamic behavior
        query_analysis = self.query_analyzer.analyze_query_words(query)
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        sig_items = [item for item in query_analysis if not item['is_stopword']]
        sig_words = [item['word'].lower() for item in sig_items]
        
        # If there are 2 or more significant words, it's NOT a single-word query
        is_single_word = len(sig_words) < 2
        
        # 1. Embedding Generation
        if is_single_word:
            # PURELY Base XLM for single word
            query_embedding = self.base_model.encode(query, show_progress_bar=False)
            dapt_query_embedding = None 
        else:
            # Dynamic Mix for multi-word
            base_emb = self.base_model.encode(query, show_progress_bar=False)
            
            # Use structured info + DAPT for adaptive mix
            structured_info = self.parser.structured_string(query)
            combined_query = f"{query} [SEP] {structured_info}"
            dapt_query_embedding = self.dapt_model.encode(combined_query, show_progress_bar=False)
            
            # Hybrid selection embedding
            query_embedding = (base_emb + dapt_query_embedding) / 2

        query_list = query_embedding.tolist()

        # 2. Candidate Selection
        candidates = []
        books = {
            'noli': 'Noli Me Tangere' if source_type == 'full' else 'noli',
            'elfili': 'El Filibusterismo' if source_type == 'full' else 'elfili'
        }

        for key, book_name in books.items():
            # A. Lexical (Exact word match)
            lex_word = sig_words[0] if sig_words else query_words[0]
            lex_results = db.scalars(
                select(Sentence)
                .filter(Sentence.book == book_name)
                .filter(Sentence.source_type == source_type)
                .filter(Sentence.sentence_text.ilike(f"%{lex_word}%"))
                .limit(top_k * 20)
            ).all()
            
            # B. Semantic (Vector) - Only if not strictly seeking literal
            if not is_single_word:
                sem_results = db.scalars(
                    select(Sentence)
                    .filter(Sentence.book == book_name)
                    .filter(Sentence.source_type == source_type)
                    .order_by(Sentence.embedding.cosine_distance(query_list))
                    .limit(top_k * 15)
                ).all()
            else:
                sem_results = []
            
            seen = set()
            for c in list(lex_results) + list(sem_results):
                if c.id not in seen:
                    candidates.append(c)
                    seen.add(c.id)

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
            
            # STRICT FILTER: For single words, must have lexical match
            if is_single_word and lex_score < 0.1:
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
            match_sig = any(w in sent_words_set for w in sig_words)
            if not match_sig and sig_words:
                sem_score *= 0.2
                lex_score *= 0.05

            # Dynamic Weights
            text_len = len(sent.sentence_text.split())
            query_sig_len = len(sig_words)
            
            if is_single_word:
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
            result_item['themes'] = self._classify_themes(db, sent.embedding, sent.sentence_text)
            
            if 'noli' in sent.book.lower():
                results['noli'].append(result_item)
            else:
                results['elfili'].append(result_item)
        
        def finalize(lst, label):
            lst.sort(key=lambda x: x['scores']['final'], reverse=True)
            if lst:
                print(f"DEBUG: Top score for {label}: {lst[0]['scores']['final']}%")
            seen_text = set()
            unique = []
            for itm in lst:
                txt = itm['sentence_text'].strip()
                if txt not in seen_text:
                    seen_text.add(txt)
                    unique.append(itm)
            return unique[:top_k]

        results['noli'] = finalize(results['noli'], 'noli')
        results['elfili'] = finalize(results['elfili'], 'elfili')
        return results

    def _compute_lexical_score(self, query, sentence_text, query_analysis, stopword_ratio):
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        sentence_words = set(extract_words(sentence_lower))

        if query_lower == sentence_lower: return 1.0
        
        query_pattern = r'\b' + re.escape(query_lower) + r'\b'
        if re.search(query_pattern, sentence_lower):
            return min(1.0, 0.5 + (len(query_lower) / len(sentence_lower)))

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
            
            if self.theme_matrix:
                self.theme_matrix = np.array(self.theme_matrix)
            else:
                self.theme_matrix = None

    def _validate_query_semantics(self, db: Session, query: str) -> bool:
        words = extract_words(query.lower())
        if not words: return False
        unknown_words = [w for w in words if w not in self.vocabulary]
        return len(unknown_words) == 0
        
    def _classify_themes(self, db: Session, sentence_embedding, sentence_text):
        self._ensure_themes_loaded(db)
        if self.theme_matrix is None or sentence_embedding is None:
            return []
            
        sent_vec = np.array(sentence_embedding)
        sent_norm = np.linalg.norm(sent_vec)
        if sent_norm > 0: 
            sent_vec = sent_vec / sent_norm
        else:
            return []

        # Vectorized dot product for all themes
        sem_sims = np.dot(self.theme_matrix, sent_vec)
        
        candidates = []
        for i, theme in enumerate(self.theme_cache):
            sem_sim = max(0.0, min(1.0, float(sem_sims[i])))
            lex_score = self._compute_simple_lexical(sentence_text, theme['meaning'])
            theme_score = (0.5 * sem_sim) + (0.5 * lex_score)
            candidates.append({'id': str(theme['id']), 'label': theme['tagalog_title'], 'score': theme_score})
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:1]

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
