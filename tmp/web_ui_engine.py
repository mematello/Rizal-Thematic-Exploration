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
        query_words = extract_words(query.lower())
        # Query Analysis for dynamic behavior
        query_analysis = self.query_analyzer.analyze_query_words(query)
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        sig_items = [item for item in query_analysis if not item['is_stopword']]
        sig_words = [item['word'].lower() for item in sig_items]
        
        # If there are 2 or more significant words, it's NOT a single-word query
        is_single_word = len(sig_words) < 2
        
        # Only strict validate if NOT single word
        if not is_single_word:
            if not self._validate_query_semantics(db, query):
                 return {'noli': [], 'elfili': []}
        else:
            if not query_words:
                return {'noli': [], 'elfili': []}
             
        # valid_words = [w for w in query_words if w in self.vocabulary]
        # if not valid_words:
        #     return {'noli': [], 'elfili': []}
        
        # Query Expansion for specific keywords (Single Word focus)
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
                expanded_queries.extend(synonyms[query.lower()])

        # 1. Embedding Generation
        # For single word queries, we now involve DAPT knowledge if available
        base_emb = self.base_model.encode(query, show_progress_bar=False)
        
        # Also encode expanded queries for candidate selection
        expanded_embeddings = []
        if len(expanded_queries) > 1:
            expanded_embeddings = self.base_model.encode(expanded_queries[1:], show_progress_bar=False)

        if self.has_dapt:
            # Get DAPT embedding for the query word
            dapt_emb_single = self.dapt_model.encode(query, show_progress_bar=False)
            
            # Use structured info for contextual DAPT embedding
            structured_info = self.parser.structured_string(query)
            combined_query = f"{query} [SEP] {structured_info}"
            dapt_emb_ctx = self.dapt_model.encode(combined_query, show_progress_bar=False)
            
            if is_single_word:
                # Dynamic Mix for single-word: priority to base for lexical precision, but strong DAPT influence
                query_embedding = (base_emb * 0.4) + (dapt_emb_single * 0.6)
                dapt_query_embedding = dapt_emb_single
            else:
                # Hybrid selection embedding for multi-word
                query_embedding = (base_emb + dapt_emb_ctx) / 2
                dapt_query_embedding = dapt_emb_ctx
        else:
            query_embedding = base_emb
            dapt_query_embedding = None

        query_list = query_embedding.tolist()

        # Normalize query vectors
        v_query_base = base_emb / np.linalg.norm(base_emb)
        v_query_dapt = None
        if dapt_query_embedding is not None:
             v_query_dapt = dapt_query_embedding / np.linalg.norm(dapt_query_embedding)

        # 2. Candidate Selection
        candidates = []
        books = {
            'noli': 'Noli Me Tangere' if source_type == 'full' else 'noli',
            'elfili': 'El Filibusterismo' if source_type == 'full' else 'elfili'
        }

        for key, book_name in books.items():
            # A. Lexical (Exact word match)
            # Use all expanded queries for lexical candidates
            lex_results = []
            for eq in expanded_queries:
                batch = db.scalars(
                    select(Sentence)
                    .filter(Sentence.book == book_name)
                    .filter(Sentence.source_type == source_type)
                    .filter(Sentence.sentence_text.ilike(f"%{eq}%"))
                    .limit(top_k * 10)
                ).all()
                if batch:
                    print(f"DEBUG: Found {len(batch)} lexical candidates for '{eq}' in {book_name}")
                lex_results.extend(batch)
            
            # B. Semantic (Vector) - Always include semantic search
            # SIGNIFICANTLY increase sem_limit for single-word queries to find synonyms
            sem_limit = top_k * 100 if is_single_word else top_k * 20
            
            # Search using original query embedding
            sem_results = db.scalars(
                select(Sentence)
                .filter(Sentence.book == book_name)
                .filter(Sentence.source_type == source_type)
                .order_by(Sentence.embedding.cosine_distance(query_list))
                .limit(sem_limit)
            ).all()
            print(f"DEBUG: Found {len(sem_results)} semantic candidates for '{query}' in {book_name}")
            
            # If expanded queries exist, also get their top semantic matches
            if len(expanded_embeddings) > 0:
                for i, emb in enumerate(expanded_embeddings):
                    batch = db.scalars(
                        select(Sentence)
                        .filter(Sentence.book == book_name)
                        .filter(Sentence.source_type == source_type)
                        .order_by(Sentence.embedding.cosine_distance(emb.tolist()))
                        .limit(top_k * 5)
                    ).all()
                    if batch:
                        print(f"DEBUG: Found {len(batch)} semantic candidates for expanded query '{expanded_queries[i+1]}' in {book_name}")
                    sem_results.extend(batch)
            
            seen = set()
            for c in list(lex_results) + list(sem_results):
                if c.id not in seen:
                    candidates.append(c)
                    seen.add(c.id)
            
            if sem_results:
                top_sem = max(float(np.dot(v_query_base, np.array(s.embedding)/np.linalg.norm(s.embedding))) for s in sem_results if np.linalg.norm(s.embedding) > 0)
                print(f"DEBUG: Top raw semantic score for {book_name}: {top_sem:.4f}")

        # 3. Hybrid Re-ranking
        results = {'noli': [], 'elfili': []}
        
        # Pre-calculate significant word vectors for coverage checks
        if sig_words:
            sig_vecs = self.base_model.encode(sig_words, show_progress_bar=False)
            norm_sig_vecs = [v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v for v in sig_vecs]
        else:
            norm_sig_vecs = []

        # Batch encode and normalize candidate tokens for word-level evaluation
        candidate_texts = [s.sentence_text for s in candidates]
        candidate_tokens_norm = []
        candidate_dapt_tokens_norm = []
        
        if candidate_texts:
            # Base model tokens
            candidate_tokens = self.base_model.encode(candidate_texts, output_value='token_embeddings', show_progress_bar=False)
            for tokens in candidate_tokens:
                norms = np.linalg.norm(tokens, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                candidate_tokens_norm.append(tokens / norms)
            
            # DAPT model tokens if available
            if self.has_dapt:
                candidate_dapt_tokens = self.dapt_model.encode(candidate_texts, output_value='token_embeddings', show_progress_bar=False)
                for tokens in candidate_dapt_tokens:
                    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    candidate_dapt_tokens_norm.append(tokens / norms)

        # Normalize query vectors
        v_query_base = base_emb / np.linalg.norm(base_emb)
        v_query_dapt = None
        if dapt_query_embedding is not None:
             v_query_dapt = dapt_query_embedding / np.linalg.norm(dapt_query_embedding)

        for idx, sent in enumerate(candidates):
            v_sent = np.array(sent.embedding)
            v_sent = v_sent / np.linalg.norm(v_sent) if np.linalg.norm(v_sent) > 0 else v_sent
            
            # Lexical Score Calculation
            lex_score = self._compute_lexical_score(query, sent.sentence_text, query_analysis, stopword_ratio)
            
            # Semantic scores: Combination of Sentence-level and Word-level (MaxSim)
            # Full sentence similarity
            sem_score_base_full = float(np.dot(v_query_base, v_sent))
            
            # Word-level similarity (MaxSim)
            c_tokens_norm = candidate_tokens_norm[idx]
            word_sims = np.dot(c_tokens_norm, v_query_base)
            sem_score_base_word = float(np.max(word_sims))
            
            # DAPT semantic scores
            sem_score_dapt = 0.0
            if v_query_dapt is not None:
                sem_score_dapt_full = float(np.dot(v_query_dapt, v_sent))
                
                # DAPT Word-level (MaxSim)
                c_dapt_tokens_norm = candidate_dapt_tokens_norm[idx]
                dapt_word_sims = np.dot(c_dapt_tokens_norm, v_query_dapt)
                sem_score_dapt_word = float(np.max(dapt_word_sims))
                
                sem_score_dapt = max(sem_score_dapt_full, sem_score_dapt_word)

            # Combine semantic scores
            if is_single_word:
                 # For single words, we prioritize Word-level matches (MaxSim)
                 sem_score_base = max(sem_score_base_full * 0.8, sem_score_base_word)
                 sem_score = max(sem_score_base, sem_score_dapt) if v_query_dapt is not None else sem_score_base
            else:
                 sem_score_base = max(sem_score_base_full, sem_score_base_word)
                 sem_score = max(sem_score_base, sem_score_dapt) if v_query_dapt is not None else sem_score_base

            # RELAXED FILTER: For single words, allow semantic-only matches
            if is_single_word:
                if lex_score < 0.1 and sem_score < 0.15: # Even more relaxed threshold for single word
                    if query in ['paglilitis', 'kababata'] and idx < 5:
                        print(f"DEBUG: Candidate '{sent.sentence_text[:30]}' filtered out (Lex: {lex_score:.2f}, Sem: {sem_score:.2f})")
                    continue
            else:
                # Original filter for multi-word
                if lex_score < 0.05 and sem_score < 0.1:
                    continue
            
            if query in ['paglilitis', 'kababata'] and idx < 50:
                 print(f"DEBUG: Candidate '{sent.sentence_text[:30]}' passed (Lex: {lex_score:.2f}, Sem: {sem_score:.2f})")

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
                        # Improved: Check semantic similarity with individual words/tokens
                        # instead of just the whole sentence
                        word_to_sent_sim = float(np.dot(norm_sig_vecs[i], v_sent))
                        
                        # MaxSim for this specific significant word
                        token_sims = np.dot(c_tokens_norm, norm_sig_vecs[i])
                        word_to_token_sim = float(np.max(token_sims))
                        
                        best_word_sim = max(word_to_sent_sim, word_to_token_sim)
                        
                        if best_word_sim >= threshold:
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
                # Simplified scoring for single word to ensure semantic results show up
                # Give high weight to sem_score if lex_score is low
                if lex_score > 0.5:
                    final_score = (lex_score * 0.6) + (sem_score * 0.4)
                else:
                    # Pure semantic focus for synonyms/expansion
                    # Scale up sem_score to be in a similar range as lexical matches (0.4 - 0.8)
                    # A raw sem_score of 0.4 should result in a final_score around 0.5
                    final_score = sem_score * 1.5
            else:
                lam_lex, lam_sem = self._compute_dynamic_weights(text_len, query_sig_len)
                final_score = self._calculate_clear_score(sem_score, lex_score, lam_lex, lam_sem, text_len)
            
            # Safety clamp and noise floor
            final_score = max(0.0, min(1.0, final_score))
            
            # Increased noise floor for multi-word queries to prune unrelated results
            # Lowered for single word to ensure we see semantic results
            min_threshold = 0.45 if not is_single_word else 0.01
            
            if query in ['paglilitis', 'kababata'] and idx < 5:
                print(f"DEBUG: Candidate '{sent.sentence_text[:30]}' Final: {final_score:.4f}, Threshold: {min_threshold}")

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
            if not lst:
                return []
            
            top_score = lst[0]['scores']['final']
            print(f"DEBUG: Top score for {label}: {top_score}%")
            
            # Normalization logic: If top score is very low (e.g. for pure semantic expanded results),
            # we scale it up for the UI to be more meaningful (e.g. 80-90%).
            # But we keep original relative ranking.
            if top_score > 0 and top_score < 40:
                scale_factor = 85.0 / top_score
                for itm in lst:
                    itm['scores']['final'] = min(100, round(itm['scores']['final'] * scale_factor))
            
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
