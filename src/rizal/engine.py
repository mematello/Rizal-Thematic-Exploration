"""
Core search engine logic (Headless).
Returns pure data, no visualizations or print statements.
"""
import numpy as np
import re
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from .config import (
    MIN_SEMANTIC_THRESHOLD, THEMATIC_THRESHOLD, NEIGHBOR_RELEVANCE_THRESHOLD,
    SHORT_SENTENCE_THRESHOLD, SHORT_SENTENCE_PENALTY, MAX_CONTEXT_EXPANSION,
    HIGH_STOPWORD_RATIO, STOPWORD_PENALTY_FACTOR, DOMAIN_COHERENCE_THRESHOLD,
    DOMAIN_MIN_WORDS, DOMAIN_OUTLIER_DELTA, SEMANTIC_SIMILARITY_THRESHOLD,
    HIGH_SEMANTIC_SIMILARITY_THRESHOLD, MIN_COOCCURRENCE_NORMAL, MIN_COOCCURRENCE_STRICT,
    RELATION_SIM_THRESHOLD, RELATION_COOCC_THRESHOLD, RELATION_COOCC_THRESHOLD_NAMED
)
from .utils import extract_words, sanitize_text, shorten_sentence, extract_relations_regex
from .query_analyzer import QueryAnalyzer
from .loader import DataLoader
from .errors import TopicNotFoundError

class RizalEngine:
    """
    Dynamic Dual-Formula CLEAR System Engine.
    """
    def __init__(self, data_loader: DataLoader):
        self.data = data_loader
        self.query_analyzer = QueryAnalyzer()
        # Shortcut references to data loaded
        self.books_data = self.data.books_data
        self.model = self.data.model

    def query(self, user_query):
        """Main query interface. Returns a standardized result dictionary."""
        # Phase 1: Query Validation
        is_valid, validation_info = self.query_analyzer.validate_filipino_query(user_query)

        if not is_valid:
            recovery = self.generate_recovery_suggestions(
                user_query, failure_context={'stage': 'filipino_validation', 'details': validation_info}
            )
            return {
                'status': 'error',
                'error_type': 'invalid_filipino',
                'message': f"Invalid Filipino query: {validation_info['reason']}",
                'validation_info': validation_info,
                'suggestions': recovery,
                'raw_results': None
            }

        content_words = self.query_analyzer.get_content_words(user_query)
        if not content_words:
            recovery = self.generate_recovery_suggestions(user_query, failure_context={'stage': 'stopword_only'})
            return {
                'status': 'error',
                'error_type': 'no_lexical_grounding',
                'message': "Query contains only stopwords",
                'overlap_info': {'reason': 'Stopwords only'},
                'suggestions': recovery,
                'raw_results': None
            }

        # Phase 2: Lexical Presence Check
        missing_words = [w for w in content_words if w not in self.data.global_vocabulary]
        if missing_words:
            recovery = self.generate_recovery_suggestions(
                user_query, failure_context={'stage': 'lexical_presence', 'missing_words': missing_words}
            )
            return {
                'status': 'error',
                'error_type': 'no_lexical_grounding',
                'message': "Words not found in corpus",
                'overlap_info': {
                    'missing_words': missing_words,
                    'content_words': content_words
                },
                'suggestions': recovery,
                'raw_results': None
            }

        # Phase 2.5: Semantic Query Validation
        semantic_validation = self._validate_semantic_query(content_words)
        if not semantic_validation['proceed']:
            recovery = self.generate_recovery_suggestions(
                user_query, failure_context={'stage': 'semantic_validation', 'details': semantic_validation}
            )
            return {
                'status': 'error',
                'error_type': 'semantic_validation_failed',
                'message': semantic_validation['reason'],
                'details': semantic_validation,
                'suggestions': recovery,
                'raw_results': None
            }

        # Phase 2.6: Domain Coherence
        if len(content_words) >= DOMAIN_MIN_WORDS:
            sim_info = self._compute_domain_coherence(content_words)
            outliers = [w for w, avg in sim_info['per_word_avg'].items() if avg + DOMAIN_OUTLIER_DELTA < sim_info['overall_avg']]
            
            if sim_info['overall_avg'] < DOMAIN_COHERENCE_THRESHOLD or outliers:
                 recovery = self.generate_recovery_suggestions(
                    user_query, failure_context={'stage': 'domain_coherence', 'details': sim_info}
                )
                 return {
                    'status': 'error',
                    'error_type': 'domain_incoherent',
                    'message': "Query words are not semantically coherent",
                    'details': {
                        'overall_avg': sim_info['overall_avg'],
                        'outliers': outliers
                    },
                    'suggestions': recovery,
                    'raw_results': None
                }

        # Phase 5: Semantic Grounding (Embedding check)
        query_embedding = self.model.encode([user_query])[0]
        grounding_ok, grounding_reason = self._validate_semantic_grounding(query_embedding)
        if not grounding_ok:
             recovery = self.generate_recovery_suggestions(
                user_query, failure_context={'stage': 'semantic_grounding'}, top_k=3, query_embedding=query_embedding
            )
             return {
                'status': 'error',
                'error_type': 'semantic_grounding_rejected',
                'message': grounding_reason,
                'suggestions': recovery,
                'raw_results': None
            }

        # Execution
        query_analysis = self.query_analyzer.analyze_query_words(user_query)
        results_by_book = {}
        
        for book_key in self.books_data.keys():
            vocab = self.data.corpus_vocabulary[book_key]
            if not any(w in vocab for w in content_words):
                continue
                
            passages = self._retrieve_passages(user_query, query_analysis, book_key, query_embedding=query_embedding)
            
            if passages:
                thematic_passages, has_themes, avg_theme_conf = self._get_thematic_classification(passages, user_query, book_key)
                
                # Calculate aggregate metrics
                avg_sem = np.mean([p['semantic_score'] for p in passages]) if passages else 0
                avg_lex = np.mean([p['lexical_score'] for p in passages]) if passages else 0
                avg_final = np.mean([p['final_score'] for p in passages]) if passages else 0
                
                results_by_book[book_key] = {
                    'results': thematic_passages,
                    'has_themes': has_themes,
                    'metrics': {
                        'avg_semantic': avg_sem,
                        'avg_lexical': avg_lex,
                        'avg_final': avg_final,
                        'avg_theme_conf': avg_theme_conf
                    }
                }

        if not results_by_book:
             recovery = self.generate_recovery_suggestions(
                user_query, failure_context={'stage': 'no_results'}, top_k=3, query_embedding=query_embedding
            )
             return {
                'status': 'empty',
                'message': "No matches found",
                'suggestions': recovery,
                'raw_results': None
            }

        next_queries = self.generate_followup_suggestions(user_query, results_by_book)
        
        return {
            'status': 'success',
            'results_by_book': results_by_book,
            'query_analysis': query_analysis,
            'suggestions': [],
            'next_queries': next_queries,
            'raw_results': results_by_book # Redundant but keeps schema consistent
        }

    # --- Core Algorithms (Private) ---

    def _retrieve_passages(self, query, query_analysis, book_key, top_k=9, query_embedding=None):
        """CLEAR-based hybrid retrieval."""
        self.data.used_passages[book_key] = set() # Reset for this query
        
        book_data = self.books_data[book_key]
        chapters_df = book_data['chapters']
        embeddings = book_data['embeddings']
        
        if query_embedding is None:
            query_embedding_vec = self.model.encode([query]).reshape(1, -1)
        else:
            query_embedding_vec = np.asarray(query_embedding).reshape(1, -1)

        semantic_similarities = cosine_similarity(query_embedding_vec, embeddings)[0]
        candidates = []

        for idx, semantic_sim in enumerate(semantic_similarities):
            if semantic_sim < MIN_SEMANTIC_THRESHOLD:
                continue
                
            row = chapters_df.iloc[idx]
            sentence_text = row['sentence_text']
            sentence_length = len(sentence_text.split())
            
            lambda_lex, lambda_sem = self._compute_dynamic_weights_by_length(sentence_length)
            lexical_score = self._compute_lexical_score_weighted(query, sentence_text, query_analysis)
            
            if lexical_score >= 0.95: match_type = 'exact'
            elif lexical_score >= 0.3: match_type = 'partial_lexical'
            else: match_type = 'semantic'
            
            final_score = self._calculate_clear_score(semantic_sim, lexical_score, lambda_lex, lambda_sem, sentence_length)
            
            candidates.append({
                'chapter_number': row['chapter_number'],
                'chapter_title': row['chapter_title'],
                'sentence_number': row['sentence_number'],
                'sentence_text': sentence_text,
                'semantic_score': semantic_sim,
                'lexical_score': lexical_score,
                'final_score': final_score,
                'match_type': match_type,
                'word_count': sentence_length,
                'lambda_lex': lambda_lex,
                'lambda_sem': lambda_sem
            })

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Deduplication and Context Expansion
        results = []
        chapter_counts = {}
        
        for candidate in candidates:
            # Check usage
            pass_id = self.data.get_passage_id(candidate['chapter_number'], candidate['sentence_number'])
            if pass_id in self.data.used_passages[book_key]: continue
            
            # Check diversity
            ch_num = candidate['chapter_number']
            if chapter_counts.get(ch_num, 0) >= 3: continue
            
            self.data.used_passages[book_key].add(pass_id)
            
            # Get Context
            context = self._get_expanded_context(
                candidate['chapter_number'], candidate['sentence_number'], 
                query, book_key, candidate['sentence_text'], candidate['word_count']
            )
            candidate['context'] = context
            candidate['has_relevant_context'] = (context['prev_relevant_count'] > 0 or context['next_relevant_count'] > 0)
            
            # Mark context used
            for sent in context['prev_sentences'] + context['next_sentences']:
                self.data.used_passages[book_key].add(self.data.get_passage_id(ch_num, sent['sentence_number']))
            
            results.append(candidate)
            chapter_counts[ch_num] = chapter_counts.get(ch_num, 0) + 1
            if len(results) >= top_k: break
            
        return results

    def _get_expanded_context(self, chapter_num, sentence_num, query, book_key, main_text, main_length):
        """Get neighbor context."""
        context = {'prev_sentences': [], 'next_sentences': [], 'prev_relevant_count': 0, 'next_relevant_count': 0}
        chapters_df = self.books_data[book_key]['chapters']
        chapter_sentences = chapters_df[chapters_df['chapter_number'] == chapter_num].sort_values('sentence_number')
        
        # Find current index in list
        try:
             # Fast lookup if possible, otherwise iterate
             current_row = chapter_sentences[chapter_sentences['sentence_number'] == sentence_num]
             if current_row.empty: return context
             current_idx = current_row.index[0]
             
             chapter_list = chapter_sentences.index.tolist()
             current_pos = chapter_list.index(current_idx)
        except:
            return context

        # Helper to process neighbor
        def process_neighbor(idx, direction_list, count_key, distance):
            row = chapters_df.loc[idx]
            pass_id = self.data.get_passage_id(row['chapter_number'], row['sentence_number'])
            if pass_id in self.data.used_passages[book_key]: return False # Stop expansion
             
            text = row['sentence_text']
            length = len(text.split())
            sem_sim, lex_sim, neigh_score, l_lex, l_sem = self._compute_neighbor_similarity(main_text, main_length, text, length)
            is_relevant = neigh_score >= NEIGHBOR_RELEVANCE_THRESHOLD
            
            item = {
                'sentence_number': row['sentence_number'],
                'sentence_text': text,
                'is_relevant': is_relevant,
                'distance': distance,
                'neighbor_score': neigh_score,
                'semantic_similarity': sem_sim,
                'lexical_similarity': lex_sim,
                'lambda_lex': l_lex,
                'lambda_sem': l_sem
            }
            direction_list.append(item)
            if is_relevant: context[count_key] += 1
            return is_relevant # Continue only if relevant? Logic in vbest says 'break' if not relevant?
            # Review vbest logic: "if is_relevant: count++ else: break"
        
        # Expand Backward
        for i in range(1, MAX_CONTEXT_EXPANSION + 1):
            if current_pos - i < 0: break
            prev_idx = chapter_list[current_pos - i]
            
            # Replicating logic structure: check existing used FIRST
            prev_row = chapters_df.loc[prev_idx]
            if self.data.get_passage_id(prev_row['chapter_number'], prev_row['sentence_number']) in self.data.used_passages[book_key]: break
            
            text = prev_row['sentence_text']
            sem_sim, lex_sim, neigh_score, l_lex, l_sem = self._compute_neighbor_similarity(main_text, main_length, text, len(text.split()))
            is_relevant = neigh_score >= NEIGHBOR_RELEVANCE_THRESHOLD
            
            context['prev_sentences'].append({
                'sentence_number': prev_row['sentence_number'],
                'sentence_text': text,
                'is_relevant': is_relevant,
                'distance': i,
                'score': neigh_score, # Simplified key name
                'neighbor_score': neigh_score,
                'semantic_similarity': sem_sim,
                'lexical_similarity': lex_sim,
                'lambda_lex': l_lex,
                'lambda_sem': l_sem
            })
            if is_relevant: context['prev_relevant_count'] += 1
            else: break
        context['prev_sentences'].reverse()
        
        # Expand Forward (Similar logic)
        for i in range(1, MAX_CONTEXT_EXPANSION + 1):
            if current_pos + i >= len(chapter_list): break
            next_idx = chapter_list[current_pos + i]
            
            next_row = chapters_df.loc[next_idx]
            if self.data.get_passage_id(next_row['chapter_number'], next_row['sentence_number']) in self.data.used_passages[book_key]: break
            
            text = next_row['sentence_text']
            sem_sim, lex_sim, neigh_score, l_lex, l_sem = self._compute_neighbor_similarity(main_text, main_length, text, len(text.split()))
            is_relevant = neigh_score >= NEIGHBOR_RELEVANCE_THRESHOLD
            
            context['next_sentences'].append({
                'sentence_number': next_row['sentence_number'],
                'sentence_text': text,
                'is_relevant': is_relevant,
                'distance': i,
                'neighbor_score': neigh_score,
                'semantic_similarity': sem_sim,
                'lexical_similarity': lex_sim,
                'lambda_lex': l_lex,
                'lambda_sem': l_sem
            })
            if is_relevant: context['next_relevant_count'] += 1
            else: break
            
        return context

    # --- Scoring & Helpers ---

    def _compute_dynamic_weights_by_length(self, text_length, reference_length=None):
        """Proxy to DataLoader logic or reimplement if plain function."""
        # Use simple function logic for speed
        if reference_length is None:
            if text_length <= 5: return 0.75, 0.25
            elif text_length <= 10: return 0.65, 0.35
            elif text_length <= 15: return 0.55, 0.45
            elif text_length <= 20: return 0.45, 0.55
            else: return 0.35, 0.65
        else:
            length_ratio = text_length / max(reference_length, 1)
            if length_ratio >= 1.5: return 0.30, 0.70
            elif length_ratio >= 1.2: return 0.40, 0.60
            elif length_ratio >= 0.8: return 0.50, 0.50
            elif length_ratio >= 0.5: return 0.60, 0.40
            else: return 0.70, 0.30

    def _compute_lexical_score_weighted(self, query, sentence_text, query_analysis):
        """Weighted lexical overlap."""
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        if query_lower == sentence_lower: return 1.0
        
        # Exact substring match check
        if re.search(r'\b' + re.escape(query_lower) + r'\b', sentence_lower):
            return min(1.0, len(query_lower) / len(sentence_lower) * 2)

        # Token match
        q_weights = {item['word'].lower(): item['semantic_weight'] for item in query_analysis}
        s_words = set(extract_words(sentence_lower))
        
        total_weight = sum(q_weights.values())
        if total_weight == 0: return 0.0
        
        matched_weight = sum(w for w, w_val in q_weights.items() if w in s_words)
        score = matched_weight / total_weight
        
        # Penalty
        stop_ratio = self.query_analyzer.get_stopword_ratio(query)
        if stop_ratio > HIGH_STOPWORD_RATIO:
            penalty = (stop_ratio - HIGH_STOPWORD_RATIO) * STOPWORD_PENALTY_FACTOR
            score *= (1.0 - penalty)
        return score

    def _compute_lexical_score_simple(self, text1, text2):
        w1 = set(extract_words(text1.lower()))
        w2 = set(extract_words(text2.lower()))
        if not w1 or not w2: return 0.0
        return len(w1 & w2) / len(w1 | w2)

    def _calculate_clear_score(self, sem_sim, lex_sim, lambda_lex, lambda_sem, word_count=None):
        score = (lambda_sem * sem_sim) + (lambda_lex * lex_sim)
        if word_count and word_count < SHORT_SENTENCE_THRESHOLD:
            penalty = SHORT_SENTENCE_PENALTY * (SHORT_SENTENCE_THRESHOLD - word_count) / SHORT_SENTENCE_THRESHOLD
            score -= penalty
        return max(0.0, min(1.0, score))

    def _compute_neighbor_similarity(self, main_text, main_len, neighbor_text, neighbor_len):
        l_lex, l_sem = self._compute_dynamic_weights_by_length(neighbor_len, main_len)
        
        sem_sim = cosine_similarity(self.model.encode([main_text]), self.model.encode([neighbor_text]))[0][0]
        lex_sim = self._compute_lexical_score_simple(main_text, neighbor_text)
        
        combined = self._calculate_clear_score(sem_sim, lex_sim, l_lex, l_sem)
        return float(sem_sim), float(lex_sim), float(combined), float(l_lex), float(l_sem)

    def _get_thematic_classification(self, passages, query, book_key):
        """Attach themes to passages."""
        if not passages: return passages, False, 0.0
        themes_df = self.books_data[book_key]['themes']
        results = []
        
        for p in passages:
            s_text = p['sentence_text']
            s_vec = self.model.encode([s_text])
            s_len = len(s_text.split())
            
            matches = []
            for idx, row in themes_df.iterrows():
                meaning = row['Meaning']
                m_len = len(meaning.split())
                l_lex, l_sem = self._compute_dynamic_weights_by_length(m_len, s_len)
                
                m_vec = self.model.encode([meaning])
                sem_sim = cosine_similarity(s_vec, m_vec)[0][0]
                lex_sim = self._compute_lexical_score_simple(s_text, meaning)
                score = self._calculate_clear_score(sem_sim, lex_sim, l_lex, l_sem)
                
                if score >= THEMATIC_THRESHOLD:
                    matches.append({
                        'tagalog_title': row['Tagalog Title'],
                        'meaning': meaning,
                        'confidence': float(score),
                        'lambda_lex': float(l_lex),
                        'lambda_sem': float(l_sem)
                    })
            
            matches.sort(key=lambda x: x['confidence'], reverse=True)
            p_new = p.copy()
            p_new['themes'] = matches[:2]
            p_new['primary_theme'] = matches[0] if matches else None
            p_new['has_theme'] = bool(matches)
            results.append(p_new)
            
        # Summary stats
        has_t_count = sum(1 for p in results if p['has_theme'])
        coverage = has_t_count / len(results)
        avg_conf = np.mean([p['primary_theme']['confidence'] for p in results if p['has_theme']]) if has_t_count else 0.0
        
        global_has = coverage >= 0.3 and avg_conf >= THEMATIC_THRESHOLD
        return results, global_has, avg_conf

    def _analyze_relations(self, query, content_words):
        """Analyze conceptual relations in query."""
        relations = extract_relations_regex(query)
        if not relations:
            return {'passed': True, 'relations': []}

        # Check relation coherence
        valid_relations = []
        scores = []
        
        # We need coherence matrix/stats to validate.
        # This relies on domain coherence logic.
        sim_info = self._compute_domain_coherence(content_words)
        
        for rel in relations:
            left, right = rel['left'], rel['right']
            if left in sim_info['per_word_avg'] and right in sim_info['per_word_avg']:
                # Simple check: are they coherent with each other?
                # We can't access sim matrix easily from here without re-computing or changing return of compute_domain_coherence
                # Let's re-compute pair sim
                emb = self.model.encode([left, right])
                sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
                scores.append({'relation': rel['span'], 'score': float(sim)})
                
                if sim >= RELATION_SIM_THRESHOLD:
                    valid_relations.append(rel)
        
        # If we found relations but they all failed threshold?
        # vbest logic was a bit more complex, involving plots.
        # Here we just return status.
        return {
            'passed': True, # We are permissive by default in headless mode unless strict check requested
            'relations': relations,
            'coherence': sim_info,
            'relation_scores': scores
        }
    
    def _validate_semantic_query(self, content_words):
        """Simplified semantic validator."""
        if len(content_words) < 2: return {'proceed': True, 'reason': 'Single word'}
        
        tokens = [w.lower() for w in content_words]
        embeddings = self.model.encode(tokens)
        sim_matrix = cosine_similarity(embeddings)
        
        # Avg Sim
        triu = np.triu_indices(len(tokens), k=1)
        avg_sim = float(np.mean(sim_matrix[triu])) if triu[0].size > 0 else 0.0
        
        # Co-occurrence
        min_coocc = float('inf')
        word_pairs = []
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                w1, w2 = tokens[i], tokens[j]
                coocc = self.data.build_co_occurrence([f"{w1} {w2}"], [w1, w2]).values.sum() # Simplified coocc check via data loader logic? 
                # Wait, data.build_co_occurrence expects sentences.
                # We need raw co-occurrence count. vbest used `_count_cooccurrence` which iterates all sentences.
                # This is expensive to re-implement here without access to raw sentences easily? 
                # `DataLoader` has `global_passages`.
                # Let's use a simpler check or move `_count_cooccurrence` to `DataLoader` or `RizalEngine`
                coocc = self._count_cooccurrence(w1, w2) 
                
                if coocc < min_coocc: min_coocc = coocc
                word_pairs.append({'word1': w1, 'word2': w2, 'similarity': float(sim_matrix[i][j]), 'cooccurrence': coocc})
        
        if min_coocc == float('inf'): min_coocc = 0

        # Rules
        if avg_sim < SEMANTIC_SIMILARITY_THRESHOLD:
            return {'proceed': False, 'reason': f'Low semantic similarity ({avg_sim:.2f})', 'details': word_pairs}
            
        if min_coocc == 0:
            if avg_sim >= HIGH_SEMANTIC_SIMILARITY_THRESHOLD:
                return {'proceed': True, 'reason': 'High similarity override'}
            else:
                return {'proceed': False, 'reason': 'Zero co-occurrence'}
        
        req = MIN_COOCCURRENCE_STRICT if avg_sim < 0.5 else MIN_COOCCURRENCE_NORMAL
        if min_coocc < req:
            return {'proceed': False, 'reason': f'Low co-occurrence ({min_coocc})'}
            
        return {'proceed': True, 'reason': 'OK'}

    def _count_cooccurrence(self, w1, w2):
        """Count how many sentences contain both words."""
        # This is slow, but vbest.py did it this way.
        count = 0
        w1_re = re.compile(rf"\b{re.escape(w1)}\b", re.IGNORECASE)
        w2_re = re.compile(rf"\b{re.escape(w2)}\b", re.IGNORECASE)
        
        for p in self.data.global_passages:
            text = p['sentence_text']
            if w1_re.search(text) and w2_re.search(text):
                count += 1
        return count

    def _compute_domain_coherence(self, words):
        """Compute coherence metrics."""
        tokens = [w.lower() for w in words]
        emb = self.model.encode(tokens)
        sim = cosine_similarity(emb)
        
        per_word = {}
        for i, w in enumerate(tokens):
            row = np.delete(sim[i], i)
            per_word[w] = float(np.mean(row)) if row.size else 1.0
            
        triu = np.triu_indices(len(tokens), k=1)
        overall = float(np.mean(sim[triu])) if triu[0].size else 1.0
        
        return {'per_word_avg': per_word, 'overall_avg': overall}

    def _validate_semantic_grounding(self, query_vec):
        """Check if query aligns with corpus."""
        if not self.data.ready: return True, None
        
        qv = query_vec.reshape(1, -1)
        max_passage = float(np.max(cosine_similarity(qv, self.data.passage_embeddings_matrix)))
        if max_passage < 0.20: return False, "No semantically similar passages"
        
        if self.data.theme_embeddings_matrix.shape[0] > 0:
            max_theme = float(np.max(cosine_similarity(qv, self.data.theme_embeddings_matrix)))
            if max_theme < 0.30: return False, "Not aligned with themes"
            
        return True, None

    # --- Suggestions ---
    
    def generate_recovery_suggestions(self, query, failure_context=None, top_k=3, query_embedding=None):
        """Generate alternative queries."""
        # Simple placeholder logic replicating vbest intent
        if not self.data.ready: return []
        
        if query_embedding is None:
            query_embedding = self.model.encode([query])[0]
            
        # Get nearest passages
        qv = query_embedding.reshape(1, -1)
        sims = cosine_similarity(qv, self.data.passage_embeddings_matrix)[0]
        top_idx = np.argsort(sims)[::-1][:20]
        
        suggestions = []
        seen = set()
        
        for idx in top_idx:
            p = self.data.global_passages[idx]
            s = shorten_sentence(p['sentence_text'], 14)
            if s and s not in seen and s.lower() != query.lower():
                suggestions.append(s)
                seen.add(s)
            if len(suggestions) >= top_k: break
            
        return suggestions

    def generate_followup_suggestions(self, query, results_by_book, top_k=3):
        """Generate next steps."""
        cands = []
        for b_res in results_by_book.values():
            for p in b_res['results']:
                if p.get('primary_theme'):
                    t = p['primary_theme']['tagalog_title']
                    if t and t not in cands: cands.append(t)
                # Entity extraction could go here
        return cands[:top_k]
