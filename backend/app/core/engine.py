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

    def search(self, db: Session, query: str, top_k: int = 10):
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
        
        for sent in candidates:
            # Re-compute cosine similarity (1 - distance) locally or rely on order?
            # We need the actual score for the formula.
            # Cosine Sim = dot product if normalized? SentenceTransformer output is normalized.
            # Let's compute it manually to be safe and get exact float
            sem_score = float(np.dot(query_embedding, np.array(sent.embedding)))
            # Clamp between 0 and 1
            sem_score = max(0.0, min(1.0, sem_score))
            
            # Lexical Score
            lex_score = self._compute_lexical_score(query, sent.sentence_text, query_analysis, stopword_ratio)
            
            # Dynamic Weights
            text_len = len(sent.sentence_text.split())
            lambda_lex, lambda_sem = self._compute_dynamic_weights(text_len)
            
            # Final Score
            final_score = self._calculate_clear_score(sem_score, lex_score, lambda_lex, lambda_sem, text_len)
            
            result_item = {
                'id': sent.id,
                'chapter_number': sent.chapter_number,
                'chapter_title': sent.chapter_title,
                'sentence_text': sent.sentence_text,
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
        results['noli'] = sorted(results['noli'], key=lambda x: x['scores']['final'], reverse=True)[:top_k]
        results['elfili'] = sorted(results['elfili'], key=lambda x: x['scores']['final'], reverse=True)[:top_k]
        
        return results

    def _compute_lexical_score(self, query, sentence_text, query_analysis, stopword_ratio):
        sentence_lower = sentence_text.lower()
        sentence_words = set(extract_words(sentence_lower))
        
        matched_weight = 0.0
        total_weight = sum(item['semantic_weight'] for item in query_analysis)
        
        if total_weight == 0:
            return 0.0
            
        for item in query_analysis:
            if item['word'].lower() in sentence_words:
                matched_weight += item['semantic_weight']
        
        score = matched_weight / total_weight
        
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

    def _classify_themes(self, db: Session, sentence_embedding, sentence_text):
        if not hasattr(self, 'theme_cache'):
            # Lazy load themes
            from app.models.database import Theme
            themes = db.query(Theme).all()
            self.theme_cache = [
                {
                    'id': t.id,
                    'tagalog_title': t.tagalog_title,
                    'meaning': t.meaning,
                    'embedding': np.array(t.embedding),
                    'meaning_len': len(t.meaning.split())
                }
                for t in themes
            ]
        
        matches = []
        sent_vec = np.array(sentence_embedding)
        sent_len = len(sentence_text.split())

        for theme in self.theme_cache:
            # Semantic Sim
            sem_sim = float(np.dot(sent_vec, theme['embedding']))
            sem_sim = max(0.0, min(1.0, sem_sim))
            
            # Use same query analysis logic for text? Or just word overlap?
            # vbest.py uses simple lexical overlap:
            lex_score = self._compute_simple_lexical(sentence_text, theme['meaning'])
            
            # Dynamic weights for themes
            lambda_lex, lambda_sem = self._compute_dynamic_weights_by_length(theme['meaning_len'], sent_len)
            
            theme_score = (lambda_sem * sem_sim) + (lambda_lex * lex_score)
            
            if theme_score >= 0.70: # THEMATIC_THRESHOLD form vbest
                matches.append({
                    'id': str(theme['id']),
                    'label': theme['tagalog_title'],
                    'score': theme_score
                })
        
        # Sort and take top 2
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:2]

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

@lru_cache()
def get_engine():
    return RizalEngine()
