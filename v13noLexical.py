import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

warnings.filterwarnings('ignore')

class QueryAnalyzer:
    """Handles query validation and linguistic analysis with official Tagalog stopwords"""
    
    def __init__(self):
        self.MIN_FILIPINO_FREQUENCY = 1e-8
        self.MIN_VALID_WORD_RATIO = 0.5
        
        # Load official Tagalog stopwords from stopwords-iso
        self.STOPWORDS = self._load_official_stopwords()
    
    def _load_official_stopwords(self):
        """Load official Tagalog stopwords from stopwords-iso package"""
        try:
            from stopwordsiso import stopwords
            tagalog_stops = set(stopwords('tl'))
            print(f"Loaded {len(tagalog_stops)} official Tagalog stopwords")
            return tagalog_stops
        except ImportError:
            print("Warning: stopwords-iso not installed. Install with: pip install stopwords-iso")
            print("Falling back to minimal stopword set")
            # Minimal fallback if package not available
            return {'ng', 'sa', 'ang', 'na', 'ay', 'at', 'mga'}
    
    def get_word_frequency(self, word, lang='tl'):
        """Get word frequency using wordfreq library"""
        try:
            from wordfreq import word_frequency
            return word_frequency(word.lower(), lang)
        except Exception:
            return 0.0
    
    def is_valid_filipino_word(self, word):
        """Check if a word is valid Filipino"""
        if len(word) < 2:
            return False
        
        freq = self.get_word_frequency(word, 'tl')
        return freq >= self.MIN_FILIPINO_FREQUENCY
    
    def validate_filipino_query(self, query):
        """Validate if query contains valid Filipino words"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        
        if not words:
            return False, {
                'reason': 'No valid words found in query',
                'total_words': 0,
                'valid_words': 0,
                'invalid_words': [],
                'valid_ratio': 0.0
            }
        
        valid_words = []
        invalid_words = []
        
        for word in words:
            if self.is_valid_filipino_word(word):
                valid_words.append(word)
            else:
                invalid_words.append(word)
        
        total_words = len(words)
        valid_count = len(valid_words)
        valid_ratio = valid_count / total_words if total_words > 0 else 0.0
        is_valid = valid_ratio >= self.MIN_VALID_WORD_RATIO
        
        validation_info = {
            'total_words': total_words,
            'valid_words': valid_count,
            'invalid_words': invalid_words,
            'valid_words_list': valid_words,
            'valid_ratio': valid_ratio,
            'reason': ''
        }
        
        if not is_valid:
            if valid_count == 0:
                validation_info['reason'] = 'No valid Filipino words detected'
            else:
                validation_info['reason'] = f'Only {valid_ratio:.1%} of words are valid Filipino'
        
        return is_valid, validation_info
    
    def analyze_query_words(self, query):
        """Analyze query words: frequencies, stopword status, and semantic weight"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        analysis = []
        
        for word in words:
            word_lower = word.lower()
            freq = self.get_word_frequency(word_lower, 'tl')
            is_stopword = word_lower in self.STOPWORDS
            
            # Determine semantic weight
            if is_stopword:
                semantic_weight = 0.05  # Very low weight for stopwords
            elif freq > 0.001:
                semantic_weight = 0.3   # Low weight for very common words
            elif freq > 0.0001:
                semantic_weight = 0.7   # Medium weight
            else:
                semantic_weight = 1.0   # Full weight for rare/content words
            
            analysis.append({
                'word': word,
                'frequency': freq,
                'is_stopword': is_stopword,
                'is_content_word': not is_stopword and freq < 0.001,
                'semantic_weight': semantic_weight
            })
        
        return analysis
    
    def get_stopword_ratio(self, query):
        """Calculate the ratio of stopwords in query"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        if not words:
            return 0.0
        
        stopword_count = sum(1 for w in words if w.lower() in self.STOPWORDS)
        return stopword_count / len(words)
    
    def get_content_words(self, query):
        """Extract non-stopword content words from query"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        content_words = [w.lower() for w in words if w.lower() not in self.STOPWORDS]
        return content_words

class CleanNoliSystem:
    """
    Clean CLEAR-inspired hybrid retrieval supporting multiple books
    Formula: s_final(q,d) = λ_emb · s_emb(q,d) + λ_lex · s_lex_weighted(q,d)
    
    Supports both Noli Me Tangere and El Filibusterismo with separate indexing
    Includes lexical grounding check to block queries with zero corpus overlap
    """
    
    def __init__(self):
        self.console = Console()
        
        self.console.print("Loading XLM-RoBERTa model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        self.console.print("Loading datasets...", style="cyan")
        self.books_data = {}
        self.used_passages = {}
        self.corpus_vocabulary = {}  # Store corpus vocabulary per book
        self._load_books()
        
        self.console.print("Initializing query analyzer with official stopwords...", style="cyan")
        self.query_analyzer = QueryAnalyzer()
        
        self.console.print("Computing embeddings for all books...", style="cyan")
        self._compute_all_embeddings()
        
        self.console.print("Building corpus vocabulary...", style="cyan")
        self._build_corpus_vocabulary()
        
        # System parameters
        self.MIN_SEMANTIC_THRESHOLD = 0.20
        self.THEMATIC_THRESHOLD = 0.45
        self.CONTEXT_RELEVANCE_THRESHOLD = 0.30
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.08
        self.MAX_CONTEXT_EXPANSION = 5
        
        # Stopword penalty thresholds
        self.HIGH_STOPWORD_RATIO = 0.6
        self.STOPWORD_PENALTY_FACTOR = 0.5
        
        self.console.print("Clean CLEAR multi-book system with lexical grounding ready!", style="bold green")
    
    def _load_books(self):
        """Load data for both Noli Me Tangere and El Filibusterismo"""
        books = [
            ('noli', 'noli_chapters.csv', 'noli_themes.csv'),
            ('elfili', 'elfili_chapters.csv', 'elfili_themes.csv')
        ]
        
        for book_key, chapters_file, themes_file in books:
            try:
                chapters_df = pd.read_csv(chapters_file)
                themes_df = pd.read_csv(themes_file)
                
                chapters_df.columns = chapters_df.columns.str.strip()
                themes_df.columns = themes_df.columns.str.strip()
                
                self.books_data[book_key] = {
                    'chapters': chapters_df,
                    'themes': themes_df,
                    'embeddings': None,
                    'theme_embeddings': None
                }
                
                self.used_passages[book_key] = set()
                
                self.console.print(f"  Loaded {book_key}: {len(chapters_df)} chapters, {len(themes_df)} themes", style="green")
            except FileNotFoundError:
                self.console.print(f"  Warning: {chapters_file} or {themes_file} not found", style="yellow")
    
    def _compute_all_embeddings(self):
        """Compute embeddings for all books"""
        for book_key, book_data in self.books_data.items():
            chapters_df = book_data['chapters']
            themes_df = book_data['themes']
            
            # Compute chapter embeddings
            chapters_df['combined_text'] = (
                chapters_df['chapter_title'].astype(str) + " " + 
                chapters_df['sentence_text'].astype(str)
            )
            
            chapters_df['sentence_word_count'] = (
                chapters_df['sentence_text'].astype(str).apply(lambda x: len(x.split()))
            )
            
            texts = chapters_df['combined_text'].tolist()
            book_data['embeddings'] = self.model.encode(texts, show_progress_bar=False)
            
            # Compute theme embeddings
            themes_df['theme_text'] = (
                themes_df['Tagalog Title'].astype(str) + " " + 
                themes_df['Meaning'].astype(str)
            )
            
            theme_texts = themes_df['theme_text'].tolist()
            book_data['theme_embeddings'] = self.model.encode(theme_texts, show_progress_bar=False)
    
    def _build_corpus_vocabulary(self):
        """Build vocabulary from corpus for lexical grounding check"""
        for book_key, book_data in self.books_data.items():
            chapters_df = book_data['chapters']
            vocabulary = set()
            
            for text in chapters_df['sentence_text'].astype(str):
                words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', text.lower())
                vocabulary.update(words)
            
            # Remove stopwords from vocabulary
            vocabulary = vocabulary - self.query_analyzer.STOPWORDS
            
            self.corpus_vocabulary[book_key] = vocabulary
            self.console.print(f"  {book_key} corpus vocabulary: {len(vocabulary)} content words", style="green")
    
    def _check_lexical_grounding(self, query):
        """
        Check if query has any lexical overlap with corpus (excluding stopwords)
        Returns: (is_grounded, overlap_info_dict)
        """
        content_words = self.query_analyzer.get_content_words(query)
        
        if not content_words:
            return False, {
                'reason': 'Query contains only stopwords',
                'content_words': [],
                'matched_words': {},
                'total_content_words': 0,
                'total_matched': 0
            }
        
        matched_words = {}
        total_matched = 0
        
        for book_key, vocab in self.corpus_vocabulary.items():
            book_matches = [w for w in content_words if w in vocab]
            if book_matches:
                matched_words[book_key] = book_matches
                total_matched += len(book_matches)
        
        is_grounded = total_matched > 0
        
        overlap_info = {
            'content_words': content_words,
            'matched_words': matched_words,
            'total_content_words': len(content_words),
            'total_matched': total_matched,
            'reason': '' if is_grounded else 'No lexical overlap with corpus (zero content words match)'
        }
        
        return is_grounded, overlap_info
    
    def _get_passage_id(self, chapter_num, sentence_num):
        """Create unique identifier for passages"""
        return (int(chapter_num), int(sentence_num))
    
    def _compute_dynamic_weights(self, query_length, stopword_ratio):
        """Dynamic weighting with stopword awareness"""
        L = query_length
        
        if L <= 1:
            alpha, beta = 0.40, 0.60
        elif L == 2:
            alpha, beta = 0.55, 0.45
        elif L == 3:
            alpha, beta = 0.68, 0.32
        elif L == 4:
            alpha, beta = 0.75, 0.25
        elif L == 5:
            alpha, beta = 0.80, 0.20
        else:
            alpha, beta = 0.85, 0.15
        
        if stopword_ratio > self.HIGH_STOPWORD_RATIO:
            adjustment = (stopword_ratio - self.HIGH_STOPWORD_RATIO) * 0.5
            alpha = min(0.95, alpha + adjustment)
            beta = 1.0 - alpha
        
        return alpha, beta
    
    def _compute_lexical_score_weighted(self, query, sentence_text, query_analysis):
        """Compute weighted lexical overlap score with exact word boundary matching"""
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        
        if query_lower == sentence_lower:
            return 1.0
        
        # Check if query is a complete phrase match with word boundaries
        # Use word boundary regex to ensure exact matching
        query_pattern = r'\b' + re.escape(query_lower) + r'\b'
        if re.search(query_pattern, sentence_lower):
            return min(1.0, len(query_lower) / len(sentence_lower) * 2)
        
        query_words_data = {item['word'].lower(): item['semantic_weight'] 
                           for item in query_analysis}
        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        if not query_words_data:
            return 0.0
        
        total_weight = sum(query_words_data.values())
        matched_weight = sum(
            weight for word, weight in query_words_data.items() 
            if word in sentence_words
        )
        
        if total_weight == 0:
            return 0.0
        
        weighted_score = matched_weight / total_weight
        
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        if stopword_ratio > self.HIGH_STOPWORD_RATIO:
            penalty = (stopword_ratio - self.HIGH_STOPWORD_RATIO) * self.STOPWORD_PENALTY_FACTOR
            weighted_score *= (1.0 - penalty)
        
        return weighted_score
    
    def _calculate_clear_score(self, semantic_sim, lexical_score, weights, word_count):
        """CLEAR hybrid scoring"""
        lambda_emb, lambda_lex = weights
        
        final_score = (lambda_emb * semantic_sim) + (lambda_lex * lexical_score)
        
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (
                self.SHORT_SENTENCE_THRESHOLD - word_count
            ) / self.SHORT_SENTENCE_THRESHOLD
            final_score -= penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _check_context_relevance(self, context_text, query, book_key, theme_context=None):
        """Check if context sentence is relevant to query/theme"""
        try:
            query_embedding = self.model.encode([query])
            context_embedding = self.model.encode([context_text])
            query_similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
            
            theme_similarity = 0.0
            if theme_context:
                theme_embedding = self.model.encode([theme_context])
                theme_similarity = cosine_similarity(theme_embedding, context_embedding)[0][0]
            
            return max(query_similarity, theme_similarity) >= self.CONTEXT_RELEVANCE_THRESHOLD
        except:
            return False
    
    def _compute_neighbor_similarities(self, main_text, neighbor_text, query_analysis):
        """Compute semantic and lexical similarity between sentences"""
        main_embedding = self.model.encode([main_text])
        neighbor_embedding = self.model.encode([neighbor_text])
        semantic_sim = cosine_similarity(main_embedding, neighbor_embedding)[0][0]
        
        main_words = set(re.findall(r'\b\w+\b', main_text.lower()))
        neighbor_words = set(re.findall(r'\b\w+\b', neighbor_text.lower()))
        
        if not main_words or not neighbor_words:
            lexical_sim = 0.0
        else:
            intersection = len(main_words & neighbor_words)
            union = len(main_words | neighbor_words)
            lexical_sim = intersection / union if union > 0 else 0.0
        
        return semantic_sim, lexical_sim
    
    def _get_expanded_context(self, chapter_num, sentence_num, query, book_key, theme_context=None):
        """Get context sentences with strict global deduplication"""
        context = {
            'prev_sentences': [],
            'next_sentences': [],
            'prev_relevant_count': 0,
            'next_relevant_count': 0
        }
        
        chapters_df = self.books_data[book_key]['chapters']
        chapter_sentences = chapters_df[
            chapters_df['chapter_number'] == chapter_num
        ].sort_values('sentence_number')
        
        current_idx = None
        main_sentence_text = None
        for idx, row in chapter_sentences.iterrows():
            if row['sentence_number'] == sentence_num:
                current_idx = idx
                main_sentence_text = row['sentence_text']
                break
        
        if current_idx is None:
            return context
        
        chapter_list = chapter_sentences.index.tolist()
        current_pos = chapter_list.index(current_idx)
        
        # Expand backward
        for i in range(1, self.MAX_CONTEXT_EXPANSION + 1):
            if current_pos - i < 0:
                break
            
            prev_idx = chapter_list[current_pos - i]
            prev_row = chapters_df.loc[prev_idx]
            prev_id = self._get_passage_id(prev_row['chapter_number'], prev_row['sentence_number'])
            
            if prev_id in self.used_passages[book_key]:
                break
            
            is_relevant = self._check_context_relevance(
                prev_row['sentence_text'], query, book_key, theme_context
            )
            
            semantic_sim, lexical_sim = self._compute_neighbor_similarities(
                main_sentence_text, prev_row['sentence_text'], []
            )
            
            context['prev_sentences'].append({
                'sentence_number': prev_row['sentence_number'],
                'sentence_text': prev_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i,
                'semantic_similarity': semantic_sim,
                'lexical_similarity': lexical_sim
            })
            
            if is_relevant:
                context['prev_relevant_count'] += 1
            else:
                break
        
        context['prev_sentences'].reverse()
        
        # Expand forward
        for i in range(1, self.MAX_CONTEXT_EXPANSION + 1):
            if current_pos + i >= len(chapter_list):
                break
            
            next_idx = chapter_list[current_pos + i]
            next_row = chapters_df.loc[next_idx]
            next_id = self._get_passage_id(next_row['chapter_number'], next_row['sentence_number'])
            
            if next_id in self.used_passages[book_key]:
                break
            
            is_relevant = self._check_context_relevance(
                next_row['sentence_text'], query, book_key, theme_context
            )
            
            semantic_sim, lexical_sim = self._compute_neighbor_similarities(
                main_sentence_text, next_row['sentence_text'], []
            )
            
            context['next_sentences'].append({
                'sentence_number': next_row['sentence_number'],
                'sentence_text': next_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i,
                'semantic_similarity': semantic_sim,
                'lexical_similarity': lexical_sim
            })
            
            if is_relevant:
                context['next_relevant_count'] += 1
            else:
                break
        
        return context
    
    def _retrieve_passages(self, query, query_analysis, book_key, top_k=9):
        """CLEAR-based hybrid retrieval for specified book with immediate deduplication"""
        self.used_passages[book_key] = set()
        
        book_data = self.books_data[book_key]
        chapters_df = book_data['chapters']
        embeddings = book_data['embeddings']
        
        query_words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        query_length = len(query_words)
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        weights = self._compute_dynamic_weights(query_length, stopword_ratio)
        lambda_emb, lambda_lex = weights
        
        query_embedding = self.model.encode([query])
        semantic_similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Create scored candidates first (without context)
        candidates = []
        
        for idx, semantic_sim in enumerate(semantic_similarities):
            if semantic_sim < self.MIN_SEMANTIC_THRESHOLD:
                continue
            
            row = chapters_df.iloc[idx]
            passage_id = self._get_passage_id(row['chapter_number'], row['sentence_number'])
            
            if passage_id in self.used_passages[book_key]:
                continue
            
            lexical_score = self._compute_lexical_score_weighted(
                query, row['sentence_text'], query_analysis
            )
            
            if lexical_score >= 0.95:
                match_type = 'exact'
            elif lexical_score >= 0.3:
                match_type = 'partial_lexical'
            else:
                match_type = 'semantic'
            
            final_score = self._calculate_clear_score(
                semantic_sim, lexical_score, weights, row['sentence_word_count']
            )
            
            candidates.append({
                'index': idx,
                'chapter_number': row['chapter_number'],
                'chapter_title': row['chapter_title'],
                'sentence_number': row['sentence_number'],
                'sentence_text': row['sentence_text'],
                'semantic_score': semantic_sim,
                'lexical_score': lexical_score,
                'final_score': final_score,
                'match_type': match_type,
                'word_count': row['sentence_word_count'],
                'weights': {'lambda_emb': lambda_emb, 'lambda_lex': lambda_lex},
                'stopword_ratio': stopword_ratio
            })
        
        # Sort candidates by score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Now build final results with immediate deduplication
        chapter_counts = {}
        results = []
        
        for candidate in candidates:
            # Check again if passage was used (might have been added as context)
            passage_id = self._get_passage_id(candidate['chapter_number'], candidate['sentence_number'])
            if passage_id in self.used_passages[book_key]:
                continue
            
            # Check chapter limit
            ch_num = candidate['chapter_number']
            count = chapter_counts.get(ch_num, 0)
            if count >= 3:
                continue
            
            # Mark main passage as used FIRST
            self.used_passages[book_key].add(passage_id)
            
            # Now get context (this will respect the updated used_passages)
            try:
                context = self._get_expanded_context(
                    candidate['chapter_number'], candidate['sentence_number'], query, book_key
                )
                has_relevant_context = (
                    context['prev_relevant_count'] > 0 or context['next_relevant_count'] > 0
                )
            except:
                context = {
                    'prev_sentences': [], 'next_sentences': [],
                    'prev_relevant_count': 0, 'next_relevant_count': 0
                }
                has_relevant_context = False
            
            # Mark context passages as used
            for sent in context.get('prev_sentences', []):
                self.used_passages[book_key].add(self._get_passage_id(ch_num, sent['sentence_number']))
            for sent in context.get('next_sentences', []):
                self.used_passages[book_key].add(self._get_passage_id(ch_num, sent['sentence_number']))
            
            # Add full result
            candidate['context'] = context
            candidate['has_relevant_context'] = has_relevant_context
            candidate['total_context_sentences'] = len(context['prev_sentences']) + len(context['next_sentences'])
            
            results.append(candidate)
            chapter_counts[ch_num] = count + 1
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _get_thematic_classification(self, passages, query, book_key):
        """Classify passages by themes"""
        if not passages:
            return passages, False, 0.0
        
        book_data = self.books_data[book_key]
        theme_embeddings = book_data['theme_embeddings']
        themes_df = book_data['themes']
        
        thematic_results = []
        
        for passage in passages:
            sentence_embedding = self.model.encode([passage['sentence_text']])
            theme_similarities = cosine_similarity(sentence_embedding, theme_embeddings)[0]
            
            matching_themes = []
            for idx, similarity in enumerate(theme_similarities):
                if similarity >= self.THEMATIC_THRESHOLD:
                    theme_row = themes_df.iloc[idx]
                    matching_themes.append({
                        'tagalog_title': theme_row['Tagalog Title'],
                        'meaning': theme_row['Meaning'],
                        'confidence': similarity
                    })
            
            matching_themes.sort(key=lambda x: x['confidence'], reverse=True)
            
            enhanced = passage.copy()
            if matching_themes:
                enhanced['themes'] = matching_themes[:2]
                enhanced['primary_theme'] = matching_themes[0]
                enhanced['has_theme'] = True
            else:
                enhanced['themes'] = []
                enhanced['primary_theme'] = None
                enhanced['has_theme'] = False
            
            thematic_results.append(enhanced)
        
        sentences_with_themes = sum(1 for s in thematic_results if s['has_theme'])
        thematic_coverage = sentences_with_themes / len(thematic_results) if thematic_results else 0
        
        avg_theme_conf = 0.0
        if sentences_with_themes > 0:
            theme_confs = [s['primary_theme']['confidence'] for s in thematic_results if s['has_theme']]
            avg_theme_conf = np.mean(theme_confs)
        
        has_themes = thematic_coverage >= 0.3 and avg_theme_conf >= self.THEMATIC_THRESHOLD
        
        return thematic_results, has_themes, avg_theme_conf
    
    def query(self, user_query):
        """Main query interface - searches both books with lexical grounding check"""
        # First validate Filipino
        is_valid, validation_info = self.query_analyzer.validate_filipino_query(user_query)
        
        if not is_valid:
            return {
                'type': 'invalid_filipino',
                'validation_info': validation_info,
                'message': f"Invalid Filipino query: {validation_info['reason']}"
            }
        
        # LEXICAL GROUNDING CHECK - block if zero corpus overlap
        is_grounded, overlap_info = self._check_lexical_grounding(user_query)
        
        if not is_grounded:
            return {
                'type': 'no_lexical_grounding',
                'overlap_info': overlap_info,
                'message': f"Query blocked: {overlap_info['reason']}"
            }
        
        query_analysis = self.query_analyzer.analyze_query_words(user_query)
        query_length = len(re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', user_query))
        stopword_ratio = self.query_analyzer.get_stopword_ratio(user_query)
        
        results_by_book = {}
        
        for book_key in self.books_data.keys():
            passages = self._retrieve_passages(user_query, query_analysis, book_key)
            
            if passages:
                thematic_passages, has_themes, avg_theme_conf = self._get_thematic_classification(
                    passages, user_query, book_key
                )
                
                avg_semantic = np.mean([p['semantic_score'] for p in passages])
                avg_lexical = np.mean([p['lexical_score'] for p in passages])
                avg_final = np.mean([p['final_score'] for p in passages])
                
                exact_count = sum(1 for p in passages if p['match_type'] == 'exact')
                partial_lex_count = sum(1 for p in passages if p['match_type'] == 'partial_lexical')
                semantic_only_count = sum(1 for p in passages if p['match_type'] == 'semantic')
                context_count = sum(1 for p in thematic_passages if p['has_relevant_context'])
                total_context = sum(p.get('total_context_sentences', 0) for p in thematic_passages)
                
                results_by_book[book_key] = {
                    'results': thematic_passages,
                    'has_themes': has_themes,
                    'avg_semantic': avg_semantic,
                    'avg_lexical': avg_lexical,
                    'avg_final': avg_final,
                    'avg_theme_conf': avg_theme_conf,
                    'exact_matches': exact_count,
                    'partial_lexical_matches': partial_lex_count,
                    'semantic_only_matches': semantic_only_count,
                    'context_matches': context_count,
                    'total_context_sentences': total_context,
                    'weights': passages[0]['weights'] if passages else {}
                }
        
        if not results_by_book:
            return {
                'type': 'no_matches',
                'message': "No matches found in either novel",
                'query_analysis': query_analysis,
                'query_length': query_length,
                'stopword_ratio': stopword_ratio,
                'overlap_info': overlap_info
            }
        
        return {
            'type': 'success',
            'results_by_book': results_by_book,
            'query_length': query_length,
            'stopword_ratio': stopword_ratio,
            'query_analysis': query_analysis,
            'overlap_info': overlap_info
        }
    
    def _display_neighbor_similarities(self, context):
        """Display neighbor similarities in clean table format"""
        prev_sentences = context.get('prev_sentences', [])
        next_sentences = context.get('next_sentences', [])
        
        if not prev_sentences and not next_sentences:
            return
        
        sim_table = Table(
            title="Neighbor Similarity Metrics",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=False
        )
        
        sim_table.add_column("Position", style="bright_white", width=12, justify="center")
        sim_table.add_column("S#", style="bright_yellow", width=6, justify="center")
        sim_table.add_column("Semantic %", style="bright_cyan", width=14, justify="center")
        sim_table.add_column("Lexical %", style="bright_green", width=14, justify="center")
        
        for sent in prev_sentences:
            sim_table.add_row(
                "Previous",
                str(sent['sentence_number']),
                f"{sent['semantic_similarity']:.1%}",
                f"{sent['lexical_similarity']:.1%}"
            )
        
        for sent in next_sentences:
            sim_table.add_row(
                "Next",
                str(sent['sentence_number']),
                f"{sent['semantic_similarity']:.1%}",
                f"{sent['lexical_similarity']:.1%}"
            )
        
        self.console.print(sim_table)
    
    def display_results(self, response, query=""):
        """Display results with rich formatting, separated by book"""
        result_type = response['type']
        
        if result_type == 'invalid_filipino':
            validation_info = response['validation_info']
            
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)
            
            validation_panel = Panel(
                f"{response['message']}\n\n"
                f"Analysis:\n"
                f"  Total words: {validation_info['total_words']}\n"
                f"  Valid Filipino: {validation_info['valid_words']}\n"
                f"  Invalid: {', '.join(validation_info['invalid_words']) if validation_info['invalid_words'] else 'N/A'}\n"
                f"  Valid ratio: {validation_info['valid_ratio']:.1%}",
                title="Invalid Query",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(validation_panel)
            return
        
        if result_type == 'no_lexical_grounding':
            overlap_info = response['overlap_info']
            
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)
            
            grounding_panel = Panel(
                f"{response['message']}\n\n"
                f"Lexical Grounding Analysis:\n"
                f"  Content words in query: {overlap_info['total_content_words']}\n"
                f"  Query words: {', '.join(overlap_info['content_words'])}\n"
                f"  Matched in corpus: {overlap_info['total_matched']}\n"
                f"  Matched by book: {', '.join([f'{k}: {len(v)}' for k, v in overlap_info['matched_words'].items()]) if overlap_info['matched_words'] else 'None'}\n\n"
                f"This query contains words not found in the corpus vocabulary.\n"
                f"Only queries with at least one content word matching the corpus are processed.",
                title="❌ No Lexical Grounding",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(grounding_panel)
            return
        
        if result_type != 'success':
            error_panel = Panel(
                f"{response['message']}",
                title="No Results",
                style="yellow",
                box=box.ROUNDED
            )
            self.console.print(error_panel)
            
            if 'query_analysis' in response:
                self._display_query_analysis(response['query_analysis'])
            return
        
        # Display lexical grounding info
        if 'overlap_info' in response:
            overlap_info = response['overlap_info']
            grounding_text = (
                f"✓ Lexical Grounding: {overlap_info['total_matched']}/{overlap_info['total_content_words']} content words matched | "
                f"Matched: {', '.join(overlap_info['content_words'][:5])}"
            )
            if len(overlap_info['content_words']) > 5:
                grounding_text += "..."
            
            grounding_panel = Panel(
                grounding_text,
                style="green",
                box=box.SIMPLE
            )
            self.console.print(grounding_panel)
        
        # Display query analysis
        if 'query_analysis' in response:
            self._display_query_analysis(response['query_analysis'])
        
        results_by_book = response['results_by_book']
        query_len = response.get('query_length', 0)
        stopword_ratio = response.get('stopword_ratio', 0.0)
        
        # Display header
        header_text = Text(f"Results for: '{query}'", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(header_panel)
        
        # Display results for each book
        book_names = {'noli': 'Noli Me Tangere', 'elfili': 'El Filibusterismo'}
        book_colors = {'noli': 'bright_yellow', 'elfili': 'bright_magenta'}
        
        for book_key in ['noli', 'elfili']:
            if book_key not in results_by_book:
                continue
            
            book_results = results_by_book[book_key]
            results = book_results['results']
            has_themes = book_results['has_themes']
            
            weights = book_results.get('weights', {})
            
            # Book separator
            book_title = book_names[book_key]
            book_header = Panel(
                Align.center(Text(f"📖 {book_title} 📖", style=f"bold {book_colors[book_key]}")),
                style=book_colors[book_key],
                box=box.HEAVY,
                padding=(1, 2)
            )
            self.console.print(book_header)
            
            # Book metrics
            metrics_text = (
                f"CLEAR Hybrid | Query: {query_len} words | Stopwords: {stopword_ratio:.1%}\n"
                f"Weights: λ_emb={weights.get('lambda_emb', 0):.2f} | λ_lex={weights.get('lambda_lex', 0):.2f}\n"
                f"Semantic: {book_results['avg_semantic']:.1%} | "
                f"Lexical: {book_results['avg_lexical']:.1%} | "
                f"Final: {book_results['avg_final']:.1%}\n"
                f"Exact: {book_results['exact_matches']} | "
                f"Partial: {book_results['partial_lexical_matches']} | "
                f"Semantic-only: {book_results['semantic_only_matches']}\n"
                f"Context: {book_results['context_matches']} | "
                f"Total Context: {book_results['total_context_sentences']} | "
                f"Results: {len(results)}"
            )
            if book_results['avg_theme_conf'] > 0:
                metrics_text += f"\nThematic: {book_results['avg_theme_conf']:.1%}"
            
            book_metrics_panel = Panel(
                metrics_text,
                style=book_colors[book_key],
                box=box.ROUNDED,
                padding=(1, 2)
            )
            self.console.print(book_metrics_panel)
            
            # Display results (max 3 chapters, 3 sentences per chapter)
            for i, result in enumerate(results, 1):
                self.console.print(f"\nResult {i} / {len(results)}", style=f"bold {book_colors[book_key]}")
                
                main_table = Table(
                    show_header=True,
                    header_style=f"bold {book_colors[book_key]}",
                    border_style=book_colors[book_key],
                    box=box.ROUNDED,
                    padding=(0, 1),
                    expand=True
                )
                
                main_table.add_column("Book", style="cyan", width=15)
                main_table.add_column("Ch", style="bright_green", width=4, justify="center")
                main_table.add_column("S#", style="yellow", width=4, justify="center")
                main_table.add_column("Semantic", style="bright_cyan", width=9, justify="center")
                main_table.add_column("Lexical", style="bright_yellow", width=9, justify="center")
                main_table.add_column("Final", style="bright_white", width=9, justify="center")
                main_table.add_column("Type", style="bright_green", width=14, justify="center")
                main_table.add_column("Ctx", style="dim white", width=5, justify="center")
                
                match_type = result['match_type']
                if match_type == 'exact':
                    type_display = "Exact Match"
                elif match_type == 'partial_lexical':
                    type_display = "Partial Lexical"
                else:
                    type_display = "Semantic"
                
                context_count = result.get('total_context_sentences', 0)
                
                main_table.add_row(
                    book_title,
                    str(result['chapter_number']),
                    str(result['sentence_number']),
                    f"{result['semantic_score']:.1%}",
                    f"{result['lexical_score']:.1%}",
                    f"{result['final_score']:.1%}",
                    type_display,
                    str(context_count) if context_count > 0 else "-"
                )
                
                self.console.print(main_table)
                
                chapter_panel = Panel(
                    f"{result['chapter_title']}",
                    style="bright_green",
                    box=box.SIMPLE
                )
                self.console.print(chapter_panel)
                
                content_panel = Panel(
                    result['sentence_text'],
                    style="white",
                    box=box.ROUNDED,
                    padding=(1, 2)
                )
                self.console.print(content_panel)
                
                context = result.get('context', {})
                prev_sentences = context.get('prev_sentences', [])
                next_sentences = context.get('next_sentences', [])
                
                if prev_sentences or next_sentences:
                    # Display neighbor similarity metrics
                    self.console.print("")
                    self._display_neighbor_similarities(context)
                    
                    # Display full context content
                    self.console.print("")
                    context_table = Table(
                        title=f"Context ({len(prev_sentences) + len(next_sentences)} sentences)",
                        show_header=True,
                        header_style="bold yellow",
                        border_style="yellow",
                        box=box.SIMPLE,
                        expand=True
                    )
                    
                    context_table.add_column("Position", style="yellow", width=10)
                    context_table.add_column("S#", style="bright_yellow", width=4, justify="center")
                    context_table.add_column("Dist", style="dim yellow", width=5, justify="center")
                    context_table.add_column("Content", style="white", min_width=40)
                    
                    for sent in prev_sentences:
                        context_table.add_row(
                            "Previous",
                            str(sent['sentence_number']),
                            f"-{sent['distance']}",
                            sent['sentence_text']
                        )
                    
                    for sent in next_sentences:
                        context_table.add_row(
                            "Next",
                            str(sent['sentence_number']),
                            f"+{sent['distance']}",
                            sent['sentence_text']
                        )
                    
                    self.console.print(context_table)
                
                if has_themes and result.get('has_theme'):
                    primary_theme = result['primary_theme']
                    
                    theme_table = Table(
                        title="Thematic Analysis",
                        show_header=True,
                        header_style="bold magenta",
                        border_style="magenta",
                        box=box.SIMPLE,
                        expand=True
                    )
                    
                    theme_table.add_column("Tagalog Title", style="bright_magenta", width=25)
                    theme_table.add_column("Meaning", style="magenta", min_width=40)
                    theme_table.add_column("Confidence", style="bright_cyan", width=12, justify="center")
                    
                    theme_table.add_row(
                        primary_theme['tagalog_title'],
                        primary_theme['meaning'],
                        f"{primary_theme['confidence']:.1%}"
                    )
                    
                    self.console.print(theme_table)
                
                if i < len(results):
                    self.console.print("─" * 100, style="dim blue")
            
            # Summary for this book
            chapters_found = len(set(r['chapter_number'] for r in results))
            exact_count = sum(1 for r in results if r['match_type'] == 'exact')
            partial_count = sum(1 for r in results if r['match_type'] == 'partial_lexical')
            semantic_count = sum(1 for r in results if r['match_type'] == 'semantic')
            context_count = sum(1 for r in results if r.get('has_relevant_context', False))
            theme_count = sum(1 for r in results if r.get('has_theme', False))
            total_context = book_results['total_context_sentences']
            
            classification = "Thematic Analysis" if has_themes else "Semantic Search"
            
            summary_parts = [
                f"{classification}",
                f"{len(results)} sentences from {chapters_found} chapters",
                f"{exact_count} exact",
                f"{partial_count} partial",
                f"{semantic_count} semantic",
                f"{context_count} with context",
                f"{total_context} total context"
            ]
            
            if has_themes:
                summary_parts.append(f"{theme_count} with themes")
            
            summary = " | ".join(summary_parts)
            footer_panel = Panel(
                Align.center(Text(summary, style=f"bold {book_colors[book_key]}")),
                style=book_colors[book_key],
                box=box.DOUBLE,
                padding=(1, 2)
            )
            self.console.print(footer_panel)
            self.console.print("\n")
    
    def _display_query_analysis(self, query_analysis):
        """Display query word analysis with semantic weights"""
        if not query_analysis:
            return
        
        analysis_table = Table(
            title="Query Word Analysis (Stopword-Aware Weighting)",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True
        )
        
        analysis_table.add_column("Word", style="bright_white", width=20)
        analysis_table.add_column("Frequency", style="bright_yellow", width=12, justify="right")
        analysis_table.add_column("Type", style="bright_cyan", width=15, justify="center")
        analysis_table.add_column("Weight", style="bright_green", width=10, justify="center")
        
        for item in query_analysis:
            if item['is_stopword']:
                word_type = "Stopword"
                type_style = "dim red"
            elif item['is_content_word']:
                word_type = "Content"
                type_style = "bright_green"
            else:
                word_type = "Common"
                type_style = "yellow"
            
            weight = item['semantic_weight']
            if weight >= 0.9:
                weight_style = "bright_green"
            elif weight >= 0.5:
                weight_style = "yellow"
            else:
                weight_style = "dim red"
            
            analysis_table.add_row(
                item['word'],
                f"{item['frequency']:.6f}",
                f"[{type_style}]{word_type}[/{type_style}]",
                f"[{weight_style}]{weight:.2f}[/{weight_style}]"
            )
        
        self.console.print("\n")
        self.console.print(analysis_table)
        
        # Summary stats
        total_words = len(query_analysis)
        stopword_count = sum(1 for item in query_analysis if item['is_stopword'])
        content_count = sum(1 for item in query_analysis if item['is_content_word'])
        avg_weight = np.mean([item['semantic_weight'] for item in query_analysis])
        
        stats_text = (
            f"Total: {total_words} words | "
            f"Stopwords: {stopword_count} ({stopword_count/total_words:.1%}) | "
            f"Content: {content_count} ({content_count/total_words:.1%}) | "
            f"Avg Weight: {avg_weight:.2f}"
        )
        
        stats_panel = Panel(
            stats_text,
            style="cyan",
            box=box.SIMPLE
        )
        self.console.print(stats_panel)
        self.console.print("\n")

if __name__ == "__main__":
    system = CleanNoliSystem()
    
    welcome_panel = Panel(
        Align.center(Text(
            "Noli Me Tangere & El Filibusterismo Clean CLEAR System\n"
            "XLM-RoBERTa Semantic + Weighted Lexical Matching\n"
            "Multi-Book Support | Official Tagalog Stopwords | Strict Deduplication\n"
            "Lexical Grounding Check | Neighbor Similarity Metrics (Semantic & Lexical %)",
            style="bold white"
        )),
        style="bright_green",
        box=box.HEAVY
    )
    system.console.print(welcome_panel)
    
    while True:
        system.console.print("\n" + "─" * 80, style="dim")
        user_input = system.console.input("[bold cyan]Enter query (or 'exit'): [/bold cyan]").strip()
        
        if user_input.lower() == 'exit':
            goodbye_panel = Panel(
                Align.center(Text("Thank you for using the system!", style="bold green")),
                style="bright_green",
                box=box.ROUNDED
            )
            system.console.print(goodbye_panel)
            break
        
        if not user_input:
            system.console.print("[red]Please enter a valid query.[/red]")
            continue
        
        system.console.print(f"[dim]Processing '{user_input}'...[/dim]")
        
        try:
            response = system.query(user_input)
            system.display_results(response, user_input)
        except Exception as e:
            error_panel = Panel(
                f"System error: {str(e)}\nTry a different query.",
                title="Error",
                style="red",
                box=box.ROUNDED
            )
            system.console.print(error_panel)