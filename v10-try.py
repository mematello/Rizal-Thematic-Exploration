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
import torch

warnings.filterwarnings('ignore')

class StopwordMaskedEncoder:
    """
    Encodes text with stopword masking at the token embedding level.
    Tokens identified as stopwords (via stopwords-iso) have their embeddings zeroed
    during pooling, so only content words contribute to the final sentence embedding.
    """
    
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.tokenizer
        
        # Load official Tagalog stopwords
        try:
            from stopwordsiso import stopwords
            self.stopwords = set(stopwords('tl'))
            print(f"Loaded {len(self.stopwords)} official Tagalog stopwords for embedding masking")
        except ImportError:
            print("Warning: stopwords-iso not installed. Install with: pip install stopwords-iso")
            self.stopwords = {'ng', 'sa', 'ang', 'na', 'ay', 'at', 'mga'}
    
    def _get_token_to_word_mapping(self, text, input_ids):
        """
        Map token IDs back to original words to identify stopwords.
        Returns a mask: 1 for content words, 0 for stopwords.
        """
        # Decode tokens to get individual token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        # Extract words from original text
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', text.lower())
        
        # Create mask: 1 for content, 0 for stopwords
        mask = []
        for token in tokens:
            # Clean token (remove special characters like ▁ for sentencepiece)
            clean_token = token.replace('▁', '').replace('##', '').strip()
            
            # Check if this token corresponds to a stopword
            is_stopword = clean_token.lower() in self.stopwords
            mask.append(0.0 if is_stopword else 1.0)
        
        return torch.tensor(mask, dtype=torch.float32)
    
    def encode_with_stopword_masking(self, texts, show_progress_bar=False):
        """
        Encode texts with stopword masking applied during pooling.
        
        Process:
        1. Tokenize and encode normally
        2. Map tokens to words and identify stopwords
        3. Zero out stopword token embeddings
        4. Pool only non-zero (content word) embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            # Get token embeddings from the model
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            with torch.no_grad():
                # Get token-level embeddings - FIXED: use the SentenceTransformer directly
                token_embeddings = self.model.encode(text, output_value='token_embeddings', 
                                                   convert_to_tensor=True)
                if token_embeddings.dim() == 3:
                    token_embeddings = token_embeddings[0]  # Shape: (num_tokens, embedding_dim)
            
            # Create stopword mask
            mask = self._get_token_to_word_mapping(text, inputs['input_ids'])
            
            # Ensure mask matches token_embeddings length
            if len(mask) > token_embeddings.shape[0]:
                mask = mask[:token_embeddings.shape[0]]
            elif len(mask) < token_embeddings.shape[0]:
                # Pad mask if needed
                mask_padding = torch.ones(token_embeddings.shape[0] - len(mask), dtype=torch.float32)
                mask = torch.cat([mask, mask_padding])
            
            # Expand mask to match embedding dimensions
            mask_expanded = mask.unsqueeze(1).expand_as(token_embeddings)
            
            # Zero out stopword embeddings
            masked_embeddings = token_embeddings * mask_expanded
            
            # Pool only non-zero embeddings (mean pooling of content words)
            content_token_count = mask.sum()
            if content_token_count > 0:
                sentence_embedding = masked_embeddings.sum(dim=0) / content_token_count
            else:
                # Fallback: if all stopwords, use mean of all tokens
                sentence_embedding = token_embeddings.mean(dim=0)
            
            embeddings.append(sentence_embedding.cpu().numpy())
        
        return np.array(embeddings)

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
            
            # Determine semantic weight (note: embeddings handle this via masking now)
            if is_stopword:
                semantic_weight = 0.0  # Completely masked in embeddings
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

class CleanNoliSystem:
    """
    Clean CLEAR-inspired hybrid retrieval with stopword-masked embeddings
    Formula: s_final(q,d) = λ_emb · s_emb_masked(q,d) + λ_lex · s_lex_weighted(q,d)
    
    Key improvements:
    - Stopword masking at token embedding level (zero contribution)
    - Official Tagalog stopwords from stopwords-iso
    - Dynamic stopword weighting based on query composition
    - Strict global deduplication across all results and contexts
    """
    
    def __init__(self):
        self.console = Console()
        
        self.console.print("Loading XLM-RoBERTa model...", style="cyan")
        base_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        # Create stopword-masked encoder
        self.encoder = StopwordMaskedEncoder(base_model)
        
        self.console.print("Loading datasets...", style="cyan")
        try:
            self.chapters_df = pd.read_csv('chapters.csv')
            self.themes_df = pd.read_csv('themes.csv')
        except FileNotFoundError as e:
            self.console.print(f"Error loading datasets: {e}", style="red")
            self.console.print("Please ensure chapters.csv and themes.csv are in the current directory", style="yellow")
            # Create empty dataframes to prevent further errors
            self.chapters_df = pd.DataFrame(columns=['chapter_number', 'chapter_title', 'sentence_number', 'sentence_text'])
            self.themes_df = pd.DataFrame(columns=['Tagalog Title', 'Meaning'])
        
        if not self.chapters_df.empty:
            self.chapters_df.columns = self.chapters_df.columns.str.strip()
        if not self.themes_df.empty:
            self.themes_df.columns = self.themes_df.columns.str.strip()
        
        self.console.print("Initializing query analyzer with official stopwords...", style="cyan")
        self.query_analyzer = QueryAnalyzer()
        
        if not self.chapters_df.empty:
            self.console.print("Computing stopword-masked embeddings...", style="cyan")
            self._compute_embeddings()
        else:
            self.console.print("No chapter data available for embedding computation", style="yellow")
            self.chapter_embeddings = np.array([])
            self.theme_embeddings = np.array([])
        
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
        
        # Global deduplication tracking
        self.used_passages = set()
        
        self.console.print("Clean CLEAR system with stopword-masked embeddings ready!", style="bold green")
    
    def _compute_embeddings(self):
        """Compute stopword-masked embeddings for chapters and themes"""
        if self.chapters_df.empty:
            return
            
        self.chapters_df['combined_text'] = (
            self.chapters_df['chapter_title'].astype(str) + " " + 
            self.chapters_df['sentence_text'].astype(str)
        )
        
        self.chapters_df['sentence_word_count'] = (
            self.chapters_df['sentence_text'].astype(str).apply(lambda x: len(x.split()))
        )
        
        texts = self.chapters_df['combined_text'].tolist()
        self.chapter_embeddings = self.encoder.encode_with_stopword_masking(texts)
        
        if not self.themes_df.empty:
            self.themes_df['theme_text'] = (
                self.themes_df['Tagalog Title'].astype(str) + " " + 
                self.themes_df['Meaning'].astype(str)
            )
            
            theme_texts = self.themes_df['theme_text'].tolist()
            self.theme_embeddings = self.encoder.encode_with_stopword_masking(theme_texts)
        else:
            self.theme_embeddings = np.array([])
    
    def _get_passage_id(self, chapter_num, sentence_num):
        """Create unique identifier for passages"""
        return (int(chapter_num), int(sentence_num))
    
    def _compute_dynamic_weights(self, query_length, stopword_ratio):
        """
        Dynamic weighting with stopword awareness
        - Longer queries: trust semantic more
        - High stopword ratio: reduce lexical weight significantly
        """
        L = query_length
        
        # Base weights by query length
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
        
        # Adjust for stopword-heavy queries
        if stopword_ratio > self.HIGH_STOPWORD_RATIO:
            adjustment = (stopword_ratio - self.HIGH_STOPWORD_RATIO) * 0.5
            alpha = min(0.95, alpha + adjustment)
            beta = 1.0 - alpha
        
        return alpha, beta
    
    def _compute_lexical_score_weighted(self, query, sentence_text, query_analysis):
        """Compute weighted lexical overlap score"""
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()
        
        if query_lower == sentence_lower:
            return 1.0
        
        if query_lower in sentence_lower:
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
        """CLEAR hybrid scoring: s_final = λ_emb·s_emb + λ_lex·s_lex - penalty"""
        lambda_emb, lambda_lex = weights
        
        final_score = (lambda_emb * semantic_sim) + (lambda_lex * lexical_score)
        
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (
                self.SHORT_SENTENCE_THRESHOLD - word_count
            ) / self.SHORT_SENTENCE_THRESHOLD
            final_score -= penalty
        
        return max(0.0, min(1.0, final_score))
    
    def _check_context_relevance(self, context_text, query, theme_context=None):
        """Check if context sentence is relevant to query/theme"""
        try:
            query_embedding = self.encoder.encode_with_stopword_masking([query])
            context_embedding = self.encoder.encode_with_stopword_masking([context_text])
            query_similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
            
            theme_similarity = 0.0
            if theme_context:
                theme_embedding = self.encoder.encode_with_stopword_masking([theme_context])
                theme_similarity = cosine_similarity(theme_embedding, context_embedding)[0][0]
            
            return max(query_similarity, theme_similarity) >= self.CONTEXT_RELEVANCE_THRESHOLD
        except:
            return False
    
    def _get_expanded_context(self, chapter_num, sentence_num, query, theme_context=None):
        """Get context sentences with STRICT global deduplication"""
        context = {
            'prev_sentences': [],
            'next_sentences': [],
            'prev_relevant_count': 0,
            'next_relevant_count': 0
        }
        
        if self.chapters_df.empty:
            return context
            
        chapter_sentences = self.chapters_df[
            self.chapters_df['chapter_number'] == chapter_num
        ].sort_values('sentence_number')
        
        current_idx = None
        for idx, row in chapter_sentences.iterrows():
            if row['sentence_number'] == sentence_num:
                current_idx = idx
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
            prev_row = self.chapters_df.loc[prev_idx]
            prev_id = self._get_passage_id(prev_row['chapter_number'], prev_row['sentence_number'])
            
            if prev_id in self.used_passages:
                break
            
            is_relevant = self._check_context_relevance(
                prev_row['sentence_text'], query, theme_context
            )
            
            context['prev_sentences'].append({
                'sentence_number': prev_row['sentence_number'],
                'sentence_text': prev_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i
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
            next_row = self.chapters_df.loc[next_idx]
            next_id = self._get_passage_id(next_row['chapter_number'], next_row['sentence_number'])
            
            if next_id in self.used_passages:
                break
            
            is_relevant = self._check_context_relevance(
                next_row['sentence_text'], query, theme_context
            )
            
            context['next_sentences'].append({
                'sentence_number': next_row['sentence_number'],
                'sentence_text': next_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i
            })
            
            if is_relevant:
                context['next_relevant_count'] += 1
            else:
                break
        
        return context
    
    def _mark_passages_as_used(self, chapter_num, sentence_num, context):
        """Mark all passages (main + context) as globally used"""
        self.used_passages.add(self._get_passage_id(chapter_num, sentence_num))
        
        for sent in context.get('prev_sentences', []):
            self.used_passages.add(self._get_passage_id(chapter_num, sent['sentence_number']))
        
        for sent in context.get('next_sentences', []):
            self.used_passages.add(self._get_passage_id(chapter_num, sent['sentence_number']))
    
    def _retrieve_passages(self, query, query_analysis, top_k=9):
        """CLEAR-based hybrid retrieval with stopword-masked embeddings"""
        self.used_passages = set()
        
        if self.chapters_df.empty or len(self.chapter_embeddings) == 0:
            return []
            
        query_words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        query_length = len(query_words)
        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        weights = self._compute_dynamic_weights(query_length, stopword_ratio)
        lambda_emb, lambda_lex = weights
        
        query_embedding = self.encoder.encode_with_stopword_masking([query])
        semantic_similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
        
        results = []
        
        for idx, semantic_sim in enumerate(semantic_similarities):
            if semantic_sim < self.MIN_SEMANTIC_THRESHOLD:
                continue
            
            row = self.chapters_df.iloc[idx]
            passage_id = self._get_passage_id(row['chapter_number'], row['sentence_number'])
            
            if passage_id in self.used_passages:
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
            
            try:
                context = self._get_expanded_context(
                    row['chapter_number'], row['sentence_number'], query
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
            
            final_score = self._calculate_clear_score(
                semantic_sim, lexical_score, weights, row['sentence_word_count']
            )
            
            results.append({
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
                'context': context,
                'has_relevant_context': has_relevant_context,
                'total_context_sentences': len(context['prev_sentences']) + len(context['next_sentences']),
                'weights': {'lambda_emb': lambda_emb, 'lambda_lex': lambda_lex},
                'stopword_ratio': stopword_ratio
            })
        
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        for result in results[:top_k]:
            self._mark_passages_as_used(
                result['chapter_number'], result['sentence_number'], result['context']
            )
        
        chapter_counts = {}
        final_results = []
        
        for result in results:
            ch_num = result['chapter_number']
            count = chapter_counts.get(ch_num, 0)
            
            if count < 3:
                final_results.append(result)
                chapter_counts[ch_num] = count + 1
                
                if len(final_results) >= top_k:
                    break
        
        return final_results[:top_k]
    
    def _get_thematic_classification(self, passages, query):
        """Classify passages by themes using stopword-masked embeddings"""
        if not passages or self.themes_df.empty or len(self.theme_embeddings) == 0:
            return passages, False, 0.0
        
        thematic_results = []
        
        for passage in passages:
            sentence_embedding = self.encoder.encode_with_stopword_masking([passage['sentence_text']])
            theme_similarities = cosine_similarity(sentence_embedding, self.theme_embeddings)[0]
            
            matching_themes = []
            for idx, similarity in enumerate(theme_similarities):
                if similarity >= self.THEMATIC_THRESHOLD:
                    theme_row = self.themes_df.iloc[idx]
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
        """Main query interface"""
        is_valid, validation_info = self.query_analyzer.validate_filipino_query(user_query)
        
        if not is_valid:
            return {
                'type': 'invalid_filipino',
                'validation_info': validation_info,
                'message': f"Invalid Filipino query: {validation_info['reason']}"
            }
        
        query_analysis = self.query_analyzer.analyze_query_words(user_query)
        query_length = len(re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', user_query))
        stopword_ratio = self.query_analyzer.get_stopword_ratio(user_query)
        
        passages = self._retrieve_passages(user_query, query_analysis)
        
        if not passages:
            return {
                'type': 'no_matches',
                'message': "No matches found in Noli Me Tangere",
                'query_analysis': query_analysis,
                'query_length': query_length,
                'stopword_ratio': stopword_ratio
            }
        
        thematic_passages, has_themes, avg_theme_conf = self._get_thematic_classification(
            passages, user_query
        )
        
        avg_semantic = np.mean([p['semantic_score'] for p in passages])
        avg_lexical = np.mean([p['lexical_score'] for p in passages])
        avg_final = np.mean([p['final_score'] for p in passages])
        
        exact_count = sum(1 for p in passages if p['match_type'] == 'exact')
        partial_lex_count = sum(1 for p in passages if p['match_type'] == 'partial_lexical')
        semantic_only_count = sum(1 for p in passages if p['match_type'] == 'semantic')
        context_count = sum(1 for p in thematic_passages if p['has_relevant_context'])
        total_context = sum(p.get('total_context_sentences', 0) for p in thematic_passages)
        
        weights = passages[0]['weights'] if passages else {}
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'results': thematic_passages,
            'query_length': query_length,
            'stopword_ratio': stopword_ratio,
            'weights': weights,
            'semantic_confidence': avg_semantic,
            'lexical_confidence': avg_lexical,
            'final_confidence': avg_final,
            'theme_confidence': avg_theme_conf,
            'exact_matches': exact_count,
            'partial_lexical_matches': partial_lex_count,
            'semantic_only_matches': semantic_only_count,
            'context_matches': context_count,
            'total_results': len(thematic_passages),
            'total_context_sentences': total_context,
            'query_analysis': query_analysis
        }
    
    def display_results(self, response, query=""):
        """Display results with rich formatting"""
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
        
        if 'query_analysis' in response:
            self._display_query_analysis(response['query_analysis'])
        
        results = response['results']
        subtype = response['subtype']
        has_themes = subtype == 'thematic'
        
        weights = response.get('weights', {})
        query_len = response.get('query_length', 0)
        stopword_ratio = response.get('stopword_ratio', 0.0)
        
        metrics_text = (
            f"CLEAR Hybrid (Stopword-Masked Embeddings + Lexical) | Query: {query_len} words | Stopwords: {stopword_ratio:.1%}\n"
            f"Weights: λ_emb={weights.get('lambda_emb', 0):.2f} | λ_lex={weights.get('lambda_lex', 0):.2f}\n"
            f"Semantic: {response['semantic_confidence']:.1%} | "
            f"Lexical: {response['lexical_confidence']:.1%} | "
            f"Final: {response['final_confidence']:.1%}\n"
            f"Exact: {response['exact_matches']} | "
            f"Partial: {response.get('partial_lexical_matches', 0)} | "
            f"Semantic-only: {response.get('semantic_only_matches', 0)}\n"
            f"Context: {response.get('context_matches', 0)} | "
            f"Total Context: {response.get('total_context_sentences', 0)} | "
            f"Results: {response['total_results']}"
        )
        if response['theme_confidence'] > 0:
            metrics_text += f"\nThematic: {response['theme_confidence']:.1%}"
        
        header_text = Text(f"Results for: '{query}'\n{metrics_text}", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(header_panel)
        
        for i, result in enumerate(results, 1):
            self.console.print(f"\nResult {i}", style="bold cyan")
            
            main_table = Table(
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
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
                "Noli Me Tangere",
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
                context_table.add_column("Rel", style="bright_green", width=4, justify="center")
                context_table.add_column("Content", style="white", min_width=40)
                
                for sent in prev_sentences:
                    rel_icon = "✓" if sent['is_relevant'] else "○"
                    context_table.add_row(
                        "Previous",
                        str(sent['sentence_number']),
                        f"-{sent['distance']}",
                        rel_icon,
                        sent['sentence_text']
                    )
                
                for sent in next_sentences:
                    rel_icon = "✓" if sent['is_relevant'] else "○"
                    context_table.add_row(
                        "Next",
                        str(sent['sentence_number']),
                        f"+{sent['distance']}",
                        rel_icon,
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
        
        chapters_found = len(set(r['chapter_number'] for r in results))
        exact_count = sum(1 for r in results if r['match_type'] == 'exact')
        partial_count = sum(1 for r in results if r['match_type'] == 'partial_lexical')
        semantic_count = sum(1 for r in results if r['match_type'] == 'semantic')
        context_count = sum(1 for r in results if r.get('has_relevant_context', False))
        theme_count = sum(1 for r in results if r.get('has_theme', False))
        total_context = response.get('total_context_sentences', 0)
        
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
            Align.center(Text(summary, style="bold white")),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(footer_panel)
    
    def _display_query_analysis(self, query_analysis):
        """Display query word analysis with semantic weights"""
        if not query_analysis:
            return
        
        analysis_table = Table(
            title="Query Word Analysis (Stopword Masking Applied)",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True
        )
        
        analysis_table.add_column("Word", style="bright_white", width=20)
        analysis_table.add_column("Frequency", style="bright_yellow", width=12, justify="right")
        analysis_table.add_column("Type", style="bright_cyan", width=15, justify="center")
        analysis_table.add_column("Emb Weight", style="bright_green", width=12, justify="center")
        
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
            if weight == 0.0:
                weight_style = "dim red"
                weight_display = "0.00 (masked)"
            elif weight >= 0.9:
                weight_style = "bright_green"
                weight_display = f"{weight:.2f}"
            elif weight >= 0.5:
                weight_style = "yellow"
                weight_display = f"{weight:.2f}"
            else:
                weight_style = "dim yellow"
                weight_display = f"{weight:.2f}"
            
            analysis_table.add_row(
                item['word'],
                f"{item['frequency']:.6f}",
                f"[{type_style}]{word_type}[/{type_style}]",
                f"[{weight_style}]{weight_display}[/{weight_style}]"
            )
        
        self.console.print("\n")
        self.console.print(analysis_table)
        
        total_words = len(query_analysis)
        stopword_count = sum(1 for item in query_analysis if item['is_stopword'])
        content_count = sum(1 for item in query_analysis if item['is_content_word'])
        avg_weight = np.mean([item['semantic_weight'] for item in query_analysis])
        
        stats_text = (
            f"Total: {total_words} words | "
            f"Stopwords: {stopword_count} ({stopword_count/total_words:.1%}) [MASKED in embeddings] | "
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
            "Noli Me Tangere CLEAR System with Stopword-Masked Embeddings\n"
            "XLM-RoBERTa + Token-Level Stopword Masking + Weighted Lexical\n"
            "Official Tagalog Stopwords (stopwords-iso) | Zero Contribution from Stopwords",
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