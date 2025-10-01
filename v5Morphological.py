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
from collections import Counter
import nltk
import string

# Try to import Tagalog NLP libraries
try:
    from tagalog_stemmer import TagalogStemmer
    STEMMER_AVAILABLE = True
except:
    STEMMER_AVAILABLE = False
    print("⚠️  tagalog-stemmer not available. Install: pip install tagalog-stemmer")

try:
    import requests
    REQUESTS_AVAILABLE = True
except:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests not available. Install: pip install requests")

try:
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    english_words = set(words.words())
except:
    english_words = set()

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

warnings.filterwarnings('ignore')

class TagalogWordValidator:
    """Validates Tagalog words using actual wordlists from GitHub"""
    
    def __init__(self):
        self.tagalog_wordlist = set()
        self.console = Console()
        self._load_tagalog_wordlist()
    
    def _load_tagalog_wordlist(self):
        """Load Tagalog wordlist from GitHub repositories"""
        wordlist_sources = [
            # Try multiple sources
            "https://raw.githubusercontent.com/stopwords-iso/stopwords-tl/master/stopwords-tl.txt",
            "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/tl/tl_50k.txt"
        ]
        
        if not REQUESTS_AVAILABLE:
            self.console.print("[yellow]⚠️  Cannot load online wordlists. Using corpus vocabulary only.[/yellow]")
            return
        
        for url in wordlist_sources:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    words = response.text.lower().split()
                    # Clean and add words
                    for word in words:
                        cleaned = re.sub(r'[^\w]', '', word)
                        if len(cleaned) >= 2:
                            self.tagalog_wordlist.add(cleaned)
                    self.console.print(f"[green]✓ Loaded {len(self.tagalog_wordlist)} words from {url.split('/')[-2]}[/green]")
                    break
            except:
                continue
        
        if not self.tagalog_wordlist:
            self.console.print("[yellow]⚠️  Could not load online wordlists. Using corpus vocabulary only.[/yellow]")
    
    def is_valid_tagalog_word(self, word):
        """Check if word is in Tagalog wordlist"""
        word_lower = word.lower()
        return word_lower in self.tagalog_wordlist
    
    def add_corpus_vocabulary(self, vocab):
        """Add corpus vocabulary to validation"""
        self.tagalog_wordlist.update(vocab)

class MorphologicalAnalyzer:
    """Real Tagalog morphological analysis using tagalog-stemmer library"""
    
    def __init__(self, validator):
        self.validator = validator
        self.console = Console()
        
        if STEMMER_AVAILABLE:
            self.stemmer = TagalogStemmer()
            self.console.print("[green]✓ Tagalog stemmer loaded[/green]")
        else:
            self.stemmer = None
            self.console.print("[yellow]⚠️  Tagalog stemmer not available[/yellow]")
    
    def analyze_word(self, word):
        """
        Analyze a Tagalog word and return morphological information.
        Returns only real results from the stemmer library.
        """
        word_lower = word.lower()
        
        # Check if word is valid first
        if not self.validator.is_valid_tagalog_word(word_lower):
            return {
                'is_valid': False,
                'root': None,
                'variants': [],
                'affixes': []
            }
        
        # If stemmer is available, use it
        if self.stemmer:
            try:
                root = self.stemmer.stem(word_lower)
                
                # Only return root if it's different from the word
                if root and root != word_lower:
                    return {
                        'is_valid': True,
                        'root': root,
                        'variants': [root, word_lower],
                        'affixes': self._detect_affixes(word_lower, root)
                    }
                else:
                    # Word is its own root
                    return {
                        'is_valid': True,
                        'root': word_lower,
                        'variants': [],
                        'affixes': []
                    }
            except:
                pass
        
        # If stemmer not available or failed, return valid but no variants
        return {
            'is_valid': True,
            'root': word_lower,
            'variants': [],
            'affixes': []
        }
    
    def _detect_affixes(self, word, root):
        """Detect affixes by comparing word with root"""
        if not root or word == root:
            return []
        
        affixes = []
        
        # Find prefix
        if word.startswith(root):
            suffix = word[len(root):]
            if suffix:
                affixes.append(f"-{suffix}")
        elif word.endswith(root):
            prefix = word[:-len(root)]
            if prefix:
                affixes.append(f"{prefix}-")
        else:
            # Root is somewhere in the middle (infix case)
            idx = word.find(root)
            if idx > 0:
                prefix = word[:idx]
                affixes.append(f"{prefix}-")
            if idx + len(root) < len(word):
                suffix = word[idx + len(root):]
                affixes.append(f"-{suffix}")
        
        return affixes
    
    def find_corpus_variants(self, word, corpus_vocab):
        """
        Find actual morphological variants from the corpus.
        Only returns words that share the same root according to the stemmer.
        """
        if not self.stemmer:
            return []
        
        try:
            word_root = self.stemmer.stem(word.lower())
            if not word_root:
                return []
            
            variants = []
            for corpus_word in corpus_vocab:
                if corpus_word == word.lower():
                    continue
                try:
                    corpus_root = self.stemmer.stem(corpus_word)
                    if corpus_root and corpus_root == word_root:
                        variants.append(corpus_word)
                except:
                    continue
            
            return variants[:10]  # Limit to 10 variants
        except:
            return []

class EnhancedNoliSemanticSystem:
    def __init__(self):
        self.console = Console()
        
        self.console.print("🔄 Initializing Tagalog NLP components...", style="cyan")
        
        # Initialize validator and morphological analyzer
        self.validator = TagalogWordValidator()
        self.morph_analyzer = MorphologicalAnalyzer(self.validator)
        
        self.console.print("🔄 Loading XLM-RoBERTa multilingual model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        self.console.print("📚 Loading datasets...", style="cyan")
        self.chapters_df = pd.read_csv('chapters.csv')
        self.themes_df = pd.read_csv('themes.csv')
        
        self.chapters_df.columns = self.chapters_df.columns.str.strip()
        self.themes_df.columns = self.themes_df.columns.str.strip()
        
        # Build vocabulary and add to validator
        self.tagalog_vocab = self._build_tagalog_vocabulary()
        self.validator.add_corpus_vocabulary(self.tagalog_vocab)
        
        self.console.print("🧠 Computing chapter embeddings...", style="cyan")
        self._compute_chapter_embeddings()
        self.console.print("🎯 Computing theme embeddings...", style="cyan")
        self._compute_theme_embeddings()
        
        # Thresholds
        self.NONSENSE_THRESHOLD = 0.20
        self.EXACT_MATCH_THRESHOLD = 1.0
        self.MORPHOLOGICAL_MATCH_THRESHOLD = 0.80
        self.BOOK_RELEVANCE_THRESHOLD = 0.35
        self.THEMATIC_THRESHOLD = 0.45
        self.CONTEXT_RELEVANCE_THRESHOLD = 0.30
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.15
        self.EXACT_MATCH_BONUS = 0.30
        self.MORPHOLOGICAL_BONUS = 0.20
        self.CONTEXT_BONUS = 0.10
        self.MAX_CONTEXT_EXPANSION = 5
        
        self.used_context_sentences = set()
        
        self.console.print("✅ System ready with real Tagalog NLP!", style="bold green")
    
    def _build_tagalog_vocabulary(self):
        all_text = ' '.join(self.chapters_df['sentence_text'].astype(str))
        all_text += ' '.join(self.chapters_df['chapter_title'].astype(str))
        all_text += ' '.join(self.themes_df['Tagalog Title'].astype(str))
        all_text += ' '.join(self.themes_df['Meaning'].astype(str))
        
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]{2,}\b', all_text.lower())
        word_freq = Counter(words)
        vocab_set = set(word for word, freq in word_freq.items() if freq >= 2 and len(word) > 2)
        
        self.console.print(f"📖 Built vocabulary: {len(vocab_set)} words from corpus", style="dim cyan")
        return vocab_set
    
    def _compute_chapter_embeddings(self):
        self.chapters_df['combined_text'] = (
            self.chapters_df['chapter_title'].astype(str) + " " + 
            self.chapters_df['sentence_text'].astype(str)
        )
        
        self.chapters_df['sentence_word_count'] = self.chapters_df['sentence_text'].astype(str).apply(
            lambda x: len(x.split())
        )
        
        texts = self.chapters_df['combined_text'].tolist()
        self.chapter_embeddings = self.model.encode(texts, show_progress_bar=False)
    
    def _compute_theme_embeddings(self):
        self.themes_df['theme_text'] = (
            self.themes_df['Tagalog Title'].astype(str) + " " + 
            self.themes_df['Meaning'].astype(str)
        )
        
        theme_texts = self.themes_df['theme_text'].tolist()
        self.theme_embeddings = self.model.encode(theme_texts, show_progress_bar=False)
    
    def _is_real_word(self, word):
        word_lower = word.lower()
        
        # Check English
        if english_words and word_lower in english_words:
            return True
        
        # Check Tagalog using validator
        if self.validator.is_valid_tagalog_word(word_lower):
            return True
        
        # Check corpus
        if word_lower in self.tagalog_vocab:
            return True
        
        # Proper nouns
        if word[0].isupper() and 3 <= len(word) <= 20:
            return True
        
        # Very short words
        if len(word) <= 3:
            return True
            
        return False
    
    def _detect_language(self, text):
        if not text or len(text.strip()) < 2:
            return 'unknown'
            
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                detected = blob.detect_language()
                return detected if detected else 'unknown'
            except:
                pass
        
        english_indicators = sum(1 for word in text.lower().split() 
                               if english_words and word in english_words)
        tagalog_indicators = sum(1 for word in text.lower().split() 
                               if self.validator.is_valid_tagalog_word(word))
        
        total_words = len(text.split())
        
        if total_words == 0:
            return 'unknown'
        
        english_ratio = english_indicators / total_words
        tagalog_ratio = tagalog_indicators / total_words
        
        if english_ratio > 0.5:
            return 'en'
        elif tagalog_ratio > 0.3:
            return 'tl'
        else:
            return 'unknown'
    
    def _find_exact_matches(self, query):
        """Find exact string matches in the corpus"""
        query_lower = query.lower().strip()
        exact_matches = []
        
        for idx, row in self.chapters_df.iterrows():
            sentence_lower = str(row['sentence_text']).lower()
            
            if query_lower == sentence_lower.strip():
                exact_matches.append({
                    'index': idx,
                    'match_type': 'exact_full',
                    'confidence': 1.0
                })
            elif query_lower in sentence_lower:
                exact_matches.append({
                    'index': idx,
                    'match_type': 'exact_substring',
                    'confidence': 0.95
                })
        
        return exact_matches
    
    def _find_morphological_matches(self, query):
        """Find morphological variants using real stemmer"""
        query_words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query.lower())
        morphological_matches = []
        
        for query_word in query_words:
            # Analyze the word
            analysis = self.morph_analyzer.analyze_word(query_word)
            
            if not analysis['is_valid']:
                continue
            
            # Find corpus variants
            variants = self.morph_analyzer.find_corpus_variants(query_word, self.tagalog_vocab)
            
            if variants:
                # Search for these variants in the corpus
                for idx, row in self.chapters_df.iterrows():
                    sentence_lower = str(row['sentence_text']).lower()
                    sentence_words = set(re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', sentence_lower))
                    
                    matching_variants = set(variants) & sentence_words
                    if matching_variants and query_word not in sentence_words:
                        morphological_matches.append({
                            'index': idx,
                            'match_type': 'morphological',
                            'query_word': query_word,
                            'matched_variants': list(matching_variants),
                            'root': analysis.get('root', query_word),
                            'confidence': 0.85
                        })
        
        return morphological_matches
    
    def _calculate_confidence_with_adjustments(self, base_confidence, sentence_text, 
                                              match_type='semantic', has_relevant_context=False):
        adjusted_confidence = base_confidence
        
        if match_type == 'exact_full':
            adjusted_confidence = self.EXACT_MATCH_THRESHOLD
        elif match_type == 'exact_substring':
            adjusted_confidence = max(adjusted_confidence, 0.95)
        elif match_type == 'morphological':
            adjusted_confidence += self.MORPHOLOGICAL_BONUS
        
        word_count = len(str(sentence_text).split())
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (self.SHORT_SENTENCE_THRESHOLD - word_count) / self.SHORT_SENTENCE_THRESHOLD
            adjusted_confidence -= penalty
        
        if has_relevant_context:
            adjusted_confidence += self.CONTEXT_BONUS
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _is_nonsense_query(self, query):
        query = query.strip()
        
        if len(query) < 2:
            return True, 0.95, "Too short"
        
        nonsense_patterns = [
            (r'^[a-z]*\d+[a-z]*\d*$', 0.9, "Mixed alphanumeric"),
            (r'^[^a-zA-ZÀ-ÿñÑ\s]+$', 0.85, "Only symbols/numbers"),
            (r'^(.)\1{3,}', 0.8, "Repeated characters"),
            (r'^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+$', 0.7, "Only consonants"),
            (r'^\W+$', 0.9, "Only non-word characters")
        ]
        
        for pattern, conf, reason in nonsense_patterns:
            if re.match(pattern, query):
                return True, conf, reason
        
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        
        if not words:
            return True, 0.9, "No valid words found"
        
        valid_words = sum(1 for word in words if self._is_real_word(word))
        valid_ratio = valid_words / len(words)
        
        if valid_ratio < 0.6:
            return True, 0.9 - valid_ratio * 0.3, f"Only {valid_ratio:.1%} valid words"
        
        return False, valid_ratio, "Valid query"
    
    def _check_context_relevance(self, context_text, query, theme_context=None):
        try:
            query_embedding = self.model.encode([query])
            context_embedding = self.model.encode([context_text])
            query_similarity = cosine_similarity(query_embedding, context_embedding)[0][0]
            
            theme_similarity = 0.0
            if theme_context:
                theme_embedding = self.model.encode([theme_context])
                theme_similarity = cosine_similarity(theme_embedding, context_embedding)[0][0]
            
            max_similarity = max(query_similarity, theme_similarity)
            return max_similarity >= self.CONTEXT_RELEVANCE_THRESHOLD
        except:
            return False
    
    def _get_expanded_context_sentences(self, chapter_num, sentence_num, query, theme_context=None):
        context = {
            'prev_sentences': [],
            'next_sentences': [],
            'prev_relevant_count': 0,
            'next_relevant_count': 0
        }
        
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
        current_sentence_id = (chapter_num, sentence_num)
        
        for i in range(1, self.MAX_CONTEXT_EXPANSION + 1):
            if current_pos - i < 0:
                break
            
            prev_idx = chapter_list[current_pos - i]
            prev_row = self.chapters_df.loc[prev_idx]
            prev_sentence_id = (prev_row['chapter_number'], prev_row['sentence_number'])
            
            if prev_sentence_id in self.used_context_sentences:
                break
            
            is_relevant = self._check_context_relevance(
                prev_row['sentence_text'], query, theme_context
            )
            
            sentence_data = {
                'sentence_number': prev_row['sentence_number'],
                'sentence_text': prev_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i
            }
            
            context['prev_sentences'].append(sentence_data)
            
            if is_relevant:
                context['prev_relevant_count'] += 1
            else:
                break
        
        context['prev_sentences'].reverse()
        
        for i in range(1, self.MAX_CONTEXT_EXPANSION + 1):
            if current_pos + i >= len(chapter_list):
                break
            
            next_idx = chapter_list[current_pos + i]
            next_row = self.chapters_df.loc[next_idx]
            next_sentence_id = (next_row['chapter_number'], next_row['sentence_number'])
            
            if next_sentence_id in self.used_context_sentences:
                break
            
            is_relevant = self._check_context_relevance(
                next_row['sentence_text'], query, theme_context
            )
            
            sentence_data = {
                'sentence_number': next_row['sentence_number'],
                'sentence_text': next_row['sentence_text'],
                'is_relevant': is_relevant,
                'distance': i
            }
            
            context['next_sentences'].append(sentence_data)
            
            if is_relevant:
                context['next_relevant_count'] += 1
            else:
                break
        
        return context
    
    def _mark_context_as_used(self, chapter_num, sentence_num, context):
        self.used_context_sentences.add((chapter_num, sentence_num))
        
        for prev_sent in context.get('prev_sentences', []):
            self.used_context_sentences.add((chapter_num, prev_sent['sentence_number']))
        
        for next_sent in context.get('next_sentences', []):
            self.used_context_sentences.add((chapter_num, next_sent['sentence_number']))
    
    def _get_semantic_results(self, query, top_k=9):
        """Enhanced semantic search with real morphological analysis"""
        self.used_context_sentences = set()
        
        exact_matches = self._find_exact_matches(query)
        exact_indices = {match['index']: match for match in exact_matches}
        
        morphological_matches = self._find_morphological_matches(query)
        morph_indices = {}
        for match in morphological_matches:
            idx = match['index']
            if idx not in exact_indices:
                morph_indices[idx] = match
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
        
        results = []
        
        for idx, sim_score in enumerate(similarities):
            match_info = None
            match_type = 'semantic'
            base_confidence = sim_score
            
            if idx in exact_indices:
                match_info = exact_indices[idx]
                match_type = match_info['match_type']
                base_confidence = match_info['confidence']
            elif idx in morph_indices:
                match_info = morph_indices[idx]
                match_type = 'morphological'
                base_confidence = max(sim_score, match_info['confidence'])
            elif sim_score < self.BOOK_RELEVANCE_THRESHOLD:
                continue
            
            row = self.chapters_df.iloc[idx]
            
            try:
                context = self._get_expanded_context_sentences(
                    row['chapter_number'], 
                    row['sentence_number'], 
                    query
                )
                has_relevant_context = (context['prev_relevant_count'] > 0 or 
                                      context['next_relevant_count'] > 0)
            except:
                context = {'prev_sentences': [], 'next_sentences': [], 
                         'prev_relevant_count': 0, 'next_relevant_count': 0}
                has_relevant_context = False
            
            adjusted_confidence = self._calculate_confidence_with_adjustments(
                base_confidence, row['sentence_text'], match_type, has_relevant_context
            )
            
            result_data = {
                'index': idx,
                'chapter_number': row['chapter_number'],
                'chapter_title': row['chapter_title'],
                'sentence_number': row['sentence_number'],
                'sentence_text': row['sentence_text'],
                'confidence': adjusted_confidence,
                'match_type': match_type,
                'word_count': row['sentence_word_count'],
                'context': context,
                'has_relevant_context': has_relevant_context,
                'total_context_sentences': len(context['prev_sentences']) + len(context['next_sentences'])
            }
            
            if match_type == 'morphological' and match_info:
                result_data['morphological_info'] = {
                    'query_word': match_info['query_word'],
                    'matched_variants': match_info['matched_variants'],
                    'root': match_info.get('root', match_info['query_word'])
                }
            
            results.append(result_data)
        
        def sort_key(x):
            if x['match_type'] == 'exact_full':
                return (0, -x['confidence'])
            elif x['match_type'] == 'exact_substring':
                return (1, -x['confidence'])
            elif x['match_type'] == 'morphological':
                return (2, -x['confidence'])
            else:
                return (3, -x['confidence'])
        
        results.sort(key=sort_key)
        
        for result in results[:top_k]:
            self._mark_context_as_used(
                result['chapter_number'],
                result['sentence_number'],
                result['context']
            )
        
        chapter_groups = {}
        for result in results:
            ch_num = result['chapter_number']
            if ch_num not in chapter_groups:
                chapter_groups[ch_num] = []
            if len(chapter_groups[ch_num]) < 3:
                chapter_groups[ch_num].append(result)
        
        final_results = []
        seen_chapters = set()
        
        for result in results:
            ch_num = result['chapter_number']
            if ch_num not in seen_chapters or len([r for r in final_results if r['chapter_number'] == ch_num]) < 3:
                final_results.append(result)
                seen_chapters.add(ch_num)
                
                if len(final_results) >= top_k:
                    break
        
        return final_results[:top_k]
    
    def _get_thematic_classification(self, retrieved_sentences, query):
        if not retrieved_sentences:
            return retrieved_sentences, False, 0.0
        
        thematic_results = []
        
        for sentence_data in retrieved_sentences:
            sentence_text = sentence_data['sentence_text']
            context = sentence_data.get('context', {})
            
            sentence_embedding = self.model.encode([sentence_text])
            theme_similarities = cosine_similarity(sentence_embedding, self.theme_embeddings)[0]
            
            matching_themes = []
            for idx, similarity in enumerate(theme_similarities):
                if similarity >= self.THEMATIC_THRESHOLD:
                    theme_row = self.themes_df.iloc[idx]
                    theme_context = f"{theme_row['Tagalog Title']} {theme_row['Meaning']}"
                    
                    for prev_sent in context.get('prev_sentences', []):
                        if not prev_sent['is_relevant']:
                            prev_sent['is_relevant'] = self._check_context_relevance(
                                prev_sent['sentence_text'], query, theme_context
                            )
                    
                    for next_sent in context.get('next_sentences', []):
                        if not next_sent['is_relevant']:
                            next_sent['is_relevant'] = self._check_context_relevance(
                                next_sent['sentence_text'], query, theme_context
                            )
                    
                    matching_themes.append({
                        'tagalog_title': theme_row['Tagalog Title'],
                        'meaning': theme_row['Meaning'],
                        'theme_confidence': similarity
                    })
            
            matching_themes.sort(key=lambda x: x['theme_confidence'], reverse=True)
            
            enhanced_sentence = sentence_data.copy()
            
            if matching_themes:
                enhanced_sentence['themes'] = matching_themes[:2]
                enhanced_sentence['primary_theme'] = matching_themes[0]
                enhanced_sentence['has_theme'] = True
            else:
                enhanced_sentence['themes'] = []
                enhanced_sentence['primary_theme'] = None
                enhanced_sentence['has_theme'] = False
            
            context['prev_relevant_count'] = sum(1 for s in context.get('prev_sentences', []) if s['is_relevant'])
            context['next_relevant_count'] = sum(1 for s in context.get('next_sentences', []) if s['is_relevant'])
            enhanced_sentence['context'] = context
            enhanced_sentence['has_relevant_context'] = (context['prev_relevant_count'] > 0 or 
                                                        context['next_relevant_count'] > 0)
            
            thematic_results.append(enhanced_sentence)
        
        sentences_with_themes = sum(1 for s in thematic_results if s['has_theme'])
        thematic_coverage = sentences_with_themes / len(thematic_results) if thematic_results else 0
        
        avg_theme_confidence = 0.0
        if sentences_with_themes > 0:
            theme_confidences = [s['primary_theme']['theme_confidence'] 
                               for s in thematic_results if s['has_theme']]
            avg_theme_confidence = np.mean(theme_confidences)
        
        has_strong_thematic_connection = thematic_coverage >= 0.3 and avg_theme_confidence >= self.THEMATIC_THRESHOLD
        
        return thematic_results, has_strong_thematic_connection, avg_theme_confidence
    
    def query(self, user_query):
        is_nonsense, confidence, reason = self._is_nonsense_query(user_query)
        
        if is_nonsense:
            return {
                'type': 'nonsense',
                'confidence': confidence,
                'reason': reason,
                'message': f"Query classified as nonsense: {reason} (confidence: {confidence:.1%})"
            }
        
        semantic_results = self._get_semantic_results(user_query)
        
        if not semantic_results:
            return {
                'type': 'no_matches',
                'message': "No high-confidence matches found in Noli Me Tangere"
            }
        
        thematic_results, has_themes, avg_theme_conf = self._get_thematic_classification(semantic_results, user_query)
        
        book_relevance = np.mean([r['confidence'] for r in semantic_results])
        exact_matches = sum(1 for r in semantic_results if r['match_type'].startswith('exact'))
        morphological_matches = sum(1 for r in semantic_results if r['match_type'] == 'morphological')
        context_matches = sum(1 for r in thematic_results if r['has_relevant_context'])
        total_context_sentences = sum(r.get('total_context_sentences', 0) for r in thematic_results)
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'results': thematic_results,
            'book_confidence': book_relevance,
            'theme_confidence': avg_theme_conf,
            'exact_matches': exact_matches,
            'morphological_matches': morphological_matches,
            'context_matches': context_matches,
            'total_results': len(thematic_results),
            'avg_confidence': book_relevance,
            'total_context_sentences': total_context_sentences
        }
    
    def display_results(self, response, query=""):
        result_type = response['type']
        
        if result_type != 'success':
            if result_type == 'nonsense':
                table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
                table.add_column("Result", style="bold red", justify="center")
                table.add_row("none")
                self.console.print(table)
                
                info_panel = Panel(
                    f"🚫 {response['message']}",
                    title="Nonsense Query Detected",
                    style="red",
                    box=box.ROUNDED
                )
                self.console.print(info_panel)
                return
            else:
                error_panel = Panel(
                    f"❌ {response['message']}",
                    title="No Results",
                    style="yellow",
                    box=box.ROUNDED
                )
                self.console.print(error_panel)
                return
        
        results = response['results']
        subtype = response['subtype']
        has_themes = subtype == 'thematic'
        
        metrics_text = (
            f"📊 Book Relevance: {response['book_confidence']:.1%} | "
            f"Exact: {response['exact_matches']} | "
            f"Morphological: {response.get('morphological_matches', 0)} | "
            f"Context: {response.get('context_matches', 0)} | "
            f"Total Context: {response.get('total_context_sentences', 0)} | "
            f"Results: {response['total_results']}"
        )
        if response['theme_confidence'] > 0:
            metrics_text += f"\n🎭 Thematic Confidence: {response['theme_confidence']:.1%}"
        
        header_text = Text(f"🎯 Enhanced Results for: '{query}'\n{metrics_text}", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(header_panel)
        
        for i, result in enumerate(results, 1):
            self.console.print(f"\n📍 **Result {i}**", style="bold cyan")
            
            main_table = Table(
                show_header=True,
                header_style="bold magenta",
                border_style="bright_blue",
                box=box.ROUNDED,
                padding=(0, 1),
                expand=True
            )
            
            main_table.add_column("📖 Book", style="cyan", width=12, no_wrap=True)
            main_table.add_column("📚 Ch", style="bright_green", width=6, justify="center")
            main_table.add_column("📝 S#", style="yellow", width=6, justify="center")
            main_table.add_column("🎯 Conf", style="bright_cyan", width=8, justify="center")
            main_table.add_column("✅ Match Type", style="bright_yellow", width=15, justify="center")
            main_table.add_column("🔄 Context", style="bright_white", width=12, justify="center")
            
            confidence_str = f"{result['confidence']:.1%}"
            
            match_type = result['match_type']
            if match_type == 'exact_full':
                match_display = "🎯 Exact Full"
            elif match_type == 'exact_substring':
                match_display = "🎯 Exact Sub"
            elif match_type == 'morphological':
                match_display = "🔤 Morphological"
            else:
                match_display = "📊 Semantic"
            
            context_count = result.get('total_context_sentences', 0)
            context_status = f"{context_count} sent" if context_count > 0 else "None"
            
            main_table.add_row(
                "Noli Me Tangere",
                str(result['chapter_number']),
                str(result['sentence_number']),
                confidence_str,
                match_display,
                context_status
            )
            
            self.console.print(main_table)
            
            # Show morphological information with root
            if match_type == 'morphological' and 'morphological_info' in result:
                morph_info = result['morphological_info']
                
                # Display morphological match info
                morph_text = Text()
                morph_text.append(f"Query word '{morph_info['query_word']}' → ", style="yellow")
                
                if morph_info['matched_variants']:
                    morph_text.append(f"Found variants: {', '.join(morph_info['matched_variants'])}", style="bright_yellow")
                else:
                    morph_text.append("Found variants: none", style="dim yellow")
                
                morph_panel = Panel(
                    morph_text,
                    style="yellow",
                    box=box.SIMPLE
                )
                self.console.print(morph_panel)
                
                # Display root in a box
                root_table = Table(show_header=False, box=box.HEAVY, border_style="yellow", padding=(0, 2))
                root_table.add_column("Root", style="bold yellow", justify="center")
                root_table.add_row(f"root: {morph_info['root']}")
                self.console.print(root_table)
            
            chapter_panel = Panel(
                f"📑 **{result['chapter_title']}**",
                style="bright_green",
                box=box.SIMPLE
            )
            self.console.print(chapter_panel)
            
            content_text = Text()
            content_text.append("💬 ", style="bold blue")
            content_text.append(result['sentence_text'], style="white")
            
            content_panel = Panel(
                content_text,
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
                    title=f"📋 Expanded Context ({len(prev_sentences) + len(next_sentences)} sentences)",
                    show_header=True,
                    header_style="bold yellow",
                    border_style="yellow",
                    box=box.SIMPLE,
                    expand=True
                )
                
                context_table.add_column("Position", style="yellow", width=12)
                context_table.add_column("S#", style="bright_yellow", width=6, justify="center")
                context_table.add_column("Dist", style="dim yellow", width=6, justify="center")
                context_table.add_column("Relevant", style="bright_green", width=10, justify="center")
                context_table.add_column("Content", style="white", min_width=40)
                
                for prev_sent in prev_sentences:
                    relevance_icon = "✓" if prev_sent['is_relevant'] else "○"
                    context_table.add_row(
                        "⬆️ Previous",
                        str(prev_sent['sentence_number']),
                        f"-{prev_sent['distance']}",
                        relevance_icon,
                        prev_sent['sentence_text']
                    )
                
                for next_sent in next_sentences:
                    relevance_icon = "✓" if next_sent['is_relevant'] else "○"
                    context_table.add_row(
                        "⬇️ Next",
                        str(next_sent['sentence_number']),
                        f"+{next_sent['distance']}",
                        relevance_icon,
                        next_sent['sentence_text']
                    )
                
                self.console.print(context_table)
            
            if has_themes and result.get('has_theme'):
                primary_theme = result['primary_theme']
                
                theme_table = Table(
                    title="🎭 Thematic Analysis",
                    show_header=True,
                    header_style="bold magenta",
                    border_style="magenta",
                    box=box.SIMPLE,
                    expand=True
                )
                
                theme_table.add_column("🏷️ Tagalog Title", style="bright_magenta", width=25)
                theme_table.add_column("📖 Meaning", style="magenta", min_width=40)
                theme_table.add_column("🎯 Theme Conf", style="bright_cyan", width=12, justify="center")
                
                theme_table.add_row(
                    primary_theme['tagalog_title'],
                    primary_theme['meaning'],
                    f"{primary_theme['theme_confidence']:.1%}"
                )
                
                self.console.print(theme_table)
            
            if i < len(results):
                self.console.print("─" * 100, style="dim blue")
        
        chapters_found = len(set(r['chapter_number'] for r in results))
        exact_count = sum(1 for r in results if r['match_type'].startswith('exact'))
        morph_count = sum(1 for r in results if r['match_type'] == 'morphological')
        context_count = sum(1 for r in results if r.get('has_relevant_context', False))
        theme_count = sum(1 for r in results if r.get('has_theme', False))
        total_context = response.get('total_context_sentences', 0)
        
        classification = "🎭 Thematic Analysis" if has_themes else "📚 Semantic Search"
        
        summary_parts = [
            f"{classification}",
            f"{len(results)} sentences from {chapters_found} chapters",
            f"{exact_count} exact",
            f"{morph_count} morphological",
            f"{context_count} with context",
            f"{total_context} total context sentences"
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
    
    def analyze_word_morphology(self, word):
        """Standalone word analysis with proper formatting"""
        analysis = self.morph_analyzer.analyze_word(word)
        
        if not analysis['is_valid']:
            # Show "none" result
            result_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            result_table.add_column("Result", style="bold red", justify="center")
            result_table.add_row("none")
            self.console.print(result_table)
            
            # Show root as "none"
            root_table = Table(show_header=False, box=box.HEAVY, border_style="red", padding=(0, 2))
            root_table.add_column("Root", style="bold red", justify="center")
            root_table.add_row("none")
            self.console.print(root_table)
            
            # Calculate valid word percentage
            words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', word)
            valid_words = sum(1 for w in words if self._is_real_word(w))
            valid_ratio = valid_words / len(words) if words else 0
            
            info_panel = Panel(
                f"🚫 Query classified as nonsense: Only {valid_ratio:.1%} valid words (confidence: {1-valid_ratio:.1%})",
                title="Nonsense Query Detected",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(info_panel)
            return
        
        # Find corpus variants
        corpus_variants = self.morph_analyzer.find_corpus_variants(word, self.tagalog_vocab)
        
        # Display variants
        variant_text = Text()
        variant_text.append(f"Query word '{word}' → ", style="yellow")
        
        if corpus_variants:
            variant_text.append(f"Found variants: {', '.join(corpus_variants[:10])}", style="bright_yellow")
        else:
            variant_text.append("Found variants: none", style="dim yellow")
        
        variant_panel = Panel(
            variant_text,
            style="yellow",
            box=box.SIMPLE
        )
        self.console.print(variant_panel)
        
        # Display root in box
        root_table = Table(show_header=False, box=box.HEAVY, border_style="yellow", padding=(0, 2))
        root_table.add_column("Root", style="bold yellow", justify="center")
        root_table.add_row(f"root: {analysis['root']}")
        self.console.print(root_table)
        
        # Display affixes if any
        if analysis.get('affixes'):
            affix_text = Text()
            affix_text.append("Affixes: ", style="bold cyan")
            affix_text.append(", ".join(analysis['affixes']), style="cyan")
            self.console.print(affix_text)

if __name__ == "__main__":
    system = EnhancedNoliSemanticSystem()
    
    welcome_panel = Panel(
        Align.center(Text(
            "📚 Enhanced Noli Me Tangere Semantic System\n"
            "🎯 XLM-RoBERTa with Real Tagalog NLP\n"
            "✅ Exact Match Priority | 🔤 Real Morphological Analysis | 🎭 Thematic Analysis\n"
            "🔄 Non-Overlapping Context | 📊 Multi-Level Matching\n\n"
            "Commands:\n"
            "- Enter a query to search in Noli Me Tangere\n"
            "- Type 'analyze: <word>' to analyze Tagalog morphology\n"
            "- Type 'exit' to quit",
            style="bold white"
        )),
        style="bright_green",
        box=box.HEAVY
    )
    system.console.print(welcome_panel)
    
    while True:
        system.console.print("\n" + "─" * 80, style="dim")
        user_input = system.console.input("[bold cyan]Enter query or command (or 'exit' to quit): [/bold cyan]").strip()
        
        if user_input.lower() == 'exit':
            goodbye_panel = Panel(
                Align.center(Text("Thank you for using the Enhanced Noli System!", style="bold green")),
                style="bright_green",
                box=box.ROUNDED
            )
            system.console.print(goodbye_panel)
            break
        
        if not user_input:
            system.console.print("[red]Please enter a valid query.[/red]")
            continue
        
        # Check for word analysis command
        if user_input.lower().startswith('analyze:'):
            word = user_input[8:].strip()
            if word:
                system.console.print(f"[dim]Analyzing '{word}'...[/dim]")
                try:
                    system.analyze_word_morphology(word)
                except Exception as e:
                    error_panel = Panel(
                        f"Analysis error: {str(e)}",
                        title="Error",
                        style="red",
                        box=box.ROUNDED
                    )
                    system.console.print(error_panel)
            else:
                system.console.print("[red]Please provide a word to analyze.[/red]")
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