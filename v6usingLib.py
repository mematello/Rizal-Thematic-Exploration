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
from tglstemmer import stem
from wordfreq import word_frequency

warnings.filterwarnings('ignore')

class StemmedWordAnalyzer:
    """Handles Filipino word stemming and frequency analysis"""
    
    def __init__(self):
        self.stem_cache = {}
        self.MIN_FILIPINO_FREQUENCY = 1e-8  # Minimum frequency threshold for valid Filipino words
        self.MIN_VALID_WORD_RATIO = 0.5  # At least 50% of words should be valid Filipino
    
    def get_stem(self, word):
        """Get the stem of a Filipino word using TglStemmer"""
        word_lower = word.lower()
        
        if word_lower in self.stem_cache:
            return self.stem_cache[word_lower]
        
        try:
            stemmed = stem(word_lower)
            self.stem_cache[word_lower] = stemmed
            return stemmed
        except Exception:
            # If stemming fails, return original word
            self.stem_cache[word_lower] = word_lower
            return word_lower
    
    def get_word_frequency(self, word, lang='tl'):
        """Get word frequency using wordfreq library"""
        try:
            freq = word_frequency(word.lower(), lang)
            return freq
        except Exception:
            return 0.0
    
    def is_valid_filipino_word(self, word):
        """Check if a word is a valid Filipino word using wordfreq"""
        if len(word) < 2:
            return False
        
        # Check frequency in Filipino
        freq = self.get_word_frequency(word, 'tl')
        
        # Also check the stemmed version
        stemmed = self.get_stem(word)
        freq_stemmed = self.get_word_frequency(stemmed, 'tl')
        
        # Valid if either original or stemmed form has frequency above threshold
        return (freq >= self.MIN_FILIPINO_FREQUENCY or 
                freq_stemmed >= self.MIN_FILIPINO_FREQUENCY)
    
    def validate_filipino_query(self, query):
        """
        Validate if the query contains valid Filipino words.
        Returns: (is_valid, validation_info)
        """
        # Extract words from query
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
                validation_info['reason'] = 'No valid Filipino words detected - appears to be gibberish or non-Filipino text'
            else:
                validation_info['reason'] = f'Only {valid_ratio:.1%} of words are valid Filipino (need at least {self.MIN_VALID_WORD_RATIO:.0%})'
        
        return is_valid, validation_info
        """Get word frequency using wordfreq library"""
        try:
            freq = word_frequency(word.lower(), lang)
            return freq
        except Exception:
            return 0.0
    
    def analyze_query_words(self, query):
        """Analyze all words in query: stem them and get frequencies"""
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        analysis = []
        
        for word in words:
            stemmed = self.get_stem(word)
            freq_original = self.get_word_frequency(word, 'tl')
            freq_stemmed = self.get_word_frequency(stemmed, 'tl')
            
            analysis.append({
                'original': word,
                'stemmed': stemmed,
                'freq_original': freq_original,
                'freq_stemmed': freq_stemmed,
                'is_stemmed': word.lower() != stemmed
            })
        
        return analysis
    
    def find_stem_matches(self, query, corpus_df):
        """Find sentences containing stems of query words"""
        query_analysis = self.analyze_query_words(query)
        query_stems = set(item['stemmed'] for item in query_analysis)
        
        stem_matches = []
        
        for idx, row in corpus_df.iterrows():
            sentence_text = str(row['sentence_text']).lower()
            sentence_words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', sentence_text)
            
            # Stem all words in sentence
            sentence_stems = set(self.get_stem(word) for word in sentence_words)
            
            # Find matching stems
            matching_stems = query_stems & sentence_stems
            
            if matching_stems:
                # Calculate match strength based on frequency
                match_strength = sum(
                    self.get_word_frequency(stem, 'tl') 
                    for stem in matching_stems
                )
                
                stem_matches.append({
                    'index': idx,
                    'matching_stems': list(matching_stems),
                    'match_strength': match_strength,
                    'num_matches': len(matching_stems)
                })
        
        return stem_matches

class EnhancedNoliSemanticSystem:
    def __init__(self):
        self.console = Console()
        
        self.console.print("🔄 Loading XLM-RoBERTa multilingual model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        self.console.print("📚 Loading datasets...", style="cyan")
        self.chapters_df = pd.read_csv('chapters.csv')
        self.themes_df = pd.read_csv('themes.csv')
        
        self.chapters_df.columns = self.chapters_df.columns.str.strip()
        self.themes_df.columns = self.themes_df.columns.str.strip()
        
        # Initialize stemmer analyzer
        self.console.print("🔤 Initializing TglStemmer analyzer...", style="cyan")
        self.stem_analyzer = StemmedWordAnalyzer()
        
        self.console.print("🧠 Computing chapter embeddings...", style="cyan")
        self._compute_chapter_embeddings()
        self.console.print("🎯 Computing theme embeddings...", style="cyan")
        self._compute_theme_embeddings()
        
        # Thresholds
        self.EXACT_MATCH_THRESHOLD = 1.0
        self.STEM_MATCH_THRESHOLD = 0.80
        self.BOOK_RELEVANCE_THRESHOLD = 0.35
        self.THEMATIC_THRESHOLD = 0.45
        self.CONTEXT_RELEVANCE_THRESHOLD = 0.30
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.15
        self.EXACT_MATCH_BONUS = 0.30
        self.STEM_MATCH_BONUS = 0.20
        self.CONTEXT_BONUS = 0.10
        self.MAX_CONTEXT_EXPANSION = 5
        
        self.used_context_sentences = set()
        
        self.console.print("✅ System with TglStemmer ready!", style="bold green")
    
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
    
    def _calculate_confidence_with_adjustments(self, base_confidence, sentence_text, 
                                              match_type='semantic', has_relevant_context=False):
        adjusted_confidence = base_confidence
        
        if match_type == 'exact_full':
            adjusted_confidence = self.EXACT_MATCH_THRESHOLD
        elif match_type == 'exact_substring':
            adjusted_confidence = max(adjusted_confidence, 0.95)
        elif match_type == 'stem_match':
            adjusted_confidence += self.STEM_MATCH_BONUS
        
        word_count = len(str(sentence_text).split())
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (self.SHORT_SENTENCE_THRESHOLD - word_count) / self.SHORT_SENTENCE_THRESHOLD
            adjusted_confidence -= penalty
        
        if has_relevant_context:
            adjusted_confidence += self.CONTEXT_BONUS
        
        return max(0.0, min(1.0, adjusted_confidence))
    
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
        
        # Expand backward
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
        
        # Expand forward
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
        """Enhanced semantic search with exact and stem match prioritization"""
        self.used_context_sentences = set()
        
        # Priority 1: Exact matches
        exact_matches = self._find_exact_matches(query)
        exact_indices = {match['index']: match for match in exact_matches}
        
        # Priority 2: Stem matches
        stem_matches = self.stem_analyzer.find_stem_matches(query, self.chapters_df)
        stem_indices = {}
        for match in stem_matches:
            idx = match['index']
            if idx not in exact_indices:
                stem_indices[idx] = match
        
        # Priority 3: Semantic matches
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
            elif idx in stem_indices:
                match_info = stem_indices[idx]
                match_type = 'stem_match'
                base_confidence = max(sim_score, self.STEM_MATCH_THRESHOLD)
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
            
            if match_type == 'stem_match' and match_info:
                result_data['stem_info'] = {
                    'matching_stems': match_info['matching_stems'],
                    'match_strength': match_info['match_strength'],
                    'num_matches': match_info['num_matches']
                }
            
            results.append(result_data)
        
        def sort_key(x):
            if x['match_type'] == 'exact_full':
                return (0, -x['confidence'])
            elif x['match_type'] == 'exact_substring':
                return (1, -x['confidence'])
            elif x['match_type'] == 'stem_match':
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
        # First, validate if the query contains valid Filipino words
        is_valid, validation_info = self.stem_analyzer.validate_filipino_query(user_query)
        
        if not is_valid:
            return {
                'type': 'invalid_filipino',
                'validation_info': validation_info,
                'message': f"Invalid Filipino query: {validation_info['reason']}"
            }
        
        # Analyze query words for stemming and frequency
        query_analysis = self.stem_analyzer.analyze_query_words(user_query)
        
        semantic_results = self._get_semantic_results(user_query)
        
        if not semantic_results:
            return {
                'type': 'no_matches',
                'message': "No high-confidence matches found in Noli Me Tangere",
                'query_analysis': query_analysis
            }
        
        thematic_results, has_themes, avg_theme_conf = self._get_thematic_classification(semantic_results, user_query)
        
        book_relevance = np.mean([r['confidence'] for r in semantic_results])
        exact_matches = sum(1 for r in semantic_results if r['match_type'].startswith('exact'))
        stem_matches = sum(1 for r in semantic_results if r['match_type'] == 'stem_match')
        context_matches = sum(1 for r in thematic_results if r['has_relevant_context'])
        total_context_sentences = sum(r.get('total_context_sentences', 0) for r in thematic_results)
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'results': thematic_results,
            'book_confidence': book_relevance,
            'theme_confidence': avg_theme_conf,
            'exact_matches': exact_matches,
            'stem_matches': stem_matches,
            'context_matches': context_matches,
            'total_results': len(thematic_results),
            'avg_confidence': book_relevance,
            'total_context_sentences': total_context_sentences,
            'query_analysis': query_analysis
        }
    
    def display_results(self, response, query=""):
        result_type = response['type']
        
        # Handle invalid Filipino query
        if result_type == 'invalid_filipino':
            validation_info = response['validation_info']
            
            # Display "none" table
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)
            
            # Display validation details
            validation_panel = Panel(
                f"🚫 {response['message']}\n\n"
                f"📊 Analysis:\n"
                f"   • Total words: {validation_info['total_words']}\n"
                f"   • Valid Filipino words: {validation_info['valid_words']}\n"
                f"   • Invalid/Gibberish words: {', '.join(validation_info['invalid_words']) if validation_info['invalid_words'] else 'N/A'}\n"
                f"   • Valid ratio: {validation_info['valid_ratio']:.1%}\n\n"
                f"💡 Please enter a valid Filipino (Tagalog) query.",
                title="❌ Invalid Filipino Query",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(validation_panel)
            return
        
        if result_type != 'success':
            error_panel = Panel(
                f"❌ {response['message']}",
                title="No Results",
                style="yellow",
                box=box.ROUNDED
            )
            self.console.print(error_panel)
            
            # Display query analysis even on no matches
            if 'query_analysis' in response:
                self._display_query_analysis(response['query_analysis'])
            return
        
        # Display query analysis
        if 'query_analysis' in response:
            self._display_query_analysis(response['query_analysis'])
        
        results = response['results']
        subtype = response['subtype']
        has_themes = subtype == 'thematic'
        
        metrics_text = (
            f"📊 Book Relevance: {response['book_confidence']:.1%} | "
            f"Exact: {response['exact_matches']} | "
            f"Stem: {response.get('stem_matches', 0)} | "
            f"Context: {response.get('context_matches', 0)} | "
            f"Total Context: {response.get('total_context_sentences', 0)} | "
            f"Results: {response['total_results']}"
        )
        if response['theme_confidence'] > 0:
            metrics_text += f"\n🎭 Thematic Confidence: {response['theme_confidence']:.1%}"
        
        header_text = Text(f"🎯 Results for: '{query}'\n{metrics_text}", style="bold white")
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
            elif match_type == 'stem_match':
                match_display = "🌱 Stem Match"
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
            
            if match_type == 'stem_match' and 'stem_info' in result:
                stem_info = result['stem_info']
                stem_text = Text()
                stem_text.append("🌱 Stem Match: ", style="bold green")
                stem_text.append(f"Matching stems: {', '.join(stem_info['matching_stems'])} ", style="green")
                stem_text.append(f"(strength: {stem_info['match_strength']:.4f})", style="dim green")
                
                stem_panel = Panel(
                    stem_text,
                    style="green",
                    box=box.SIMPLE
                )
                self.console.print(stem_panel)
            
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
        stem_count = sum(1 for r in results if r['match_type'] == 'stem_match')
        context_count = sum(1 for r in results if r.get('has_relevant_context', False))
        theme_count = sum(1 for r in results if r.get('has_theme', False))
        total_context = response.get('total_context_sentences', 0)
        
        classification = "🎭 Thematic Analysis" if has_themes else "📚 Semantic Search"
        
        summary_parts = [
            f"{classification}",
            f"{len(results)} sentences from {chapters_found} chapters",
            f"{exact_count} exact",
            f"{stem_count} stem",
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
    
    def _display_query_analysis(self, query_analysis):
        """Display stemming and frequency analysis for query words"""
        if not query_analysis:
            return
        
        analysis_table = Table(
            title="🔍 Query Word Analysis (TglStemmer + WordFreq)",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True
        )
        
        analysis_table.add_column("Original Word", style="bright_white", width=20)
        analysis_table.add_column("Stemmed Form", style="bright_green", width=20)
        analysis_table.add_column("Freq (Original)", style="bright_yellow", width=15, justify="right")
        analysis_table.add_column("Freq (Stemmed)", style="bright_yellow", width=15, justify="right")
        analysis_table.add_column("Status", style="bright_cyan", width=15, justify="center")
        
        for item in query_analysis:
            status = "🌱 Stemmed" if item['is_stemmed'] else "✓ Root"
            
            analysis_table.add_row(
                item['original'],
                item['stemmed'],
                f"{item['freq_original']:.6f}",
                f"{item['freq_stemmed']:.6f}",
                status
            )
        
        self.console.print("\n")
        self.console.print(analysis_table)
        self.console.print("\n")

if __name__ == "__main__":
    system = EnhancedNoliSemanticSystem()
    
    welcome_panel = Panel(
        Align.center(Text(
            "📚 Noli Me Tangere Semantic System with TglStemmer\n"
            "🎯 XLM-RoBERTa + Filipino Stemming + Word Frequency Analysis\n"
            "✅ Exact Match Priority | 🌱 Stem Matching | 🎭 Thematic Analysis\n"
            "🔄 Non-Overlapping Context | 📊 Multi-Level Matching",
            style="bold white"
        )),
        style="bright_green",
        box=box.HEAVY
    )
    system.console.print(welcome_panel)
    
    while True:
        system.console.print("\n" + "─" * 80, style="dim")
        user_input = system.console.input("[bold cyan]Enter query (or 'exit' to quit): [/bold cyan]").strip()
        
        if user_input.lower() == 'exit':
            goodbye_panel = Panel(
                Align.center(Text("Thank you for using the Noli System!", style="bold green")),
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