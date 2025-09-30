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

class EnhancedNoliSemanticSystem:
    def __init__(self):
        self.console = Console()
        
        self.console.print("Building vocabulary from datasets...", style="cyan")
        
        self.console.print("Loading XLM-RoBERTa multilingual model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        self.console.print("Loading datasets...", style="cyan")
        self.chapters_df = pd.read_csv('chapters.csv')
        self.themes_df = pd.read_csv('themes.csv')
        
        self.chapters_df.columns = self.chapters_df.columns.str.strip()
        self.themes_df.columns = self.themes_df.columns.str.strip()
        
        self.tagalog_vocab = self._build_tagalog_vocabulary()
        
        self.console.print("Computing chapter embeddings...", style="cyan")
        self._compute_chapter_embeddings()
        self.console.print("Computing theme embeddings...", style="cyan")
        self._compute_theme_embeddings()
        
        # Thresholds
        self.NONSENSE_THRESHOLD = 0.20
        self.BOOK_RELEVANCE_THRESHOLD = 0.35
        self.THEMATIC_THRESHOLD = 0.45
        self.CONTEXT_RELEVANCE_THRESHOLD = 0.30
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.15
        self.EXACT_MATCH_BONUS = 0.25
        self.CONTEXT_BONUS = 0.10
        self.HIGH_CONTEXT_BONUS = 0.20  # Bonus when surrounding sentences are also highly relevant
        
        self.console.print("Enhanced system ready!", style="bold green")
    
    def _build_tagalog_vocabulary(self):
        """Build comprehensive Tagalog vocabulary from the novel"""
        all_text = ' '.join(self.chapters_df['sentence_text'].astype(str))
        all_text += ' '.join(self.chapters_df['chapter_title'].astype(str))
        all_text += ' '.join(self.themes_df['Tagalog Title'].astype(str))
        all_text += ' '.join(self.themes_df['Meaning'].astype(str))
        
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]{2,}\b', all_text.lower())
        word_freq = Counter(words)
        
        vocab_set = set(word for word, freq in word_freq.items() if freq >= 2 and len(word) > 2)
        
        self.console.print(f"Built vocabulary: {len(vocab_set)} Tagalog words from dataset", style="dim cyan")
        return vocab_set
    
    def _compute_chapter_embeddings(self):
        """Precompute embeddings for all sentences with enhanced context"""
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
        """Precompute theme embeddings with enhanced context"""
        self.themes_df['theme_text'] = (
            self.themes_df['Tagalog Title'].astype(str) + " " + 
            self.themes_df['Meaning'].astype(str)
        )
        
        theme_texts = self.themes_df['theme_text'].tolist()
        self.theme_embeddings = self.model.encode(theme_texts, show_progress_bar=False)
    
    def _is_real_word(self, word):
        """Enhanced word validation using available libraries only"""
        word_lower = word.lower()
        
        if english_words and word_lower in english_words:
            return True
        
        if word_lower in self.tagalog_vocab:
            return True
        
        if word[0].isupper() and 3 <= len(word) <= 20:
            return True
        
        if len(word) <= 3:
            return True
            
        return False
    
    def _detect_language(self, text):
        """Simple language detection using available tools"""
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
                               if word in self.tagalog_vocab)
        
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
    
    def _is_nonsense_query(self, query):
        """RULE 1: Enhanced nonsense detection with real word validation"""
        query = query.strip()
        
        if len(query) < 2:
            return True, 0.95, "Too short"
        
        nonsense_patterns = [
    (r'^[a-z]*\d+[a-z]*\d*', 0.9, "Mixed alphanumeric"),      
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
        
        valid_words = 0
        for word in words:
            if self._is_real_word(word):
                valid_words += 1
        
        valid_ratio = valid_words / len(words)
        
        if valid_ratio < 0.6:
            return True, 0.9 - valid_ratio * 0.3, f"Only {valid_ratio:.1%} valid words"
        
        detected_lang = self._detect_language(query)
        if detected_lang not in ['en', 'tl', 'unknown']:
            return True, 0.8, f"Detected language: {detected_lang}"
        
        return False, valid_ratio, "Valid query"
    
    def _has_exact_match(self, query):
        """RULE 3: Check for exact matches in sentence text"""
        query_lower = query.lower()
        
        exact_matches = []
        for idx, row in self.chapters_df.iterrows():
            sentence_lower = str(row['sentence_text']).lower()
            if query_lower in sentence_lower:
                exact_matches.append({
                    'index': idx,
                    'is_exact': query_lower == sentence_lower.strip(),
                    'is_substring': query_lower in sentence_lower
                })
        
        return exact_matches
    
    def _get_surrounding_sentences(self, chapter_num, sentence_num, query, query_embedding, max_extend=5):
        """
        NEW: Get expanding context - checks prev/next sentences and extends if still relevant.
        Returns list of relevant surrounding sentences with their confidence scores.
        """
        surrounding = []
        
        # Get all sentences in the same chapter, sorted
        chapter_sentences = self.chapters_df[
            self.chapters_df['chapter_number'] == chapter_num
        ].sort_values('sentence_number')
        
        if chapter_sentences.empty:
            return surrounding
        
        # Find current sentence position
        current_idx = None
        for idx, row in chapter_sentences.iterrows():
            if row['sentence_number'] == sentence_num:
                current_idx = idx
                break
        
        if current_idx is None:
            return surrounding
        
        chapter_list = chapter_sentences.index.tolist()
        current_pos = chapter_list.index(current_idx)
        
        # Check previous sentences (expanding backwards)
        for offset in range(1, max_extend + 1):
            if current_pos - offset < 0:
                break
            
            prev_idx = chapter_list[current_pos - offset]
            prev_row = self.chapters_df.loc[prev_idx]
            prev_text = prev_row['sentence_text']
            
            # Calculate relevance
            prev_embedding = self.chapter_embeddings[prev_idx]
            similarity = cosine_similarity(query_embedding.reshape(1, -1), 
                                         prev_embedding.reshape(1, -1))[0][0]
            
            if similarity >= self.CONTEXT_RELEVANCE_THRESHOLD:
                surrounding.append({
                    'position': 'before',
                    'offset': offset,
                    'sentence_number': prev_row['sentence_number'],
                    'sentence_text': prev_text,
                    'confidence': similarity,
                    'index': prev_idx
                })
            else:
                # Stop extending backwards if not relevant
                break
        
        # Check next sentences (expanding forwards)
        for offset in range(1, max_extend + 1):
            if current_pos + offset >= len(chapter_list):
                break
            
            next_idx = chapter_list[current_pos + offset]
            next_row = self.chapters_df.loc[next_idx]
            next_text = next_row['sentence_text']
            
            # Calculate relevance
            next_embedding = self.chapter_embeddings[next_idx]
            similarity = cosine_similarity(query_embedding.reshape(1, -1), 
                                         next_embedding.reshape(1, -1))[0][0]
            
            if similarity >= self.CONTEXT_RELEVANCE_THRESHOLD:
                surrounding.append({
                    'position': 'after',
                    'offset': offset,
                    'sentence_number': next_row['sentence_number'],
                    'sentence_text': next_text,
                    'confidence': similarity,
                    'index': next_idx
                })
            else:
                # Stop extending forwards if not relevant
                break
        
        # Sort by position (before sentences in reverse order, after sentences in order)
        before_sentences = sorted([s for s in surrounding if s['position'] == 'before'], 
                                 key=lambda x: x['offset'], reverse=True)
        after_sentences = sorted([s for s in surrounding if s['position'] == 'after'], 
                                key=lambda x: x['offset'])
        
        return before_sentences + after_sentences
    
    def _calculate_context_boost(self, surrounding_sentences):
        """
        NEW: Calculate boost based on surrounding sentence confidences.
        Higher boost when multiple surrounding sentences are highly relevant.
        """
        if not surrounding_sentences:
            return 0.0
        
        # Calculate average confidence of surrounding sentences
        avg_surrounding_confidence = np.mean([s['confidence'] for s in surrounding_sentences])
        
        # Count high-confidence surrounding sentences
        high_conf_count = sum(1 for s in surrounding_sentences if s['confidence'] >= 0.5)
        
        # Base boost from average confidence
        base_boost = avg_surrounding_confidence * self.CONTEXT_BONUS
        
        # Additional boost for multiple high-confidence surrounding sentences
        if high_conf_count >= 2:
            extra_boost = self.HIGH_CONTEXT_BONUS * (high_conf_count / len(surrounding_sentences))
            return base_boost + extra_boost
        
        return base_boost
    
    def _calculate_confidence_with_adjustments(self, base_confidence, sentence_text, 
                                               is_exact_match=False, surrounding_sentences=None):
        """
        ENHANCED RULE 2: Apply confidence adjustments based on:
        - Short sentence penalty
        - Exact match bonus
        - Context boost from surrounding sentences
        """
        adjusted_confidence = base_confidence
        
        # Short sentence penalty
        word_count = len(str(sentence_text).split())
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (self.SHORT_SENTENCE_THRESHOLD - word_count) / self.SHORT_SENTENCE_THRESHOLD
            adjusted_confidence -= penalty
        
        # Exact match bonus
        if is_exact_match:
            adjusted_confidence += self.EXACT_MATCH_BONUS
        
        # Context boost from surrounding sentences
        if surrounding_sentences:
            context_boost = self._calculate_context_boost(surrounding_sentences)
            adjusted_confidence += context_boost
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _get_semantic_results(self, query, top_k=9):
        """
        ENHANCED: Semantic search with expanding context and context-aware ranking.
        Tracks used sentence indices to avoid duplication.
        """
        exact_matches = self._has_exact_match(query)
        
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
        
        exact_indices = set(match['index'] for match in exact_matches)
        
        # Track all sentence indices that have been used (main + surrounding)
        used_indices = set()
        
        # First pass: collect all candidates with their surrounding context
        candidates = []
        for idx, sim_score in enumerate(similarities):
            if sim_score >= self.BOOK_RELEVANCE_THRESHOLD and idx not in used_indices:
                row = self.chapters_df.iloc[idx]
                is_exact = idx in exact_indices
                
                # Get expanding surrounding sentences
                surrounding = self._get_surrounding_sentences(
                    row['chapter_number'], 
                    row['sentence_number'], 
                    query,
                    query_embedding[0],
                    max_extend=5
                )
                
                # Mark all surrounding sentence indices as used
                for surr in surrounding:
                    used_indices.add(surr['index'])
                
                # Calculate adjusted confidence with context boost
                adjusted_confidence = self._calculate_confidence_with_adjustments(
                    sim_score, 
                    row['sentence_text'], 
                    is_exact, 
                    surrounding
                )
                
                if adjusted_confidence >= self.BOOK_RELEVANCE_THRESHOLD:
                    candidates.append({
                        'index': idx,
                        'chapter_number': row['chapter_number'],
                        'chapter_title': row['chapter_title'],
                        'sentence_number': row['sentence_number'],
                        'sentence_text': row['sentence_text'],
                        'confidence': adjusted_confidence,
                        'base_confidence': sim_score,
                        'is_exact_match': is_exact,
                        'word_count': row['sentence_word_count'],
                        'surrounding_sentences': surrounding,
                        'context_count': len(surrounding)
                    })
                    
                    # Mark this sentence as used
                    used_indices.add(idx)
        
        # Sort by adjusted confidence (which includes context boost)
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Group by chapters - top 3 chapters, top 3 sentences each
        chapter_groups = {}
        for result in candidates:
            ch_num = result['chapter_number']
            if ch_num not in chapter_groups:
                chapter_groups[ch_num] = []
            if len(chapter_groups[ch_num]) < 3:
                chapter_groups[ch_num].append(result)
        
        # Rank chapters by best sentence confidence (context-aware)
        chapter_rankings = []
        for ch_num, sentences in chapter_groups.items():
            best_confidence = max(s['confidence'] for s in sentences)
            avg_confidence = np.mean([s['confidence'] for s in sentences])
            combined_score = best_confidence * 0.7 + avg_confidence * 0.3
            chapter_rankings.append((ch_num, combined_score, sentences))
        
        chapter_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Flatten results from top 3 chapters
        final_results = []
        for ch_num, _, sentences in chapter_rankings[:3]:
            final_results.extend(sentences)
        
        return final_results[:top_k]
    
    def _get_thematic_classification(self, retrieved_sentences, query):
        """RULE 4: Enhanced thematic classification"""
        if not retrieved_sentences:
            return retrieved_sentences, False, 0.0
        
        thematic_results = []
        
        for sentence_data in retrieved_sentences:
            sentence_text = sentence_data['sentence_text']
            
            sentence_embedding = self.model.encode([sentence_text])
            theme_similarities = cosine_similarity(sentence_embedding, self.theme_embeddings)[0]
            
            matching_themes = []
            for idx, similarity in enumerate(theme_similarities):
                if similarity >= self.THEMATIC_THRESHOLD:
                    theme_row = self.themes_df.iloc[idx]
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
        """Main query processing with enhanced context-aware ranking"""
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
        exact_matches = sum(1 for r in semantic_results if r['is_exact_match'])
        total_context = sum(r['context_count'] for r in thematic_results)
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'results': thematic_results,
            'book_confidence': book_relevance,
            'theme_confidence': avg_theme_conf,
            'exact_matches': exact_matches,
            'total_context_sentences': total_context,
            'total_results': len(thematic_results),
            'avg_confidence': book_relevance
        }
    
    def display_results(self, response, query=""):
        """Enhanced results display with expanding context"""
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
            f"Exact Matches: {response['exact_matches']} | "
            f"Context Sentences: {response.get('total_context_sentences', 0)} | "
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
            
            main_table.add_column("📖 Book", style="cyan", width=15, no_wrap=True)
            main_table.add_column("📚 Ch", style="bright_green", width=6, justify="center")
            main_table.add_column("📝 S#", style="yellow", width=6, justify="center")
            main_table.add_column("🎯 Conf", style="bright_cyan", width=8, justify="center")
            main_table.add_column("✅ Match", style="bright_yellow", width=10, justify="center")
            main_table.add_column("🔄 Context", style="bright_white", width=10, justify="center")
            
            confidence_str = f"{result['confidence']:.1%}"
            match_type = "Exact" if result['is_exact_match'] else "Semantic"
            context_count = result.get('context_count', 0)
            
            main_table.add_row(
                "Noli Me Tangere",
                str(result['chapter_number']),
                str(result['sentence_number']),
                confidence_str,
                match_type,
                f"{context_count} sent."
            )
            
            self.console.print(main_table)
            
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
            
            # Display expanding context sentences
            surrounding = result.get('surrounding_sentences', [])
            if surrounding:
                context_table = Table(
                    title=f"📋 Relevant Surrounding Context ({len(surrounding)} sentences)",
                    show_header=True,
                    header_style="bold yellow",
                    border_style="yellow",
                    box=box.SIMPLE,
                    expand=True
                )
                
                context_table.add_column("Position", style="yellow", width=12)
                context_table.add_column("Sentence #", style="bright_yellow", width=10, justify="center")
                context_table.add_column("Confidence", style="bright_cyan", width=10, justify="center")
                context_table.add_column("Content", style="white", min_width=50)
                
                for surr in surrounding:
                    position = f"⬆️ -{surr['offset']}" if surr['position'] == 'before' else f"⬇️ +{surr['offset']}"
                    context_table.add_row(
                        position,
                        str(surr['sentence_number']),
                        f"{surr['confidence']:.1%}",
                        surr['sentence_text']
                    )
                
                self.console.print(context_table)
            
            # Thematic information
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
        
        # Summary footer
        chapters_found = len(set(r['chapter_number'] for r in results))
        exact_count = sum(1 for r in results if r['is_exact_match'])
        context_total = sum(r.get('context_count', 0) for r in results)
        theme_count = sum(1 for r in results if r.get('has_theme', False))
        
        classification = "🎭 Thematic Analysis" if has_themes else "📚 Semantic Search"
        
        summary_parts = [
            f"{classification}",
            f"{len(results)} sentences from {chapters_found} chapters",
            f"{exact_count} exact matches",
            f"{context_total} surrounding sentences"
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

# Usage
if __name__ == "__main__":
    system = EnhancedNoliSemanticSystem()
    
    welcome_panel = Panel(
        Align.center(Text(
            "📚 Enhanced Noli Me Tangere Semantic System\n"
            "🎯 Context-Aware Ranking with Expanding Context\n"
            "✅ No Duplicate Sentences | 🔍 Smart Context Extension | 📊 Confidence Boosting",
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
                Align.center(Text("Thank you for using the Enhanced Noli System!", style="bold green")),
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