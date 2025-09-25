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

# Try to use textblob for language detection, fallback to basic detection
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except:
    TEXTBLOB_AVAILABLE = False

warnings.filterwarnings('ignore')

class EnhancedNoliSemanticSystem:
    def __init__(self):
        self.console = Console()
        
        # Initialize vocabularies (no external spell checkers needed)
        self.console.print("🔄 Building vocabulary from datasets...", style="cyan")
        
        # Load fine-tuned multilingual model (XLM-RoBERTa or multilingual BERT)
        self.console.print("🔄 Loading XLM-RoBERTa multilingual model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        
        # Load datasets
        self.console.print("📚 Loading datasets...", style="cyan")
        self.chapters_df = pd.read_csv('chapters.csv')
        self.themes_df = pd.read_csv('themes.csv')
        
        # Clean column names
        self.chapters_df.columns = self.chapters_df.columns.str.strip()
        self.themes_df.columns = self.themes_df.columns.str.strip()
        
        # Build vocabularies from dataset
        self.tagalog_vocab = self._build_tagalog_vocabulary()
        
        # Precompute embeddings
        self.console.print("🧠 Computing chapter embeddings...", style="cyan")
        self._compute_chapter_embeddings()
        self.console.print("🎯 Computing theme embeddings...", style="cyan")
        self._compute_theme_embeddings()
        
        # Enhanced thresholds based on testing
        self.NONSENSE_THRESHOLD = 0.15
        self.BOOK_RELEVANCE_THRESHOLD = 0.25
        self.THEMATIC_THRESHOLD = 0.35
        self.SHORT_SENTENCE_THRESHOLD = 5  # words
        self.SHORT_SENTENCE_PENALTY = 0.2  # confidence reduction
        self.EXACT_MATCH_BONUS = 0.15  # confidence boost for exact matches
        
        self.console.print("✅ Enhanced system ready!", style="bold green")
    
    def _build_tagalog_vocabulary(self):
        """Build comprehensive Tagalog vocabulary from the novel"""
        all_text = ' '.join(self.chapters_df['sentence_text'].astype(str))
        all_text += ' '.join(self.chapters_df['chapter_title'].astype(str))
        all_text += ' '.join(self.themes_df['Tagalog Title'].astype(str))
        all_text += ' '.join(self.themes_df['Meaning'].astype(str))
        
        # Extract meaningful words (including accented characters)
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]{2,}\b', all_text.lower())
        word_freq = Counter(words)
        
        # Keep words that appear at least twice (higher threshold for quality)
        vocab_set = set(word for word, freq in word_freq.items() if freq >= 2 and len(word) > 2)
        
        self.console.print(f"📖 Built vocabulary: {len(vocab_set)} Tagalog words from dataset", style="dim cyan")
        return vocab_set
    
    def _compute_chapter_embeddings(self):
        """Precompute embeddings for all sentences with enhanced context"""
        self.chapters_df['combined_text'] = (
            self.chapters_df['chapter_title'].astype(str) + " " + 
            self.chapters_df['sentence_text'].astype(str)
        )
        
        # Add sentence length for penalty calculation
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
        
        # Check English dictionary from nltk
        if english_words and word_lower in english_words:
            return True
        
        # Check Tagalog vocabulary from dataset
        if word_lower in self.tagalog_vocab:
            return True
        
        # Allow proper nouns (capitalized words) if reasonable length
        if word[0].isupper() and 3 <= len(word) <= 20:
            return True
        
        # Allow common short words and punctuation-adjacent words
        if len(word) <= 3:
            return True
            
        return False
    
    def _detect_language(self, text):
        """Simple language detection using available tools"""
        if not text or len(text.strip()) < 2:
            return 'unknown'
            
        # Use TextBlob if available
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                detected = blob.detect_language()
                return detected if detected else 'unknown'
            except:
                pass
        
        # Simple heuristic-based detection
        # Count English vs potential Tagalog characteristics
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
        
        # Pattern-based nonsense detection (improved patterns)
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
        
        # Extract words for validation
        words = re.findall(r'\b[a-zA-ZÀ-ÿñÑ]+\b', query)
        
        if not words:
            return True, 0.9, "No valid words found"
        
        # Real word validation using enhanced method
        valid_words = 0
        for word in words:
            if self._is_real_word(word):
                valid_words += 1
        
        valid_ratio = valid_words / len(words)
        
        # Stricter threshold for real words
        if valid_ratio < 0.6:  # At least 60% must be real words
            return True, 0.9 - valid_ratio * 0.3, f"Only {valid_ratio:.1%} valid words"
        
        # Language detection check
        detected_lang = self._detect_language(query)
        if detected_lang not in ['en', 'tl', 'unknown']:  # English, Tagalog, or uncertain
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
    
    def _calculate_confidence_with_adjustments(self, base_confidence, sentence_text, is_exact_match=False):
        """RULE 2: Apply confidence adjustments for short sentences and exact matches"""
        adjusted_confidence = base_confidence
        
        # Short sentence penalty
        word_count = len(str(sentence_text).split())
        if word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (self.SHORT_SENTENCE_THRESHOLD - word_count) / self.SHORT_SENTENCE_THRESHOLD
            adjusted_confidence -= penalty
        
        # Exact match bonus
        if is_exact_match:
            adjusted_confidence += self.EXACT_MATCH_BONUS
        
        # Ensure confidence stays within bounds
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _get_semantic_results(self, query, top_k=9):
        """Enhanced semantic search with exact match integration"""
        # Check for exact matches first
        exact_matches = self._has_exact_match(query)
        
        # Get semantic similarities
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
        
        # Combine exact matches with semantic results
        results = []
        exact_indices = set(match['index'] for match in exact_matches)
        
        # Process all sentences
        for idx, sim_score in enumerate(similarities):
            if sim_score >= self.BOOK_RELEVANCE_THRESHOLD:
                row = self.chapters_df.iloc[idx]
                is_exact = idx in exact_indices
                
                adjusted_confidence = self._calculate_confidence_with_adjustments(
                    sim_score, row['sentence_text'], is_exact
                )
                
                if adjusted_confidence >= self.BOOK_RELEVANCE_THRESHOLD:
                    results.append({
                        'index': idx,
                        'chapter_number': row['chapter_number'],
                        'chapter_title': row['chapter_title'],
                        'sentence_number': row['sentence_number'],
                        'sentence_text': row['sentence_text'],
                        'confidence': adjusted_confidence,
                        'is_exact_match': is_exact,
                        'word_count': row['sentence_word_count']
                    })
        
        # Sort by confidence and return top results
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Group by chapters and select top 3 chapters, top 3 sentences each
        chapter_groups = {}
        for result in results:
            ch_num = result['chapter_number']
            if ch_num not in chapter_groups:
                chapter_groups[ch_num] = []
            if len(chapter_groups[ch_num]) < 3:  # Max 3 sentences per chapter
                chapter_groups[ch_num].append(result)
        
        # Get top 3 chapters by best sentence in each
        chapter_rankings = []
        for ch_num, sentences in chapter_groups.items():
            best_confidence = max(s['confidence'] for s in sentences)
            chapter_rankings.append((ch_num, best_confidence, sentences))
        
        chapter_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Flatten results from top 3 chapters
        final_results = []
        for ch_num, _, sentences in chapter_rankings[:3]:
            final_results.extend(sentences)
        
        return final_results[:top_k]
    
    def _get_thematic_classification(self, retrieved_sentences):
        """RULE 4: Enhanced thematic classification based on retrieved sentences"""
        if not retrieved_sentences:
            return retrieved_sentences, False, 0.0
        
        thematic_results = []
        
        for sentence_data in retrieved_sentences:
            sentence_text = sentence_data['sentence_text']
            
            # Encode the retrieved sentence
            sentence_embedding = self.model.encode([sentence_text])
            
            # Compare against all themes
            theme_similarities = cosine_similarity(sentence_embedding, self.theme_embeddings)[0]
            
            # Find themes above threshold
            matching_themes = []
            for idx, similarity in enumerate(theme_similarities):
                if similarity >= self.THEMATIC_THRESHOLD:
                    theme_row = self.themes_df.iloc[idx]
                    matching_themes.append({
                        'tagalog_title': theme_row['Tagalog Title'],
                        'meaning': theme_row['Meaning'],
                        'theme_confidence': similarity
                    })
            
            # Sort themes by confidence
            matching_themes.sort(key=lambda x: x['theme_confidence'], reverse=True)
            
            # Add thematic information to sentence data
            enhanced_sentence = sentence_data.copy()
            
            if matching_themes:
                # Take top theme or multiple themes
                enhanced_sentence['themes'] = matching_themes[:2]  # Top 2 themes max
                enhanced_sentence['primary_theme'] = matching_themes[0]
                enhanced_sentence['has_theme'] = True
            else:
                enhanced_sentence['themes'] = []
                enhanced_sentence['primary_theme'] = None
                enhanced_sentence['has_theme'] = False
            
            thematic_results.append(enhanced_sentence)
        
        # Calculate overall thematic relevance
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
        """Main query processing with enhanced rule implementation"""
        # RULE 1: Enhanced nonsense detection
        is_nonsense, confidence, reason = self._is_nonsense_query(user_query)
        
        if is_nonsense:
            return {
                'type': 'nonsense',
                'confidence': confidence,
                'reason': reason,
                'message': f"Query classified as nonsense: {reason} (confidence: {confidence:.1%})"
            }
        
        # RULE 2: Semantic search with confidence adjustments
        semantic_results = self._get_semantic_results(user_query)
        
        if not semantic_results:
            return {
                'type': 'no_matches',
                'message': "No high-confidence matches found in Noli Me Tangere"
            }
        
        # RULE 4: Thematic classification based on retrieved sentences
        thematic_results, has_themes, avg_theme_conf = self._get_thematic_classification(semantic_results)
        
        # Calculate overall metrics
        book_relevance = np.mean([r['confidence'] for r in semantic_results])
        exact_matches = sum(1 for r in semantic_results if r['is_exact_match'])
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'results': thematic_results,
            'book_confidence': book_relevance,
            'theme_confidence': avg_theme_conf,
            'exact_matches': exact_matches,
            'total_results': len(thematic_results),
            'avg_confidence': book_relevance
        }
    
    def display_results(self, response, query=""):
        """Enhanced results display with detailed information"""
        result_type = response['type']
        
        if result_type != 'success':
            # Handle non-success cases with single-cell bordered table
            if result_type == 'nonsense':
                # Single-cell table for nonsense as per Rule 1
                table = Table(show_header=False, box=box.HEAVY, border_style="red")
                table.add_column("Result", style="bold red", justify="center")
                table.add_row("none")
                self.console.print(table)
                
                # Additional info panel
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
        
        # Display successful results
        results = response['results']
        subtype = response['subtype']
        has_themes = subtype == 'thematic'
        
        # Header with comprehensive metrics
        metrics_text = (
            f"📊 Book Relevance: {response['book_confidence']:.1%} | "
            f"Exact Matches: {response['exact_matches']} | "
            f"Total Results: {response['total_results']}"
        )
        if response['theme_confidence'] > 0:
            metrics_text += f" | Thematic: {response['theme_confidence']:.1%}"
        
        header_text = Text(f"🎯 Results for: '{query}'\n{metrics_text}", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.HEAVY
        )
        self.console.print(header_panel)
        
        # Main results table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            box=box.ROUNDED,
            show_lines=True,
            expand=True
        )
        
        # Standard columns as per Rule 2 format
        table.add_column("📖 Book", style="cyan", width=15, justify="center")
        table.add_column("📚 Ch", style="bright_green", width=8, justify="center")
        table.add_column("📑 Title", style="bright_green", min_width=20, max_width=25, overflow="fold")
        table.add_column("📝 Sent", style="yellow", width=8, justify="center")
        table.add_column("💬 Content", style="white", min_width=35, max_width=50, overflow="fold")
        table.add_column("🎯 Conf", style="bright_cyan", width=10, justify="center")
        
        # Additional columns for enhanced info
        table.add_column("✅ Type", style="bright_yellow", width=10, justify="center")
        
        if has_themes:
            table.add_column("🎭 Theme", style="bright_magenta", min_width=25, max_width=40, overflow="fold")
            table.add_column("📖 Meaning", style="magenta", min_width=30, max_width=45, overflow="fold")
        
        # Populate table rows
        for result in results:
            confidence_str = f"{result['confidence']:.1%}"
            match_type = "Exact" if result['is_exact_match'] else "Semantic"
            
            row_data = [
                "Noli Me Tangere",
                str(result['chapter_number']),
                result['chapter_title'],
                str(result['sentence_number']),
                result['sentence_text'],
                confidence_str,
                match_type
            ]
            
            if has_themes and result['has_theme']:
                primary_theme = result['primary_theme']
                row_data.extend([
                    primary_theme['tagalog_title'],
                    primary_theme['meaning']
                ])
            elif has_themes:
                row_data.extend(["No theme", "No thematic connection"])
            
            table.add_row(*row_data)
        
        self.console.print(table)
        
        # Summary footer
        chapters_found = len(set(r['chapter_number'] for r in results))
        classification = "🎭 Thematic Analysis" if has_themes else "📚 Semantic Search"
        exact_count = sum(1 for r in results if r['is_exact_match'])
        
        summary_parts = [
            f"{classification}",
            f"{len(results)} sentences from {chapters_found} chapters",
            f"{exact_count} exact matches" if exact_count > 0 else "semantic matches only"
        ]
        
        summary = " | ".join(summary_parts)
        footer_panel = Panel(
            Align.center(Text(summary, style="dim white")),
            style="dim blue",
            box=box.SIMPLE
        )
        self.console.print(footer_panel)

# Usage example
if __name__ == "__main__":
    # Initialize the enhanced system
    system = EnhancedNoliSemanticSystem()
    
    # Welcome interface
    welcome_panel = Panel(
        Align.center(Text(
            "📚 Enhanced Noli Me Tangere Semantic System\n"
            "🎯 XLM-RoBERTa with Advanced Rule Implementation\n"
            "✅ Real Word Detection | 🔍 Exact Match Priority | 🎭 Thematic Analysis",
            style="bold white"
        )),
        style="bright_green",
        box=box.HEAVY
    )
    system.console.print(welcome_panel)
    
    # Interactive loop
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