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
try:
    nltk.download('words', quiet=True)
    from nltk.corpus import words
    english_words = set(words.words())
except:
    english_words = set()

warnings.filterwarnings('ignore')

class NoliFineTunedSemanticSystem:
    def __init__(self):
        self.console = Console()
        
        # Load fine-tuned multilingual BERT model (works better for mixed content)
        self.console.print("🔄 Loading fine-tuned BERT model...", style="cyan")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load datasets
        self.console.print("📚 Loading datasets...", style="cyan")
        self.chapters_df = pd.read_csv('chapters.csv')
        self.themes_df = pd.read_csv('themes.csv')
        
        # Clean column names
        self.chapters_df.columns = self.chapters_df.columns.str.strip()
        self.themes_df.columns = self.themes_df.columns.str.strip()
        
        # Build vocabularies for better nonsense detection
        self.tagalog_vocab = self._build_tagalog_vocabulary()
        
        # Precompute embeddings
        self.console.print("🧠 Computing chapter embeddings...", style="cyan")
        self._compute_chapter_embeddings()
        self.console.print("🎯 Computing theme embeddings...", style="cyan")
        self._compute_theme_embeddings()
        
        # Fine-tuned confidence thresholds based on testing
        self.NONSENSE_THRESHOLD = 0.10
        self.BOOK_RELEVANCE_THRESHOLD = 0.30
        self.THEMATIC_THRESHOLD = 0.40
        
        self.console.print("✅ Fine-tuned system ready!", style="bold green")
    
    def _build_tagalog_vocabulary(self):
        """Build comprehensive Tagalog vocabulary from the novel"""
        all_text = ' '.join(self.chapters_df['sentence_text'].astype(str))
        all_text += ' '.join(self.chapters_df['chapter_title'].astype(str))
        all_text += ' '.join(self.themes_df['Tagalog Title'].astype(str))
        all_text += ' '.join(self.themes_df['Meaning'].astype(str))
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{2,}\b', all_text.lower())
        word_freq = Counter(words)
        
        # Keep words that appear at least once and are longer than 2 characters
        return set(word for word, freq in word_freq.items() if len(word) > 2)
    
    def _compute_chapter_embeddings(self):
        """Precompute embeddings for all sentences with better context"""
        self.chapters_df['combined_text'] = (
            self.chapters_df['chapter_title'].astype(str) + " " + 
            self.chapters_df['sentence_text'].astype(str)
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
    
    def _is_nonsense_query(self, query):
        """Enhanced nonsense detection with confidence scoring"""
        query = query.strip().lower()
        confidence_score = 0.0
        
        if len(query) < 2:
            return True, 0.95
        
        # Pattern-based nonsense detection
        nonsense_patterns = [
            (r'^[a-z]*\d+[a-z]*\d*$', 0.9),      # Mixed alphanumeric
            (r'^[^a-zA-ZÀ-ÿ\s]+$', 0.85),        # Only symbols/numbers
            (r'^(.)\1{3,}', 0.8),                 # Repeated characters
            (r'^[bcdfghjklmnpqrstvwxyz]+$', 0.7), # Only consonants
            (r'^\W+$', 0.9)                       # Only non-word chars
        ]
        
        for pattern, conf in nonsense_patterns:
            if re.match(pattern, query):
                return True, conf
        
        # Word validation
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', query)
        if not words:
            return True, 0.9
        
        # Check against vocabularies
        valid_words = 0
        for word in words:
            if (word in self.tagalog_vocab or 
                word in english_words or 
                len(word) <= 3):  # Allow short common words
                valid_words += 1
        
        vocab_ratio = valid_words / len(words)
        
        # If very few valid words, likely nonsense
        if vocab_ratio < 0.5 and len(words) > 1:
            return True, 0.8 - vocab_ratio * 0.3
        
        # Semantic coherence check
        try:
            query_embedding = self.model.encode([query])
            max_similarity = np.max(cosine_similarity(query_embedding, self.chapter_embeddings))
            
            if max_similarity < self.NONSENSE_THRESHOLD:
                return True, 0.9 - max_similarity
        except:
            return True, 0.85
        
        return False, vocab_ratio
    
    def _calculate_book_relevance(self, query):
        """Calculate how relevant query is to the book content"""
        try:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
            
            # Multiple metrics for relevance
            max_sim = np.max(similarities)
            top_10_avg = np.mean(np.sort(similarities)[-10:])
            top_50_avg = np.mean(np.sort(similarities)[-50:])
            
            # Weighted composite score
            relevance_score = (max_sim * 0.5 + top_10_avg * 0.3 + top_50_avg * 0.2)
            
            return relevance_score, max_sim >= self.BOOK_RELEVANCE_THRESHOLD
        
        except:
            return 0.0, False
    
    def _get_semantic_chapters(self, query, top_k=3):
        """Get top semantically relevant chapters with better ranking"""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chapter_embeddings)[0]
        
        # Group by chapter and calculate chapter-level scores
        chapter_scores = {}
        chapter_max_scores = {}
        
        for idx, score in enumerate(similarities):
            chapter_num = self.chapters_df.iloc[idx]['chapter_number']
            if chapter_num not in chapter_scores:
                chapter_scores[chapter_num] = []
                chapter_max_scores[chapter_num] = 0
            
            chapter_scores[chapter_num].append(score)
            chapter_max_scores[chapter_num] = max(chapter_max_scores[chapter_num], score)
        
        # Rank chapters by best sentence + average of top 3
        chapter_rankings = []
        for ch_num in chapter_scores:
            scores = sorted(chapter_scores[ch_num], reverse=True)
            top_3_avg = np.mean(scores[:3])
            combined_score = chapter_max_scores[ch_num] * 0.6 + top_3_avg * 0.4
            chapter_rankings.append((ch_num, combined_score))
        
        # Sort and return top chapters
        top_chapters = sorted(chapter_rankings, key=lambda x: x[1], reverse=True)[:top_k]
        return [ch for ch, score in top_chapters]
    
    def _get_top_sentences_for_chapter(self, query, chapter_num, top_k=3):
        """Get top sentences for a specific chapter with confidence filtering"""
        query_embedding = self.model.encode([query])
        
        # Filter sentences for this chapter
        chapter_mask = self.chapters_df['chapter_number'] == chapter_num
        chapter_indices = self.chapters_df[chapter_mask].index.tolist()
        
        if not chapter_indices:
            return []
        
        # Get similarities for this chapter's sentences
        chapter_embeddings = self.chapter_embeddings[chapter_indices]
        similarities = cosine_similarity(query_embedding, chapter_embeddings)[0]
        
        # Only include sentences above minimum threshold
        valid_results = []
        for idx, sim in enumerate(similarities):
            if sim >= self.BOOK_RELEVANCE_THRESHOLD:
                original_idx = chapter_indices[idx]
                row = self.chapters_df.iloc[original_idx]
                valid_results.append({
                    'chapter_number': row['chapter_number'],
                    'chapter_title': row['chapter_title'],
                    'sentence_number': row['sentence_number'],
                    'sentence_text': row['sentence_text'],
                    'confidence': sim,
                    'index': idx
                })
        
        # Sort by confidence and return top k
        valid_results.sort(key=lambda x: x['confidence'], reverse=True)
        return valid_results[:top_k]
    
    def _get_thematic_relevance(self, query, sentences_data):
        """Enhanced thematic classification with better matching"""
        query_embedding = self.model.encode([query])
        theme_similarities = cosine_similarity(query_embedding, self.theme_embeddings)[0]
        
        # Find best matching theme
        best_theme_idx = np.argmax(theme_similarities)
        best_similarity = theme_similarities[best_theme_idx]
        
        if best_similarity >= self.THEMATIC_THRESHOLD:
            theme_row = self.themes_df.iloc[best_theme_idx]
            theme_info = f"{theme_row['Tagalog Title']} - {theme_row['Meaning']}"
            
            # Add theme information
            for sentence in sentences_data:
                sentence['tagalog_title_meaning'] = theme_info
                sentence['theme_confidence'] = best_similarity
            
            return sentences_data, True, best_similarity
        
        return sentences_data, False, best_similarity
    
    def query(self, user_query):
        """Main query processing with improved logic"""
        # Step 1: Enhanced nonsense detection
        is_nonsense, nonsense_conf = self._is_nonsense_query(user_query)
        
        if is_nonsense:
            return {
                'type': 'nonsense',
                'confidence': nonsense_conf,
                'message': f"Query appears to be nonsense (confidence: {nonsense_conf:.1%})"
            }
        
        # Step 2: Book relevance check
        book_relevance, is_relevant = self._calculate_book_relevance(user_query)
        
        if not is_relevant:
            return {
                'type': 'not_relevant',
                'confidence': book_relevance,
                'message': f"Query not related to Noli Me Tangere (relevance: {book_relevance:.1%})"
            }
        
        # Step 3: Get semantic results
        top_chapters = self._get_semantic_chapters(user_query)
        
        all_results = []
        for chapter_num in top_chapters:
            chapter_results = self._get_top_sentences_for_chapter(user_query, chapter_num)
            all_results.extend(chapter_results)
        
        if not all_results:
            return {
                'type': 'no_matches',
                'confidence': book_relevance,
                'message': f"No high-confidence matches found (book relevance: {book_relevance:.1%})"
            }
        
        # Step 4: Thematic classification
        thematic_results, has_themes, theme_conf = self._get_thematic_relevance(user_query, all_results)
        
        return {
            'type': 'success',
            'subtype': 'thematic' if has_themes else 'semantic',
            'book_confidence': book_relevance,
            'theme_confidence': theme_conf if has_themes else 0.0,
            'results': thematic_results,
            'avg_confidence': np.mean([r['confidence'] for r in thematic_results])
        }
    
    def display_results(self, response, query=""):
        """Enhanced results display with proper error handling"""
        result_type = response['type']
        
        if result_type != 'success':
            # Handle non-success cases
            if result_type == 'nonsense':
                icon = "🚫"
                style = "red"
                title = "Nonsense Query Detected"
            elif result_type == 'not_relevant':
                icon = "⚠️"
                style = "yellow"
                title = "Query Not Book-Related"
            else:  # no_matches
                icon = "📖"
                style = "blue"
                title = "No High-Confidence Matches"
            
            panel = Panel(
                Align.center(Text(f"{icon} {response['message']}", style=f"bold {style}")),
                title=title,
                border_style=style,
                box=box.ROUNDED
            )
            self.console.print(panel)
            return
        
        # Display successful results
        results = response['results']
        subtype = response['subtype']
        
        # Header with confidence metrics
        conf_text = f"📊 Book Relevance: {response['book_confidence']:.1%} | Avg Match: {response['avg_confidence']:.1%}"
        if response['theme_confidence'] > 0:
            conf_text += f" | Theme: {response['theme_confidence']:.1%}"
        
        header_text = Text(f"🎯 Results for: '{query}'\n{conf_text}", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.HEAVY
        )
        self.console.print(header_panel)
        
        # Results table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="bright_blue",
            box=box.ROUNDED,
            show_lines=True,
            expand=True
        )
        
        has_themes = subtype == 'thematic'
        
        table.add_column("📖 Ch", style="cyan", width=8, justify="center")
        table.add_column("📚 Title", style="bright_green", min_width=20, max_width=30, overflow="fold")
        table.add_column("📝 Sent", style="yellow", width=8, justify="center")
        table.add_column("💬 Content", style="white", min_width=40, max_width=60, overflow="fold")
        table.add_column("🎯 Conf", style="bright_cyan", width=8, justify="center")
        
        if has_themes:
            table.add_column("🎭 Theme", style="bright_magenta", min_width=30, max_width=50, overflow="fold")
        
        for result in results:
            confidence_str = f"{result['confidence']:.1%}"
            
            if has_themes:
                table.add_row(
                    str(result['chapter_number']),
                    result['chapter_title'],
                    str(result['sentence_number']),
                    result['sentence_text'],
                    confidence_str,
                    result['tagalog_title_meaning']
                )
            else:
                table.add_row(
                    str(result['chapter_number']),
                    result['chapter_title'],
                    str(result['sentence_number']),
                    result['sentence_text'],
                    confidence_str
                )
        
        self.console.print(table)
        
        # Summary footer
        chapters_found = len(set(r['chapter_number'] for r in results))
        classification = "🎭 Thematic" if has_themes else "📚 Semantic"
        
        summary = f"{classification} | {len(results)} sentences from {chapters_found} chapters"
        footer_panel = Panel(
            Align.center(Text(summary, style="dim white")),
            style="dim blue",
            box=box.SIMPLE
        )
        self.console.print(footer_panel)

# Initialize the fine-tuned system
system = NoliFineTunedSemanticSystem()

# Enhanced interface
welcome_panel = Panel(
    Align.center(Text("📚 Fine-Tuned Noli Me Tangere Semantic System\n🎯 Enhanced BERT with Confidence Scoring", style="bold white")),
    style="bright_green",
    box=box.HEAVY
)
system.console.print(welcome_panel)

instructions = """
🔍 **Query Classification System:**
• **Nonsense Detection**: Filters invalid queries with confidence scoring
• **Relevance Check**: Ensures queries relate to Noli Me Tangere content
• **Semantic Retrieval**: High-confidence sentence matching
• **Thematic Classification**: Identifies thematic connections when applicable

📝 **Test Examples:**
• **Education**: 'edukasyon', 'paaralan', 'karunungan', 'kaalaman'
• **Characters**: 'Maria Clara', 'Ibarra', 'Elias', 'Padre Damaso'
• **Themes**: 'pag-ibig', 'kalayaan', 'katarungan', 'kapangyarihan'
• **Events**: 'kamatayan', 'sementeryo', 'pista', 'prusisyon'
• **Nonsense**: 'xyz123', 'asdfgh', '!@#$'
• **Unrelated**: 'basketball', 'computer', 'pizza'
"""

instructions_panel = Panel(
    instructions,
    title="🎯 System Features",
    style="cyan",
    box=box.ROUNDED
)
system.console.print(instructions_panel)

while True:
    system.console.print("\n" + "─" * 80, style="dim")
    user_input = system.console.input("[bold cyan]Enter query: [/bold cyan]").strip()
    
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