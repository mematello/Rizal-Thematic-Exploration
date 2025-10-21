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
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
import os
import torch, gc


warnings.filterwarnings('ignore')

WORD_PATTERN = re.compile(r'[0-9a-zA-ZÀ-ÿñÑ]+(?:-[0-9a-zA-ZÀ-ÿñÑ]+)*')

def extract_words(text):
    """Extract words, preserving hyphenated tokens as single units."""
    return WORD_PATTERN.findall(text)

class DAPTManager:
    """Handles Domain-Adaptive Pretraining for XLM-RoBERTa"""
    
    def __init__(self, base_model_name='sentence-transformers/paraphrase-xlm-r-multilingual-v1', 
                 output_dir='./dapt_model'):
        self.console = Console()
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.adapted_model_path = os.path.join(output_dir, 'final_model')
        
    def prepare_corpus_for_mlm(self, books_data):
        """Extract all text from corpus for MLM training"""
        self.console.print("Preparing corpus for DAPT...", style="cyan")
        
        all_texts = []
        
        for book_key, book_data in books_data.items():
            chapters_df = book_data['chapters']
            themes_df = book_data['themes']
            
            # Extract chapter sentences
            all_texts.extend(chapters_df['sentence_text'].astype(str).tolist())
            
            # Extract theme meanings
            all_texts.extend(themes_df['Meaning'].astype(str).tolist())
            
            # Extract chapter titles
            all_texts.extend(chapters_df['chapter_title'].astype(str).unique().tolist())
            
        # Remove duplicates and empty strings
        all_texts = list(set([t.strip() for t in all_texts if t.strip()]))
        
        self.console.print(f"  Prepared {len(all_texts)} unique texts for DAPT", style="green")
        return all_texts
    
    def train_dapt(self, corpus_texts, epochs=3, batch_size=8, learning_rate=5e-5, save_steps=500):
        """Perform Domain-Adaptive Pretraining with MLM objective"""
        self.console.print("Starting DAPT training...", style="bold cyan")
        
        # Load tokenizer and model
        self.console.print("  Loading base model and tokenizer...", style="cyan")
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.base_model_name)
        
        # Prepare dataset
        self.console.print("  Tokenizing corpus...", style="cyan")
        dataset = Dataset.from_dict({"text": corpus_texts})
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512, 
                           padding='max_length', return_special_tokens_mask=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, 
                                       remove_columns=dataset.column_names)
        
        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=save_steps,
            save_total_limit=2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=100,
            logging_dir=f'{self.output_dir}/logs',
            report_to='none',
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )
        
        # Train
        self.console.print("  Training in progress...", style="bold yellow")
        trainer.train()
        
        # Save adapted model
        self.console.print("  Saving adapted model...", style="cyan")
        trainer.save_model(self.adapted_model_path)
        tokenizer.save_pretrained(self.adapted_model_path)
        
        self.console.print("✓ DAPT training completed!", style="bold green")
        return self.adapted_model_path
    
    def load_adapted_model(self):
        """Load the domain-adapted model for embedding generation"""
        if not os.path.exists(self.adapted_model_path):
            raise FileNotFoundError(f"Adapted model not found at {self.adapted_model_path}")
        
        self.console.print(f"Loading DAPT model from {self.adapted_model_path}...", style="cyan")
        model = SentenceTransformer(self.adapted_model_path)
        self.console.print("✓ DAPT model loaded successfully!", style="green")
        return model


class WordRelationshipAnalyzer:
    """Analyzes semantic relationships between query words using corpus-adapted embeddings"""
    
    def __init__(self, model, stopwords, relationship_threshold=0.45):
        self.model = model
        self.stopwords = stopwords
        self.relationship_threshold = relationship_threshold
        self.console = Console()
        
    def get_word_embeddings(self, words):
        """Generate embeddings for individual words"""
        # Filter out stopwords
        content_words = [w for w in words if w.lower() not in self.stopwords]
        
        if not content_words:
            return {}, []
        
        # Generate embeddings
        embeddings = self.model.encode(content_words)
        word_embeddings = {word: emb for word, emb in zip(content_words, embeddings)}
        
        return word_embeddings, content_words
    
    def compute_pairwise_relationships(self, word_embeddings):
        """Compute cosine similarity between all word pairs"""
        words = list(word_embeddings.keys())
        relationships = []
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word1, word2 = words[i], words[j]
                emb1 = word_embeddings[word1]
                emb2 = word_embeddings[word2]
                
                # Compute cosine similarity
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                
                # Classify relationship strength
                if similarity >= 0.70:
                    strength = "Strong"
                    strength_style = "bright_green"
                elif similarity >= 0.50:
                    strength = "Moderate"
                    strength_style = "yellow"
                elif similarity >= self.relationship_threshold:
                    strength = "Weak"
                    strength_style = "orange1"
                else:
                    strength = "Very Weak"
                    strength_style = "red"
                
                relationships.append({
                    'word1': word1,
                    'word2': word2,
                    'similarity': similarity,
                    'strength': strength,
                    'strength_style': strength_style,
                    'percentage': similarity * 100
                })
        
        return relationships
    
    def analyze_query_relationships(self, query):
        """Analyze relationships between words in the query"""
        words = extract_words(query)
        
        # Get embeddings for content words
        word_embeddings, content_words = self.get_word_embeddings(words)
        
        if len(content_words) < 2:
            return {
                'type': 'insufficient_words',
                'message': 'Query must contain at least 2 content words (non-stopwords)',
                'content_words': content_words
            }
        
        # Compute pairwise relationships
        relationships = self.compute_pairwise_relationships(word_embeddings)
        
        # Check if any relationship is below threshold
        min_similarity = min([r['similarity'] for r in relationships])
        avg_similarity = np.mean([r['similarity'] for r in relationships])
        
        weak_relationships = [r for r in relationships if r['similarity'] < self.relationship_threshold]
        
        is_valid = len(weak_relationships) == 0
        
        return {
            'type': 'valid' if is_valid else 'weak_relationship',
            'is_valid': is_valid,
            'content_words': content_words,
            'relationships': relationships,
            'min_similarity': min_similarity,
            'avg_similarity': avg_similarity,
            'weak_relationships': weak_relationships,
            'threshold': self.relationship_threshold
        }
    
    def display_relationship_analysis(self, analysis):
        """Display word relationship analysis in a rich table"""
        if analysis['type'] == 'insufficient_words':
            warning_panel = Panel(
                f"{analysis['message']}\nContent words found: {', '.join(analysis['content_words']) if analysis['content_words'] else 'None'}",
                title="⚠ Insufficient Content Words",
                style="yellow",
                box=box.ROUNDED
            )
            self.console.print(warning_panel)
            return
        
        # Display relationship table
        rel_table = Table(
            title="📊 Word Relationship Analysis (Corpus-Adapted Embeddings)",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=True
        )
        
        rel_table.add_column("Word 1", style="bright_white", width=20)
        rel_table.add_column("Word 2", style="bright_white", width=20)
        rel_table.add_column("Similarity", style="bright_cyan", width=12, justify="center")
        rel_table.add_column("Percentage", style="bright_yellow", width=12, justify="center")
        rel_table.add_column("Strength", style="bright_green", width=15, justify="center")
        
        for rel in analysis['relationships']:
            rel_table.add_row(
                rel['word1'],
                rel['word2'],
                f"{rel['similarity']:.4f}",
                f"{rel['percentage']:.2f}%",
                f"[{rel['strength_style']}]{rel['strength']}[/{rel['strength_style']}]"
            )
        
        self.console.print("\n")
        self.console.print(rel_table)
        
        # Display summary
        summary_text = (
            f"Content Words: {', '.join(analysis['content_words'])}\n"
            f"Minimum Similarity: {analysis['min_similarity']:.4f} ({analysis['min_similarity']*100:.2f}%)\n"
            f"Average Similarity: {analysis['avg_similarity']:.4f} ({analysis['avg_similarity']*100:.2f}%)\n"
            f"Relationship Threshold: {analysis['threshold']:.4f} ({analysis['threshold']*100:.2f}%)"
        )
        
        if analysis['is_valid']:
            summary_panel = Panel(
                f"✓ All word pairs are sufficiently related\n\n{summary_text}",
                title="Query Relationship Status",
                style="green",
                box=box.ROUNDED
            )
        else:
            weak_pairs = [f"'{r['word1']}' ↔ '{r['word2']}' ({r['similarity']:.4f})" 
                         for r in analysis['weak_relationships']]
            summary_panel = Panel(
                f"✗ Query contains weakly related words\n\n{summary_text}\n\n"
                f"Weak pairs:\n" + "\n".join([f"  • {p}" for p in weak_pairs]),
                title="Query Relationship Status",
                style="red",
                box=box.ROUNDED
            )
        
        self.console.print(summary_panel)
        self.console.print("\n")


class QueryAnalyzer:
    """Handles query validation and linguistic analysis with official Tagalog stopwords"""

    def __init__(self):
        self.MIN_FILIPINO_FREQUENCY = 1e-8
        self.MIN_VALID_WORD_RATIO = 0.5
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
            return {'ng', 'sa', 'ang', 'na', 'ay', 'at', 'mga'}

    def get_word_frequency(self, word, lang='tl'):
        """Get word frequency using wordfreq library"""
        try:
            from wordfreq import word_frequency
            return word_frequency(word.lower(), lang)
        except Exception:
            return 0.0

    def is_valid_filipino_word(self, word):
        """Check if a word is valid Filipino with positive frequency"""
        if len(word) < 2:
            return False
        freq = self.get_word_frequency(word, 'tl')
        return freq > 0

    def validate_filipino_query(self, query):
        """Validate if query contains valid Filipino words"""
        words = extract_words(query)

        if not words:
            return False, {
                'reason': 'No valid words found in query',
                'total_words': 0,
                'valid_words': 0,
                'invalid_words': [],
                'valid_ratio': 0.0
            }

        valid_words = [word for word in words if self.is_valid_filipino_word(word)]
        invalid_words = [word for word in words if not self.is_valid_filipino_word(word)]

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
        words = extract_words(query)
        analysis = []

        for word in words:
            word_lower = word.lower()
            freq = self.get_word_frequency(word_lower, 'tl')
            is_stopword = word_lower in self.STOPWORDS

            if is_stopword:
                semantic_weight = 0.0
            elif freq > 0.001:
                semantic_weight = 0.3
            elif freq > 0.0001:
                semantic_weight = 0.7
            else:
                semantic_weight = 1.0

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
        words = extract_words(query)
        if not words:
            return 0.0
        stopword_count = sum(1 for w in words if w.lower() in self.STOPWORDS)
        return stopword_count / len(words)

    def get_content_words(self, query):
        """Extract non-stopword content words from query"""
        words = extract_words(query)
        return [w.lower() for w in words if w.lower() not in self.STOPWORDS]


class DAPTEnhancedCLEARSystem:
    """
    DAPT-Enhanced Dynamic Dual-Formula CLEAR System
    - Domain-Adaptive Pretraining for corpus-specific embeddings
    - Word relationship filtering based on embedding similarity
    - Dynamic weights based on sentence length
    """

    def __init__(self, use_dapt=True, dapt_epochs=3, relationship_threshold=0.45, 
                 force_retrain=False):
        self.console = Console()
        self.use_dapt = use_dapt
        self.relationship_threshold = relationship_threshold
        
        # Initialize DAPT Manager
        if self.use_dapt:
            self.console.print("Initializing DAPT Manager...", style="cyan")
            self.dapt_manager = DAPTManager()
        
        # Load datasets first
        self.console.print("Loading datasets...", style="cyan")
        self.books_data = {}
        self.used_passages = {}
        self.corpus_vocabulary = {}
        self.global_vocabulary = set()
        self._load_books()
        
        # Perform DAPT if requested
        if self.use_dapt:
            adapted_model_exists = os.path.exists(self.dapt_manager.adapted_model_path)
            
            if force_retrain or not adapted_model_exists:
                corpus_texts = self.dapt_manager.prepare_corpus_for_mlm(self.books_data)
                self.dapt_manager.train_dapt(corpus_texts, epochs=dapt_epochs)
                self.model = self.dapt_manager.load_adapted_model()
            else:
                self.console.print("Using existing DAPT model...", style="yellow")
                self.model = self.dapt_manager.load_adapted_model()
        else:
            self.console.print("Loading base XLM-RoBERTa model (no DAPT)...", style="cyan")
            self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

        self.console.print("Initializing query analyzer with official stopwords...", style="cyan")
        self.query_analyzer = QueryAnalyzer()
        
        # Initialize word relationship analyzer
        self.console.print("Initializing word relationship analyzer...", style="cyan")
        self.relationship_analyzer = WordRelationshipAnalyzer(
            self.model, 
            self.query_analyzer.STOPWORDS,
            self.relationship_threshold
        )

        self.console.print("Computing embeddings for all books...", style="cyan")
        self._compute_all_embeddings()

        self.console.print("Building corpus vocabulary...", style="cyan")
        self._build_corpus_vocabulary()

        # System parameters
        self.MIN_SEMANTIC_THRESHOLD = 0.20
        self.THEMATIC_THRESHOLD = 0.45
        self.NEIGHBOR_RELEVANCE_THRESHOLD = 0.40
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.08
        self.MAX_CONTEXT_EXPANSION = 5
        self.HIGH_STOPWORD_RATIO = 0.6
        self.STOPWORD_PENALTY_FACTOR = 0.5

        model_type = "DAPT-Enhanced" if self.use_dapt else "Base"
        self.console.print(f"{model_type} Dynamic Dual-Formula CLEAR system ready!", style="bold green")

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
        """Compute embeddings for all books using adapted model"""
        for book_data in self.books_data.values():
            chapters_df = book_data['chapters']
            themes_df = book_data['themes']

            chapters_df['combined_text'] = (
                chapters_df['chapter_title'].astype(str) + " " +
                chapters_df['sentence_text'].astype(str)
            )

            chapters_df['sentence_word_count'] = (
                chapters_df['sentence_text'].astype(str).apply(lambda x: len(x.split()))
            )

            texts = chapters_df['combined_text'].tolist()
            book_data['embeddings'] = self.model.encode(texts, show_progress_bar=False)

            themes_df['theme_text'] = (
                themes_df['Tagalog Title'].astype(str) + " " +
                themes_df['Meaning'].astype(str)
            )

            theme_texts = themes_df['theme_text'].tolist()
            book_data['theme_embeddings'] = self.model.encode(theme_texts, show_progress_bar=False)

    def _build_corpus_vocabulary(self):
        """Build vocabulary from corpus and themes for lexical presence check"""
        for book_key, book_data in self.books_data.items():
            vocabulary = set()

            for text in book_data['chapters']['sentence_text'].astype(str):
                vocabulary.update(extract_words(text.lower()))

            for text in book_data['themes']['Meaning'].astype(str):
                vocabulary.update(extract_words(text.lower()))

            vocabulary -= self.query_analyzer.STOPWORDS

            self.corpus_vocabulary[book_key] = vocabulary
            self.global_vocabulary.update(vocabulary)
            self.console.print(f"  {book_key} corpus vocabulary: {len(vocabulary)} content words", style="green")

    def _get_passage_id(self, chapter_num, sentence_num):
        """Create unique identifier for passages"""
        return (int(chapter_num), int(sentence_num))

    def _compute_dynamic_weights_by_length(self, text_length, reference_length=None):
        """
        Compute dynamic weights based on text length
        Longer text → higher Semantic weight, lower Lexical weight
        Shorter text → higher Lexical weight, lower Semantic weight
        """
        if reference_length is None:
            # Main retrieval: based on absolute sentence length
            if text_length <= 5:
                lambda_lex, lambda_sem = 0.75, 0.25
            elif text_length <= 10:
                lambda_lex, lambda_sem = 0.65, 0.35
            elif text_length <= 15:
                lambda_lex, lambda_sem = 0.55, 0.45
            elif text_length <= 20:
                lambda_lex, lambda_sem = 0.45, 0.55
            else:
                lambda_lex, lambda_sem = 0.35, 0.65
        else:
            # Neighbor/thematic retrieval: based on relative length
            length_ratio = text_length / max(reference_length, 1)
            
            if length_ratio >= 1.5:  # Much longer
                lambda_lex, lambda_sem = 0.30, 0.70
            elif length_ratio >= 1.2:  # Longer
                lambda_lex, lambda_sem = 0.40, 0.60
            elif length_ratio >= 0.8:  # Similar length
                lambda_lex, lambda_sem = 0.50, 0.50
            elif length_ratio >= 0.5:  # Shorter
                lambda_lex, lambda_sem = 0.60, 0.40
            else:  # Much shorter
                lambda_lex, lambda_sem = 0.70, 0.30

        return lambda_lex, lambda_sem

    def _compute_lexical_score_weighted(self, query, sentence_text, query_analysis):
        """Compute weighted lexical overlap score with exact word boundary matching"""
        query_lower = query.lower().strip()
        sentence_lower = sentence_text.lower().strip()

        if query_lower == sentence_lower:
            return 1.0

        query_pattern = r'\b' + re.escape(query_lower) + r'\b'
        if re.search(query_pattern, sentence_lower):
            return min(1.0, len(query_lower) / len(sentence_lower) * 2)

        query_words_data = {item['word'].lower(): item['semantic_weight'] for item in query_analysis}
        sentence_words = set(extract_words(sentence_lower))

        if not query_words_data:
            return 0.0

        total_weight = sum(query_words_data.values())
        matched_weight = sum(weight for word, weight in query_words_data.items() if word in sentence_words)

        if total_weight == 0:
            return 0.0

        weighted_score = matched_weight / total_weight

        stopword_ratio = self.query_analyzer.get_stopword_ratio(query)
        if stopword_ratio > self.HIGH_STOPWORD_RATIO:
            penalty = (stopword_ratio - self.HIGH_STOPWORD_RATIO) * self.STOPWORD_PENALTY_FACTOR
            weighted_score *= (1.0 - penalty)

        return weighted_score

    def _compute_lexical_score_simple(self, text1, text2):
        """Compute simple lexical overlap (Jaccard) for neighbor/thematic comparison"""
        words1 = set(extract_words(text1.lower()))
        words2 = set(extract_words(text2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _calculate_clear_score(self, semantic_sim, lexical_score, lambda_lex, lambda_sem, word_count=None):
        """CLEAR hybrid scoring with dynamic weights"""
        final_score = (lambda_sem * semantic_sim) + (lambda_lex * lexical_score)

        if word_count and word_count < self.SHORT_SENTENCE_THRESHOLD:
            penalty = self.SHORT_SENTENCE_PENALTY * (
                self.SHORT_SENTENCE_THRESHOLD - word_count
            ) / self.SHORT_SENTENCE_THRESHOLD
            final_score -= penalty

        return max(0.0, min(1.0, final_score))

    def _compute_neighbor_similarity(self, main_text, main_length, neighbor_text, neighbor_length):
        """Compute similarity between main sentence and neighbor with dynamic weights"""
        # Dynamic weights based on relative lengths
        lambda_lex, lambda_sem = self._compute_dynamic_weights_by_length(neighbor_length, main_length)

        # Semantic similarity
        main_embedding = self.model.encode([main_text])
        neighbor_embedding = self.model.encode([neighbor_text])
        semantic_sim = cosine_similarity(main_embedding, neighbor_embedding)[0][0]

        # Lexical similarity
        lexical_sim = self._compute_lexical_score_simple(main_text, neighbor_text)

        # Combined score
        combined_score = self._calculate_clear_score(semantic_sim, lexical_sim, lambda_lex, lambda_sem)

        return semantic_sim, lexical_sim, combined_score, lambda_lex, lambda_sem

    def _get_expanded_context(self, chapter_num, sentence_num, query, book_key, main_text, main_length):
        """Get context sentences with dynamic neighbor scoring"""
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
            prev_row = chapters_df.loc[prev_idx]
            prev_id = self._get_passage_id(prev_row['chapter_number'], prev_row['sentence_number'])

            if prev_id in self.used_passages[book_key]:
                break

            prev_text = prev_row['sentence_text']
            prev_length = len(prev_text.split())

            semantic_sim, lexical_sim, neighbor_score, lambda_lex, lambda_sem = self._compute_neighbor_similarity(
                main_text, main_length, prev_text, prev_length
            )

            is_relevant = neighbor_score >= self.NEIGHBOR_RELEVANCE_THRESHOLD

            context['prev_sentences'].append({
                'sentence_number': prev_row['sentence_number'],
                'sentence_text': prev_text,
                'is_relevant': is_relevant,
                'distance': i,
                'semantic_similarity': semantic_sim,
                'lexical_similarity': lexical_sim,
                'neighbor_score': neighbor_score,
                'lambda_lex': lambda_lex,
                'lambda_sem': lambda_sem
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

            next_text = next_row['sentence_text']
            next_length = len(next_text.split())

            semantic_sim, lexical_sim, neighbor_score, lambda_lex, lambda_sem = self._compute_neighbor_similarity(
                main_text, main_length, next_text, next_length
            )

            is_relevant = neighbor_score >= self.NEIGHBOR_RELEVANCE_THRESHOLD

            context['next_sentences'].append({
                'sentence_number': next_row['sentence_number'],
                'sentence_text': next_text,
                'is_relevant': is_relevant,
                'distance': i,
                'semantic_similarity': semantic_sim,
                'lexical_similarity': lexical_sim,
                'neighbor_score': neighbor_score,
                'lambda_lex': lambda_lex,
                'lambda_sem': lambda_sem
            })

            if is_relevant:
                context['next_relevant_count'] += 1
            else:
                break

        return context

    def _retrieve_passages(self, query, query_analysis, book_key, top_k=9):
        """CLEAR-based hybrid retrieval with dynamic length-based weights"""
        self.used_passages[book_key] = set()

        book_data = self.books_data[book_key]
        chapters_df = book_data['chapters']
        embeddings = book_data['embeddings']

        query_embedding = self.model.encode([query])
        semantic_similarities = cosine_similarity(query_embedding, embeddings)[0]

        candidates = []

        for idx, semantic_sim in enumerate(semantic_similarities):
            if semantic_sim < self.MIN_SEMANTIC_THRESHOLD:
                continue

            row = chapters_df.iloc[idx]
            passage_id = self._get_passage_id(row['chapter_number'], row['sentence_number'])

            if passage_id in self.used_passages[book_key]:
                continue

            sentence_text = row['sentence_text']
            sentence_length = len(sentence_text.split())

            # Dynamic weights based on sentence length
            lambda_lex, lambda_sem = self._compute_dynamic_weights_by_length(sentence_length)

            # Lexical score
            lexical_score = self._compute_lexical_score_weighted(query, sentence_text, query_analysis)

            # Match type classification
            if lexical_score >= 0.95:
                match_type = 'exact'
            elif lexical_score >= 0.3:
                match_type = 'partial_lexical'
            else:
                match_type = 'semantic'

            # Final score
            final_score = self._calculate_clear_score(
                semantic_sim, lexical_score, lambda_lex, lambda_sem, sentence_length
            )

            candidates.append({
                'index': idx,
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

        chapter_counts = {}
        results = []

        for candidate in candidates:
            passage_id = self._get_passage_id(candidate['chapter_number'], candidate['sentence_number'])
            if passage_id in self.used_passages[book_key]:
                continue

            ch_num = candidate['chapter_number']
            if chapter_counts.get(ch_num, 0) >= 3:
                continue

            self.used_passages[book_key].add(passage_id)

            try:
                context = self._get_expanded_context(
                    candidate['chapter_number'], 
                    candidate['sentence_number'], 
                    query, 
                    book_key,
                    candidate['sentence_text'],
                    candidate['word_count']
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

            for sent in context.get('prev_sentences', []):
                self.used_passages[book_key].add(self._get_passage_id(ch_num, sent['sentence_number']))
            for sent in context.get('next_sentences', []):
                self.used_passages[book_key].add(self._get_passage_id(ch_num, sent['sentence_number']))

            candidate['context'] = context
            candidate['has_relevant_context'] = has_relevant_context
            candidate['total_context_sentences'] = len(context['prev_sentences']) + len(context['next_sentences'])

            results.append(candidate)
            chapter_counts[ch_num] = chapter_counts.get(ch_num, 0) + 1

            if len(results) >= top_k:
                break

        return results

    def _get_thematic_classification(self, passages, query, book_key):
        """Classify passages by themes with dynamic length-based weights"""
        if not passages:
            return passages, False, 0.0

        book_data = self.books_data[book_key]
        themes_df = book_data['themes']

        thematic_results = []

        for passage in passages:
            sentence_text = passage['sentence_text']
            sentence_length = len(sentence_text.split())
            sentence_embedding = self.model.encode([sentence_text])

            matching_themes = []

            for idx, theme_row in themes_df.iterrows():
                meaning_text = theme_row['Meaning']
                meaning_length = len(meaning_text.split())

                # Dynamic weights based on meaning entry length relative to sentence
                lambda_lex, lambda_sem = self._compute_dynamic_weights_by_length(meaning_length, sentence_length)

                # Semantic similarity
                meaning_embedding = self.model.encode([meaning_text])
                semantic_sim = cosine_similarity(sentence_embedding, meaning_embedding)[0][0]

                # Lexical similarity
                lexical_sim = self._compute_lexical_score_simple(sentence_text, meaning_text)

                # Combined thematic score
                thematic_score = self._calculate_clear_score(semantic_sim, lexical_sim, lambda_lex, lambda_sem)

                if thematic_score >= self.THEMATIC_THRESHOLD:
                    matching_themes.append({
                        'tagalog_title': theme_row['Tagalog Title'],
                        'meaning': meaning_text,
                        'confidence': thematic_score,
                        'semantic_sim': semantic_sim,
                        'lexical_sim': lexical_sim,
                        'lambda_lex': lambda_lex,
                        'lambda_sem': lambda_sem
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
        """Main query interface with DAPT-enhanced embeddings and relationship filtering"""
        # Phase 1: Query Validation
        is_valid, validation_info = self.query_analyzer.validate_filipino_query(user_query)

        if not is_valid:
            return {
                'type': 'invalid_filipino',
                'validation_info': validation_info,
                'message': f"Invalid Filipino query: {validation_info['reason']}"
            }

        # Check for stopwords-only query
        content_words = self.query_analyzer.get_content_words(user_query)

        if not content_words:
            return {
                'type': 'no_lexical_grounding',
                'overlap_info': {
                    'reason': 'Query contains only stopwords',
                    'content_words': [],
                    'matched_words': {},
                    'total_content_words': 0,
                    'total_matched': 0
                },
                'message': "Query blocked: Query contains only stopwords"
            }

        # Phase 1.5: Word Relationship Analysis (NEW)
        if len(content_words) >= 2:
            relationship_analysis = self.relationship_analyzer.analyze_query_relationships(user_query)
            
            if not relationship_analysis['is_valid']:
                return {
                    'type': 'weak_word_relationships',
                    'relationship_analysis': relationship_analysis,
                    'message': 'Query blocked: Words are not sufficiently related in the corpus'
                }
        else:
            relationship_analysis = None

        # Phase 2: Lexical Presence Check
        missing_words = [w for w in content_words if w not in self.global_vocabulary]
        if missing_words:
            return {
                'type': 'no_lexical_grounding',
                'overlap_info': {
                    'reason': 'Query contains words not found in novels or themes',
                    'content_words': content_words,
                    'missing_words': missing_words,
                    'matched_words': {},
                    'total_content_words': len(content_words),
                    'total_matched': 0
                },
                'relationship_analysis': relationship_analysis,
                'message': "Query blocked: Some words are not present in the novels or theme files"
            }

        # Phase 3 & 4: Lexical & Semantic Scoring + Neighbor Expansion
        query_analysis = self.query_analyzer.analyze_query_words(user_query)
        query_length = len(extract_words(user_query))
        stopword_ratio = self.query_analyzer.get_stopword_ratio(user_query)

        results_by_book = {}
        no_grounding_books = []

        for book_key in self.books_data.keys():
            vocab = self.corpus_vocabulary[book_key]
            book_matches = [w for w in content_words if w in vocab]

            if not book_matches:
                no_grounding_books.append(book_key)
                continue

            passages = self._retrieve_passages(user_query, query_analysis, book_key)

            if passages:
                # Phase 5: Thematic Exploration
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
                    'total_context_sentences': total_context
                }

        matched_words = {
            book_key: [w for w in content_words if w in vocab]
            for book_key, vocab in self.corpus_vocabulary.items()
            if any(w in vocab for w in content_words)
        }

        overlap_info = {
            'content_words': content_words,
            'matched_words': matched_words,
            'total_content_words': len(content_words),
            'total_matched': sum(len(v) for v in matched_words.values()),
            'no_grounding_books': no_grounding_books
        }

        if not results_by_book:
            if len(no_grounding_books) == len(self.books_data):
                return {
                    'type': 'no_lexical_grounding',
                    'overlap_info': overlap_info,
                    'relationship_analysis': relationship_analysis,
                    'message': "No lexical grounding in both novels"
                }
            else:
                return {
                    'type': 'no_matches',
                    'message': "No matches found in either novel",
                    'query_analysis': query_analysis,
                    'query_length': query_length,
                    'stopword_ratio': stopword_ratio,
                    'overlap_info': overlap_info,
                    'relationship_analysis': relationship_analysis
                }

        return {
            'type': 'success',
            'results_by_book': results_by_book,
            'query_length': query_length,
            'stopword_ratio': stopword_ratio,
            'query_analysis': query_analysis,
            'overlap_info': overlap_info,
            'relationship_analysis': relationship_analysis
        }

    def _display_neighbor_similarities(self, context):
        """Display neighbor similarities with dynamic weights"""
        prev_sentences = context.get('prev_sentences', [])
        next_sentences = context.get('next_sentences', [])

        if not prev_sentences and not next_sentences:
            return

        sim_table = Table(
            title="Neighbor Similarity Metrics (Dynamic Weights)",
            show_header=True,
            header_style="bold cyan",
            border_style="cyan",
            box=box.ROUNDED,
            expand=False
        )

        sim_table.add_column("Position", style="bright_white", width=12, justify="center")
        sim_table.add_column("S#", style="bright_yellow", width=6, justify="center")
        sim_table.add_column("Semantic", style="bright_cyan", width=12, justify="center")
        sim_table.add_column("Lexical", style="bright_green", width=12, justify="center")
        sim_table.add_column("Combined", style="bright_magenta", width=12, justify="center")
        sim_table.add_column("λ_lex", style="yellow", width=8, justify="center")
        sim_table.add_column("λ_sem", style="cyan", width=8, justify="center")

        for sent in prev_sentences:
            sim_table.add_row(
                "Previous",
                str(sent['sentence_number']),
                f"{sent['semantic_similarity']:.1%}",
                f"{sent['lexical_similarity']:.1%}",
                f"{sent['neighbor_score']:.1%}",
                f"{sent['lambda_lex']:.2f}",
                f"{sent['lambda_sem']:.2f}"
            )

        for sent in next_sentences:
            sim_table.add_row(
                "Next",
                str(sent['sentence_number']),
                f"{sent['semantic_similarity']:.1%}",
                f"{sent['lexical_similarity']:.1%}",
                f"{sent['neighbor_score']:.1%}",
                f"{sent['lambda_lex']:.2f}",
                f"{sent['lambda_sem']:.2f}"
            )

        self.console.print(sim_table)

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

        stats_panel = Panel(stats_text, style="cyan", box=box.SIMPLE)
        self.console.print(stats_panel)
        self.console.print("\n")

    def display_results(self, response, query=""):
        """Display results with DAPT and relationship analysis information"""
        result_type = response['type']

        # Display relationship analysis if present
        if 'relationship_analysis' in response and response['relationship_analysis']:
            self.relationship_analyzer.display_relationship_analysis(response['relationship_analysis'])

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

        if result_type == 'weak_word_relationships':
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)

            weak_panel = Panel(
                f"{response['message']}\n\n"
                f"The query words do not have sufficient semantic relationships\n"
                f"in the corpus based on DAPT-adapted embeddings.\n\n"
                f"See Word Relationship Analysis above for details.",
                title="❌ Weak Word Relationships",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(weak_panel)
            return

        if result_type == 'no_lexical_grounding':
            overlap_info = response['overlap_info']

            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)

            book_names = {'noli': 'Noli Me Tangere', 'elfili': 'El Filibusterismo'}
            no_ground_books = overlap_info.get('no_grounding_books', [])

            if len(no_ground_books) == len(self.books_data):
                title_text = "❌ No Lexical Grounding in Both Novels"
            elif no_ground_books:
                no_ground_names = [book_names[k] for k in no_ground_books]
                title_text = f"❌ No Lexical Grounding for {', '.join(no_ground_names)}"
            else:
                title_text = "❌ No Lexical Grounding"

            grounding_panel = Panel(
                f"{response['message']}\n\n"
                f"Lexical Grounding Analysis:\n"
                f"  Content words in query: {overlap_info['total_content_words']}\n"
                f"  Query words: {', '.join(overlap_info['content_words']) if overlap_info['content_words'] else 'None'}\n"
                f"  Matched in corpus: {overlap_info['total_matched']}\n"
                f"  Matched by book: {', '.join([f'{book_names[k]}: {len(v)}' for k, v in overlap_info['matched_words'].items()]) if overlap_info['matched_words'] else 'None'}",
                title=title_text,
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

        if 'overlap_info' in response:
            overlap_info = response['overlap_info']

            if 'no_grounding_books' in overlap_info and overlap_info['no_grounding_books']:
                book_names = {'noli': 'Noli Me Tangere', 'elfili': 'El Filibusterismo'}
                no_ground_names = [book_names[k] for k in overlap_info['no_grounding_books']]

                warning_text = f"⚠ No Lexical Grounding for: {', '.join(no_ground_names)}"
                warning_panel = Panel(warning_text, style="yellow", box=box.SIMPLE)
                self.console.print(warning_panel)

            if overlap_info['matched_words']:
                grounding_text = (
                    f"✓ Lexical Grounding: {overlap_info['total_matched']}/{overlap_info['total_content_words']} content words matched | "
                    f"Matched: {', '.join(overlap_info['content_words'][:5])}"
                )
                if len(overlap_info['content_words']) > 5:
                    grounding_text += "..."

                grounding_panel = Panel(grounding_text, style="green", box=box.SIMPLE)
                self.console.print(grounding_panel)

        if 'query_analysis' in response:
            self._display_query_analysis(response['query_analysis'])

        results_by_book = response['results_by_book']

        model_type = "DAPT-Enhanced" if self.use_dapt else "Base"
        header_text = Text(f"{model_type} Results for: '{query}'", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(header_panel)

        book_names = {'noli': 'Noli Me Tangere', 'elfili': 'El Filibusterismo'}
        book_colors = {'noli': 'bright_yellow', 'elfili': 'bright_magenta'}

        for book_key in ['noli', 'elfili']:
            if book_key not in results_by_book:
                continue

            book_results = results_by_book[book_key]
            results = book_results['results']
            has_themes = book_results['has_themes']

            book_title = book_names[book_key]
            book_header = Panel(
                Align.center(Text(f"📖 {book_title} 📖", style=f"bold {book_colors[book_key]}")),
                style=book_colors[book_key],
                box=box.HEAVY,
                padding=(1, 2)
            )
            self.console.print(book_header)

            metrics_text = (
                f"🔍 {'DAPT-ENHANCED' if self.use_dapt else 'BASE'} DYNAMIC DUAL-FORMULA CLEAR SYSTEM 🔍\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"{'CORPUS-ADAPTED EMBEDDINGS (MLM Pretraining)' if self.use_dapt else 'GENERAL MULTILINGUAL EMBEDDINGS'}\n"
                f"MAIN RETRIEVAL: Dynamic weights based on sentence length\n"
                f"NEIGHBOR RETRIEVAL: Dynamic weights based on relative length\n"
                f"THEMATIC EXPLORATION: Dynamic weights based on meaning length\n"
                f"WORD RELATIONSHIPS: Threshold = {self.relationship_threshold:.2f}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Query: {response['query_length']} words | Stopwords: {response['stopword_ratio']:.1%}\n"
                f"Semantic: {book_results['avg_semantic']:.1%} | "
                f"Lexical: {book_results['avg_lexical']:.1%} | "
                f"Final: {book_results['avg_final']:.1%}\n"
                f"Exact: {book_results['exact_matches']} | "
                f"Partial: {book_results['partial_lexical_matches']} | "
                f"Semantic: {book_results['semantic_only_matches']}\n"
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
                main_table.add_column("λ_lex", style="yellow", width=7, justify="center")
                main_table.add_column("λ_sem", style="cyan", width=7, justify="center")

                match_type = result['match_type']
                if match_type == 'exact':
                    type_display = "Exact Match"
                elif match_type == 'partial_lexical':
                    type_display = "Partial Lexical"
                else:
                    type_display = "Semantic"

                main_table.add_row(
                    book_title,
                    str(result['chapter_number']),
                    str(result['sentence_number']),
                    f"{result['semantic_score']:.1%}",
                    f"{result['lexical_score']:.1%}",
                    f"{result['final_score']:.1%}",
                    type_display,
                    f"{result['lambda_lex']:.2f}",
                    f"{result['lambda_sem']:.2f}"
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
                    self.console.print("")
                    self._display_neighbor_similarities(context)

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
                        title="Thematic Analysis (Dynamic Weights)",
                        show_header=True,
                        header_style="bold magenta",
                        border_style="magenta",
                        box=box.SIMPLE,
                        expand=True
                    )

                    theme_table.add_column("Tagalog Title", style="bright_magenta", width=25)
                    theme_table.add_column("Meaning", style="magenta", min_width=30)
                    theme_table.add_column("Confidence", style="bright_cyan", width=11, justify="center")
                    theme_table.add_column("λ_lex", style="yellow", width=7, justify="center")
                    theme_table.add_column("λ_sem", style="cyan", width=7, justify="center")

                    theme_table.add_row(
                        primary_theme['tagalog_title'],
                        primary_theme['meaning'],
                        f"{primary_theme['confidence']:.1%}",
                        f"{primary_theme['lambda_lex']:.2f}",
                        f"{primary_theme['lambda_sem']:.2f}"
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


if __name__ == "__main__":
    # Initialize system with DAPT
    # Set use_dapt=False to use base model without domain adaptation
    # Set force_retrain=True to retrain DAPT model even if it exists
    system = DAPTEnhancedCLEARSystem(
        use_dapt=True,              # Enable DAPT
        dapt_epochs=3,              # Number of MLM training epochs
        relationship_threshold=0.45, # Minimum similarity for word relationships
        force_retrain=False         # Set to True to force DAPT retraining
    )

    model_type = "DAPT-Enhanced" if system.use_dapt else "Base"
    welcome_panel = Panel(
        Align.center(Text(
            f"🔍 {model_type.upper()} DYNAMIC DUAL-FORMULA CLEAR SYSTEM 🔍\n"
            "Noli Me Tangere & El Filibusterismo\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{'✓ Domain-Adaptive Pretraining (MLM)' if system.use_dapt else '✗ No Domain Adaptation'}\n"
            f"✓ Word Relationship Filtering (threshold={system.relationship_threshold:.2f})\n"
            "✓ Main Retrieval: Length-Based Dynamic Weights\n"
            "✓ Neighbor Retrieval: Relative Length-Based Weights\n"
            "✓ Thematic Exploration: Meaning Length-Based Weights\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "XLM-RoBERTa | Official Tagalog Stopwords | Lexical Grounding",
            style="bold white"
        )),
        style="bright_green",
        box=box.HEAVY
    )
    system.console.print(welcome_panel)

    # Display instructions
    instructions_panel = Panel(
        "Enter Filipino queries to search both novels.\n\n"
        "Features:\n"
        "  • Validates Filipino language queries\n"
        "  • Analyzes word relationships using corpus-adapted embeddings\n"
        "  • Filters queries with weak semantic connections\n"
        "  • Shows relationship percentages and strength classifications\n"
        "  • Dynamic hybrid scoring (semantic + lexical)\n"
        "  • Context expansion with neighbor relevance\n"
        "  • Thematic classification of results\n\n"
        "Type 'exit' to quit.",
        title="Instructions",
        style="cyan",
        box=box.ROUNDED
    )
    system.console.print(instructions_panel)

    while True:
        system.console.print("\n" + "─" * 80, style="dim")
        user_input = system.console.input("[bold cyan]Enter query (or 'exit'): [/bold cyan]").strip()

        if user_input.lower() == 'exit':
            goodbye_panel = Panel(
                Align.center(Text(
                    f"Thank you for using the {model_type} Dynamic Dual-Formula CLEAR system!",
                    style="bold green"
                )),
                style="bright_green",
                box=box.ROUNDED
            )
            system.console.print(goodbye_panel)
            break

        if not user_input:
            system.console.print("[red]Please enter a valid query.[/red]")
            continue

        system.console.print(
            f"[dim]Processing '{user_input}' with "
            f"{'DAPT-adapted' if system.use_dapt else 'base'} embeddings and "
            f"relationship filtering...[/dim]"
        )

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
            import traceback
            system.console.print(traceback.format_exc(), style="dim red")