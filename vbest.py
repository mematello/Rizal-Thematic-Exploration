import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import re
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box
import itertools

warnings.filterwarnings('ignore')

WORD_PATTERN = re.compile(r'[0-9a-zA-ZÀ-ÿñÑ]+(?:-[0-9a-zA-ZÀ-ÿñÑ]+)*')

def extract_words(text):
    """Extract words, preserving hyphenated tokens as single units."""
    return WORD_PATTERN.findall(text)

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


class CleanNoliSystem:
    """
    Dynamic Dual-Formula CLEAR System
    - Main Retrieval: Dynamic weights based on sentence length
    - Neighbor Retrieval: Dynamic weights based on relative lengths
    - Thematic Exploration: Dynamic weights based on meaning entry length
    """

    CORE_ENTITIES = {
        'ibarra', 'crisostomo', 'crisóstomo', 'simoun', 'maria', 'clara', 'elias',
        'basilio', 'sisa', 'kapitan', 'tiago', 'tiyago', 'tasio', 'pilosopo',
        'juli', 'isagani', 'paulita', 'salvi', 'damaso', 'camorra', 'camora',
        'kabesang', 'tales', 'kundiman', 'ben', 'zayb', 'donya', 'victorina'
    }

    def __init__(self):
        self.console = Console()
        self.console.print("Loading XLM-RoBERTa model...", style="cyan")
        self.model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

        self.console.print("Loading datasets...", style="cyan")
        self.books_data = {}
        self.used_passages = {}
        self.corpus_vocabulary = {}
        self.global_vocabulary = set()
        self.global_passages = []
        self.global_passage_embeddings = []
        self.global_theme_texts = []
        self.global_theme_embeddings = []
        self.entity_list = []
        self.co_occurrence_matrix = None
        self.semantic_grounding_ready = False
        self._load_books()

        self.console.print("Initializing query analyzer with official stopwords...", style="cyan")
        self.query_analyzer = QueryAnalyzer()

        # spaCy for dependency parsing (best-effort; fall back to regex if unavailable)
        self.console.print("Initializing spaCy for relation parsing...", style="cyan")
        self.spacy_nlp = None
        try:
            import spacy
            try:
                # Try a multilingual model with UD dependencies
                self.spacy_nlp = spacy.load("xx_sent_ud_sm")
            except Exception:
                # Fallback to a blank multilingual pipeline with tagger/dep may not be available
                self.spacy_nlp = spacy.blank("xx")
        except Exception:
            self.spacy_nlp = None

        self.console.print("Computing embeddings for all books...", style="cyan")
        self._compute_all_embeddings()

        self.console.print("Building corpus vocabulary...", style="cyan")
        self._build_corpus_vocabulary()

        self.console.print("Building semantic grounding resources...", style="cyan")
        self._build_semantic_grounding_resources()

        # System parameters
        self.MIN_SEMANTIC_THRESHOLD = 0.20
        self.THEMATIC_THRESHOLD = 0.45
        self.NEIGHBOR_RELEVANCE_THRESHOLD = 0.40
        self.SHORT_SENTENCE_THRESHOLD = 5
        self.SHORT_SENTENCE_PENALTY = 0.08
        self.MAX_CONTEXT_EXPANSION = 5
        self.HIGH_STOPWORD_RATIO = 0.6
        self.STOPWORD_PENALTY_FACTOR = 0.5

        # Domain coherence (DAPT-inspired) parameters
        # If average pairwise similarity among content words falls below this, reject
        self.DOMAIN_COHERENCE_THRESHOLD = 0.38
        # Minimum number of content words to run coherence check
        self.DOMAIN_MIN_WORDS = 2
        # Treat words whose per-word avg similarity is far below global avg as outliers
        self.DOMAIN_OUTLIER_DELTA = 0.10
        # Relation validation thresholds
        self.RELATION_SIM_THRESHOLD = 0.40
        self.RELATION_COOCC_THRESHOLD = 3  # minimal co-occurrence count in corpus sentences
        self.RELATION_COOCC_THRESHOLD_NAMED = 1  # relaxed threshold for named entities
        self.RELATION_ENABLE_TSNE = False  # default to PCA for speed

        # Semantic query validator thresholds
        self.SEMANTIC_SIMILARITY_THRESHOLD = 0.4  # Primary: average embedding similarity threshold
        self.HIGH_SEMANTIC_SIMILARITY_THRESHOLD = 0.75  # Secondary: high similarity when co-occurrence = 0
        self.MIN_COOCCURRENCE_NORMAL = 1  # Minimum co-occurrence for normal cases
        self.MIN_COOCCURRENCE_STRICT = 3  # Minimum co-occurrence for strict cases

        self.console.print("Dynamic Dual-Formula CLEAR system ready!", style="bold green")

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

            chapters_df['combined_text'] = (
                chapters_df['chapter_title'].astype(str) + " " +
                chapters_df['sentence_text'].astype(str)
            )

            chapters_df['sentence_word_count'] = (
                chapters_df['sentence_text'].astype(str).apply(lambda x: len(x.split()))
            )

            texts = chapters_df['combined_text'].tolist()
            book_data['embeddings'] = self.model.encode(texts, show_progress_bar=False)
            for row, embedding in zip(chapters_df.itertuples(index=False), book_data['embeddings']):
                self.global_passages.append({
                    'book_key': book_key,
                    'chapter_number': row.chapter_number,
                    'chapter_title': row.chapter_title,
                    'sentence_number': row.sentence_number,
                    'sentence_text': row.sentence_text
                })
                self.global_passage_embeddings.append(embedding)

            themes_df['theme_text'] = (
                themes_df['Tagalog Title'].astype(str) + " " +
                themes_df['Meaning'].astype(str)
            )

            theme_texts = themes_df['theme_text'].tolist()
            book_data['theme_embeddings'] = self.model.encode(theme_texts, show_progress_bar=False)
            self.global_theme_texts.extend(theme_texts)
            self.global_theme_embeddings.extend(book_data['theme_embeddings'])

    def _build_corpus_vocabulary(self):
        """Build vocabulary from corpus and themes for lexical presence check"""
        # Initialize case count trackers
        self.token_case_counts = {}
        self.global_token_case_counts = {}

        for book_key, book_data in self.books_data.items():
            vocabulary = set()

            for text in book_data['chapters']['sentence_text'].astype(str):
                # vocabulary (lowercased)
                tokens_lower = extract_words(text.lower())
                vocabulary.update(tokens_lower)

                # case counts for proper-noun heuristic
                tokens_original = extract_words(str(text))
                for tok in tokens_original:
                    low = tok.lower()
                    is_cap = tok[:1].isupper()
                    if book_key not in self.token_case_counts:
                        self.token_case_counts[book_key] = {}
                    d = self.token_case_counts[book_key].setdefault(low, {'cap': 0, 'lower': 0})
                    if is_cap:
                        d['cap'] += 1
                    else:
                        d['lower'] += 1

                    gd = self.global_token_case_counts.setdefault(low, {'cap': 0, 'lower': 0})
                    if is_cap:
                        gd['cap'] += 1
                    else:
                        gd['lower'] += 1

            for text in book_data['themes']['Meaning'].astype(str):
                vocabulary.update(extract_words(text.lower()))

            vocabulary -= self.query_analyzer.STOPWORDS

            self.corpus_vocabulary[book_key] = vocabulary
            self.global_vocabulary.update(vocabulary)
            self.console.print(f"  {book_key} corpus vocabulary: {len(vocabulary)} content words", style="green")

    def _build_semantic_grounding_resources(self):
        """
        Build corpus-wide resources required for semantic grounding validation.
        Aggregates passage/theme embeddings and prepares entity co-occurrence graphs.
        """
        embedding_dim = self.model.get_sentence_embedding_dimension()

        if self.global_passage_embeddings:
            self.passage_embeddings_matrix = np.vstack(self.global_passage_embeddings)
        else:
            self.passage_embeddings_matrix = np.empty((0, embedding_dim))
        self.global_passage_embeddings = []

        if self.global_theme_embeddings:
            self.theme_embeddings_matrix = np.vstack(self.global_theme_embeddings)
        else:
            self.theme_embeddings_matrix = np.empty((0, embedding_dim))
        self.global_theme_embeddings = []

        self.entity_list = self._derive_entity_list()
        self.entity_set = set(self.entity_list)

        sentences = [entry['sentence_text'] for entry in self.global_passages]
        if self.entity_list and sentences:
            self.co_occurrence_matrix = self.build_co_occurrence(sentences, self.entity_list)
        else:
            self.co_occurrence_matrix = pd.DataFrame()

        self.semantic_grounding_ready = True

    def _derive_entity_list(self):
        """
        Construct an entity inventory using case statistics and predefined core entities.
        Acts as a lightweight NER substitute tailored to Rizal's novels.
        """
        entities = set(self.CORE_ENTITIES)

        for token, counts in getattr(self, 'global_token_case_counts', {}).items():
            cap_count = counts.get('cap', 0)
            lower_count = counts.get('lower', 0)
            if cap_count >= 3 and cap_count >= lower_count and len(token) > 2 and token.isalpha():
                entities.add(token)

        return sorted(entities)

    def extract_entities(self, text):
        """
        Simple rule-based entity extraction.
        Combines capitalisation heuristics with curated entity vocabulary.
        """
        if not text:
            return []

        tokens = extract_words(str(text))
        entities = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in getattr(self, 'entity_set', set()) and token_lower not in entities:
                entities.append(token_lower)

        # Attempt spaCy named entity recognition if available for extra recall
        if self.spacy_nlp:
            try:
                doc = self.spacy_nlp(str(text))
                for ent in doc.ents:
                    ent_lower = ent.text.lower()
                    if ent_lower in self.entity_set and ent_lower not in entities:
                        entities.append(ent_lower)
            except Exception:
                # spaCy pipeline may be unavailable for some languages; ignore errors silently.
                pass

        return entities

    def build_co_occurrence(self, sentences, entities):
        """
        Count entity pair co-occurrences at the sentence level to approximate narrative proximity.
        """
        entity_order = sorted(set(entities))
        matrix = pd.DataFrame(0, index=entity_order, columns=entity_order, dtype=int)
        entity_index = set(entity_order)

        for sentence in sentences:
            sentence_entities = set(self.extract_entities(sentence))
            sentence_entities &= entity_index

            if len(sentence_entities) < 2:
                continue

            for e1, e2 in itertools.combinations(sorted(sentence_entities), 2):
                if e1 in matrix.index and e2 in matrix.columns:
                    matrix.loc[e1, e2] += 1
                    matrix.loc[e2, e1] += 1

        np.fill_diagonal(matrix.values, 0)
        return matrix

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

    def _sanitize_text(self, text):
        """Normalize whitespace and ensure strings are JSON-safe."""
        if text is None:
            return ""
        sanitized = " ".join(str(text).strip().split())
        return sanitized.replace('"', "'")

    def shorten_sentence(self, text, max_words=12):
        """
        Truncate sentences while preserving readability for suggestion strings.
        Ensures output remains JSON-safe.
        """
        if not text:
            return ""

        words = str(text).strip().split()
        if not words:
            return ""

        if len(words) > max_words:
            snippet = " ".join(words[:max_words]) + " ..."
        else:
            snippet = " ".join(words)

        return self._sanitize_text(snippet)

    def filter_by_theme_alignment(self, candidates, top_k=3, min_similarity=0.35):
        """
        Filter and rank candidate suggestions using theme embeddings for grounding.

        Args:
            candidates (list[dict]): [{'text': str, 'score': float, 'embedding': np.ndarray}, ...]
            top_k (int): number of suggestions to return.
            min_similarity (float): minimum theme similarity required to keep a suggestion.

        Returns:
            list[str]: Top-k suggestion texts passing theme alignment.
        """
        if not candidates:
            return []

        sanitized = []
        embeddings = []
        for candidate in candidates:
            text = self._sanitize_text(candidate.get('text', ''))
            if not text:
                continue
            sanitized.append({
                'text': text,
                'score': float(candidate.get('score', 0.0))
            })

            embedding = candidate.get('embedding')
            if embedding is not None:
                embeddings.append(np.asarray(embedding))
            else:
                embeddings.append(self.model.encode([text], show_progress_bar=False)[0])

        if not sanitized:
            return []

        embeddings_matrix = np.vstack(embeddings)
        if getattr(self, 'theme_embeddings_matrix', None) is not None and self.theme_embeddings_matrix.size > 0:
            theme_scores = cosine_similarity(embeddings_matrix, self.theme_embeddings_matrix).max(axis=1)
        else:
            theme_scores = np.zeros(len(sanitized))

        enriched = []
        for idx, candidate in enumerate(sanitized):
            theme_score = float(theme_scores[idx])
            candidate['theme_score'] = theme_score
            candidate['embedding'] = embeddings_matrix[idx]
            if theme_score >= min_similarity:
                enriched.append(candidate)

        if not enriched:
            enriched = sorted(
                sanitized,
                key=lambda c: (c.get('theme_score', c.get('score', 0.0)), c.get('score', 0.0)),
                reverse=True
            )[:top_k]
        else:
            enriched.sort(key=lambda c: (c['theme_score'], c.get('score', 0.0)), reverse=True)

        return [candidate['text'] for candidate in enriched[:top_k]]

    def generate_recovery_suggestions(self, user_query, failure_context=None, top_k=3, query_embedding=None):
        """
        Generate Tier 1 recovery suggestions when a query fails validation.

        Args:
            user_query (str): Original query text.
            failure_context (dict): Optional metadata about failure point.
            top_k (int): Number of suggestions to return.
            query_embedding (np.ndarray): Optional pre-computed embedding for the query.
        """
        if not getattr(self, 'semantic_grounding_ready', False):
            return []

        if self.passage_embeddings_matrix.size == 0 or not getattr(self, 'global_passages', None):
            return []

        if query_embedding is None:
            query_vec = self.model.encode([user_query], show_progress_bar=False)[0]
        else:
            query_vec = np.asarray(query_embedding)
            if query_vec.ndim > 1:
                query_vec = query_vec.flatten()

        similarities = cosine_similarity(
            query_vec.reshape(1, -1),
            self.passage_embeddings_matrix
        )[0]

        top_indices = np.argsort(similarities)[::-1]
        candidates = []
        seen_texts = set()

        for idx in top_indices:
            passage = self.global_passages[idx]
            sentence = self.shorten_sentence(passage.get('sentence_text', ''), max_words=14)

            if not sentence or sentence.lower() == self._sanitize_text(user_query).lower():
                continue

            if sentence in seen_texts:
                continue

            is_valid, validation_info = self.query_analyzer.validate_filipino_query(sentence)
            if not is_valid:
                continue

            content_words = self.query_analyzer.get_content_words(sentence)
            if not content_words:
                continue

            candidates.append({
                'text': sentence,
                'score': float(similarities[idx]),
                'embedding': self.passage_embeddings_matrix[idx]
            })
            seen_texts.add(sentence)

            if len(candidates) >= top_k * 4:
                break

        filtered = self.filter_by_theme_alignment(candidates, top_k=top_k)
        if not filtered:
            filtered = [
                candidate['text']
                for candidate in sorted(candidates, key=lambda c: c.get('score', 0.0), reverse=True)[:top_k]
            ]

        return filtered[:top_k]

    def generate_followup_suggestions(self, user_query, results_by_book, top_k=3):
        """
        Generate Tier 2 follow-up suggestions after a successful retrieval.

        Args:
            user_query (str): Original user query.
            results_by_book (dict): Retrieval results organized by book.
            top_k (int): Number of follow-up suggestions to return.
        """
        if not results_by_book:
            return []

        candidate_passages = []
        for book_data in results_by_book.values():
            candidate_passages.extend(book_data.get('results', []))

        if not candidate_passages:
            return []

        candidate_passages.sort(key=lambda p: p.get('final_score', 0.0), reverse=True)

        candidates = []
        seen = set()
        normalized_query = self._sanitize_text(user_query).lower()

        for passage in candidate_passages:
            sentence_text = passage.get('sentence_text', '')
            chapter_title = passage.get('chapter_title', '')
            primary_theme = passage.get('primary_theme')

            if primary_theme:
                base_suggestion = f"{primary_theme.get('tagalog_title', '').strip()} sa {chapter_title}".strip()
            else:
                entities = self.extract_entities(sentence_text)
                if entities:
                    base_suggestion = f"{entities[0].title()} sa {chapter_title}".strip()
                else:
                    base_suggestion = self.shorten_sentence(sentence_text, max_words=12)

            suggestion_text = self._sanitize_text(base_suggestion)

            if not suggestion_text or suggestion_text.lower() == normalized_query:
                continue

            if suggestion_text in seen:
                continue

            candidates.append({
                'text': suggestion_text,
                'score': passage.get('final_score', 0.0)
            })
            seen.add(suggestion_text)

            if len(candidates) >= top_k * 4:
                break

        filtered = self.filter_by_theme_alignment(candidates, top_k=top_k, min_similarity=0.30)

        if not filtered:
            fallback = []
            seen_fallback = set()
            for passage in candidate_passages:
                primary_theme = passage.get('primary_theme')
                if primary_theme:
                    theme_title = self._sanitize_text(primary_theme.get('tagalog_title', ''))
                    if not theme_title or theme_title.lower() == normalized_query:
                        continue
                    if theme_title in seen or theme_title in seen_fallback:
                        continue
                    fallback.append(theme_title)
                    seen_fallback.add(theme_title)
                if len(fallback) >= top_k:
                    break
            filtered = fallback

        return filtered[:top_k]

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

    def _validate_semantic_grounding(self, query, query_embedding):
        """
        Multi-level semantic grounding validation inspired by DAPT.
        Ensures that queries align with canonical passages, themes, and entity co-occurrences.
        """
        if not self.semantic_grounding_ready:
            return True, None

        query_vec = np.asarray(query_embedding)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        elif query_vec.ndim > 2:
            query_vec = query_vec.reshape(1, -1)

        if self.passage_embeddings_matrix.size == 0:
            return True, None

        # Level 1: Maximum Passage Similarity Check
        max_passage_sim = float(np.max(cosine_similarity(query_vec, self.passage_embeddings_matrix)))
        if np.isnan(max_passage_sim) or max_passage_sim < 0.20:
            return False, "No semantically similar passages found"

        # Level 2: Thematic Alignment Check
        if self.theme_embeddings_matrix.size > 0:
            max_theme_sim = float(np.max(cosine_similarity(query_vec, self.theme_embeddings_matrix)))
            if np.isnan(max_theme_sim) or max_theme_sim < 0.30:
                return False, "Query not aligned with any valid theme"

        # Level 3: Entity Co-occurrence Check
        entities = self.extract_entities(query)
        if len(entities) >= 2 and self.co_occurrence_matrix is not None and not self.co_occurrence_matrix.empty:
            for e1, e2 in itertools.combinations(entities, 2):
                if (
                    e1 not in self.co_occurrence_matrix.index
                    or e2 not in self.co_occurrence_matrix.columns
                    or self.co_occurrence_matrix.loc[e1, e2] == 0
                ):
                    return False, f"'{e1}' and '{e2}' never appear in related contexts"

        return True, None

    def _generate_semantic_suggestions(self, query_embedding, top_k=3):
        """
        Suggest alternative, corpus-grounded queries based on nearest passages/themes.
        """
        return self.generate_recovery_suggestions(
            user_query="",
            failure_context={'source': 'semantic_grounding'},
            top_k=top_k,
            query_embedding=query_embedding
        )

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

    def _retrieve_passages(self, query, query_analysis, book_key, top_k=9, query_embedding=None):
        """CLEAR-based hybrid retrieval with dynamic length-based weights"""
        self.used_passages[book_key] = set()

        book_data = self.books_data[book_key]
        chapters_df = book_data['chapters']
        embeddings = book_data['embeddings']

        if query_embedding is None:
            query_embedding_vec = self.model.encode([query])
        else:
            query_embedding_vec = np.asarray(query_embedding)
            if query_embedding_vec.ndim == 1:
                query_embedding_vec = query_embedding_vec.reshape(1, -1)
        semantic_similarities = cosine_similarity(query_embedding_vec, embeddings)[0]

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
        """Main query interface with dynamic dual-formula scoring"""
        # Phase 1: Query Validation
        is_valid, validation_info = self.query_analyzer.validate_filipino_query(user_query)

        if not is_valid:
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={'stage': 'filipino_validation', 'details': validation_info}
            )
            return {
                'type': 'invalid_filipino',
                'validation_info': validation_info,
                'message': f"Invalid Filipino query: {validation_info['reason']}",
                'suggestions': recovery_suggestions
            }

        # Check for stopwords-only query
        content_words = self.query_analyzer.get_content_words(user_query)

        if not content_words:
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={'stage': 'stopword_only'}
            )
            return {
                'type': 'no_lexical_grounding',
                'overlap_info': {
                    'reason': 'Query contains only stopwords',
                    'content_words': [],
                    'matched_words': {},
                    'total_content_words': 0,
                    'total_matched': 0
                },
                'message': "Query blocked: Query contains only stopwords",
                'suggestions': recovery_suggestions
            }

        # Phase 2: Lexical Presence Check
        missing_words = [w for w in content_words if w not in self.global_vocabulary]
        if missing_words:
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={
                    'stage': 'lexical_presence',
                    'missing_words': missing_words
                }
            )
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
                'message': "Query blocked: Some words are not present in the novels or theme files",
                'suggestions': recovery_suggestions
            }

        # Phase 2.5: Semantic Query Validation (Embedding Similarity + Co-occurrence)
        semantic_validation = self._validate_semantic_query(content_words)
        if not semantic_validation['proceed']:
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={
                    'stage': 'semantic_validation',
                    'details': semantic_validation
                }
            )
            return {
                'type': 'semantic_validation_failed',
                'message': f"Query blocked: {semantic_validation['reason']}",
                'content_words': content_words,
                'avg_similarity': semantic_validation['avg_similarity'],
                'min_cooccurrence': semantic_validation['min_cooccurrence'],
                'word_pairs': semantic_validation['word_pairs'],
                'reason': semantic_validation['reason'],
                'suggestions': recovery_suggestions
            }
        # Validation passed - proceed with query processing
        # (semantic_validation['reason'] contains the pass reason if needed for logging)

        query_embedding_vector = None

        # Phase 2.6: Domain Coherence Check (DAPT-inspired)
        # Ensure multi-word queries have semantically coherent content words within domain
        if len(content_words) >= self.DOMAIN_MIN_WORDS:
            sim_info = self._compute_domain_coherence(content_words)

            overall_avg = sim_info['overall_avg']
            per_word_avg = sim_info['per_word_avg']
            outliers = [w for w, avg in per_word_avg.items() if avg + self.DOMAIN_OUTLIER_DELTA < overall_avg]

            if overall_avg < self.DOMAIN_COHERENCE_THRESHOLD or outliers:
                recovery_suggestions = self.generate_recovery_suggestions(
                    user_query,
                    failure_context={
                        'stage': 'domain_coherence',
                        'details': sim_info,
                        'outliers': outliers
                    }
                )
                return {
                    'type': 'domain_incoherent',
                    'message': (
                        "Query blocked: Words are not semantically coherent within the domain"
                    ),
                    'content_words': content_words,
                    'similarity_matrix': sim_info['similarity_matrix'],
                    'per_word_avg': per_word_avg,
                    'overall_avg': overall_avg,
                    'outliers': outliers,
                    'suggestions': recovery_suggestions
                }

        # Phase 2.7: Relation Consistency Check (e.g., "X ng Y", "A ni B")
        relation_info = self._analyze_relations(user_query, content_words)
        if relation_info and not relation_info['passed']:
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={
                    'stage': 'relation_consistency',
                    'details': relation_info
                }
            )
            return {
                'type': 'domain_incoherent',
                'message': "Query blocked: Relation(s) are semantically inconsistent within the domain",
                'content_words': content_words,
                'similarity_matrix': relation_info['coherence'].get('similarity_matrix') if relation_info.get('coherence') else None,
                'per_word_avg': relation_info['coherence'].get('per_word_avg') if relation_info.get('coherence') else None,
                'overall_avg': relation_info['coherence'].get('overall_avg') if relation_info.get('coherence') else None,
                'outliers': relation_info.get('outliers', []),
                'relations': relation_info.get('relations', []),
                'relation_scores': relation_info.get('relation_scores', []),
                'plot_path': relation_info.get('plot_path'),
                'suggestions': recovery_suggestions
            }

        # Stage 5: Semantic Grounding Validation (DAPT-inspired)
        query_embedding_vector = self.model.encode([user_query])[0]
        grounding_ok, grounding_reason = self._validate_semantic_grounding(user_query, query_embedding_vector)
        if not grounding_ok:
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={
                    'stage': 'semantic_grounding',
                    'reason': grounding_reason
                },
                top_k=3,
                query_embedding=query_embedding_vector
            )
            return {
                'type': 'semantic_grounding_rejected',
                'status': 'rejected',
                'reason': grounding_reason or "Query concept not found in novel context.",
                'suggestions': recovery_suggestions or ["Basahin muli ang kabanata para sa mas angkop na paksa."]
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

            passages = self._retrieve_passages(
                user_query,
                query_analysis,
                book_key,
                query_embedding=query_embedding_vector
            )

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
            recovery_suggestions = self.generate_recovery_suggestions(
                user_query,
                failure_context={'stage': 'no_results', 'details': overlap_info},
                top_k=3,
                query_embedding=query_embedding_vector
            )
            if len(no_grounding_books) == len(self.books_data):
                return {
                    'type': 'no_lexical_grounding',
                    'overlap_info': overlap_info,
                    'message': "No lexical grounding in both novels",
                    'suggestions': recovery_suggestions
                }
            else:
                return {
                    'type': 'no_matches',
                    'message': "No matches found in either novel",
                    'query_analysis': query_analysis,
                    'query_length': query_length,
                    'stopword_ratio': stopword_ratio,
                    'overlap_info': overlap_info,
                    'suggestions': recovery_suggestions
                }

        next_queries = self.generate_followup_suggestions(
            user_query,
            results_by_book,
            top_k=3
        )

        return {
            'type': 'success',
            'results_by_book': results_by_book,
            'query_length': query_length,
            'stopword_ratio': stopword_ratio,
            'query_analysis': query_analysis,
            'overlap_info': overlap_info,
            'suggestions': [],
            'next_queries': next_queries
        }

    def _compute_domain_coherence(self, words):
        """Compute pairwise cosine similarities among words and summarize stats."""
        # Encode individual words (lowercased)
        tokens = [w.lower() for w in words]
        embeddings = self.model.encode(tokens, show_progress_bar=False)

        # Similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Per-word average similarity to others (exclude self)
        per_word_avg = {}
        for i, w in enumerate(tokens):
            if len(tokens) == 1:
                per_word_avg[w] = 1.0
            else:
                row = np.delete(sim_matrix[i], i)
                per_word_avg[w] = float(np.mean(row)) if row.size > 0 else 0.0

        # Global average across upper triangle (excluding diagonal)
        if len(tokens) <= 1:
            overall_avg = 1.0
        else:
            triu_indices = np.triu_indices(len(tokens), k=1)
            overall_avg = float(np.mean(sim_matrix[triu_indices])) if triu_indices[0].size > 0 else 0.0

        return {
            'tokens': tokens,
            'similarity_matrix': sim_matrix,
            'per_word_avg': per_word_avg,
            'overall_avg': overall_avg
        }

    def _validate_semantic_query(self, content_words):
        """
        Semantic query validator: checks embedding similarity and co-occurrence.
        
        Rules:
        1. Primary: Average word embedding similarity ≥ threshold (0.4)
        2. Secondary: If co-occurrence = 0, allow only if similarity ≥ 0.75
        3. Otherwise: enforce normal co-occurrence rules (≥ 1 or 3)
        
        Returns: dict with 'proceed' (bool), 'reason' (str), and diagnostic info
        """
        if len(content_words) < 2:
            # Single word queries pass (no pairs to check)
            return {
                'proceed': True,
                'reason': 'Single word query - no semantic validation needed',
                'avg_similarity': 1.0,
                'min_cooccurrence': 0,
                'word_pairs': []
            }

        # Compute word embeddings
        tokens = [w.lower() for w in content_words]
        embeddings = self.model.encode(tokens, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        # Compute average pairwise similarity (excluding diagonal)
        triu_indices = np.triu_indices(len(tokens), k=1)
        avg_similarity = float(np.mean(sim_matrix[triu_indices])) if triu_indices[0].size > 0 else 0.0

        # Compute co-occurrence for all word pairs
        word_pairs = []
        min_cooccurrence = float('inf')
        
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                word1 = tokens[i]
                word2 = tokens[j]
                cooccurrence = self._count_cooccurrence(word1, word2)
                similarity = float(sim_matrix[i][j])
                
                word_pairs.append({
                    'word1': word1,
                    'word2': word2,
                    'similarity': similarity,
                    'cooccurrence': cooccurrence
                })
                
                if cooccurrence < min_cooccurrence:
                    min_cooccurrence = cooccurrence

        # Apply validation rules
        # Rule 1: Primary condition - average similarity must meet threshold
        if avg_similarity < self.SEMANTIC_SIMILARITY_THRESHOLD:
            return {
                'proceed': False,
                'reason': f'Blocked due to low semantic similarity (avg: {avg_similarity:.2f} < {self.SEMANTIC_SIMILARITY_THRESHOLD})',
                'avg_similarity': avg_similarity,
                'min_cooccurrence': min_cooccurrence if min_cooccurrence != float('inf') else 0,
                'word_pairs': word_pairs
            }

        # Rule 2: Secondary condition - if co-occurrence = 0, require high similarity
        if min_cooccurrence == 0:
            if avg_similarity >= self.HIGH_SEMANTIC_SIMILARITY_THRESHOLD:
                return {
                    'proceed': True,
                    'reason': 'Passed via high semantic similarity (co-occurrence = 0, but similarity ≥ 0.75)',
                    'avg_similarity': avg_similarity,
                    'min_cooccurrence': 0,
                    'word_pairs': word_pairs
                }
            else:
                return {
                    'proceed': False,
                    'reason': f'Blocked due to zero co-occurrence and low similarity (similarity: {avg_similarity:.2f} < {self.HIGH_SEMANTIC_SIMILARITY_THRESHOLD})',
                    'avg_similarity': avg_similarity,
                    'min_cooccurrence': 0,
                    'word_pairs': word_pairs
                }

        # Rule 3: Normal co-occurrence rules
        # Use strict threshold (3) if similarity is borderline, normal (1) if similarity is good
        if avg_similarity < 0.5:
            required_cooccurrence = self.MIN_COOCCURRENCE_STRICT
        else:
            required_cooccurrence = self.MIN_COOCCURRENCE_NORMAL

        if min_cooccurrence < required_cooccurrence:
            return {
                'proceed': False,
                'reason': f'Blocked due to low co-occurrence (min: {min_cooccurrence} < {required_cooccurrence})',
                'avg_similarity': avg_similarity,
                'min_cooccurrence': min_cooccurrence,
                'word_pairs': word_pairs,
                'required_cooccurrence': required_cooccurrence
            }

        # All checks passed
        return {
            'proceed': True,
            'reason': f'Passed semantic validation (similarity: {avg_similarity:.2f}, co-occurrence: {min_cooccurrence})',
            'avg_similarity': avg_similarity,
            'min_cooccurrence': min_cooccurrence,
            'word_pairs': word_pairs
        }

    def _extract_relations_regex(self, text):
        """Fallback relation extraction using simple patterns for Tagalog possessives/markers.
        Detect patterns: <X> ng <Y>, <A> ni <B>
        Returns list of tuples: { 'pattern': 'ng'|'ni', 'left': str, 'right': str, 'span': str }
        """
        tokens = extract_words(text.lower())
        relations = []
        for i, tok in enumerate(tokens):
            if tok in {"ng", "ni"} and i - 1 >= 0 and i + 1 < len(tokens):
                left = tokens[i - 1]
                right = tokens[i + 1]
                span = f"{left} {tok} {right}"
                relations.append({'pattern': tok, 'left': left, 'right': right, 'span': span})
        return relations

    def _extract_relations_spacy(self, text):
        """Try to extract relations with spaCy. If not robust, return regex fallback."""
        if not self.spacy_nlp or not hasattr(self.spacy_nlp, "pipe"):
            return self._extract_relations_regex(text)
        try:
            doc = self.spacy_nlp(text)
        except Exception:
            return self._extract_relations_regex(text)

        # Heuristic: look for tokens literally 'ng' or 'ni' connecting nouns
        relations = []
        words = [t.text for t in doc]
        for i, t in enumerate(doc):
            tok_lower = t.text.lower()
            if tok_lower in {"ng", "ni"} and i - 1 >= 0 and i + 1 < len(doc):
                left = doc[i - 1].text.lower()
                right = doc[i + 1].text.lower()
                span = f"{left} {tok_lower} {right}"
                relations.append({'pattern': tok_lower, 'left': left, 'right': right, 'span': span})
        if relations:
            return relations
        return self._extract_relations_regex(text)

    def _count_cooccurrence(self, left, right):
        """Count sentence-level co-occurrence of two terms in both corpora."""
        total = 0
        for book_key, book_data in self.books_data.items():
            sentences = book_data['chapters']['sentence_text'].astype(str).str.lower()
            # word-boundary contains check
            left_re = re.compile(rf"\b{re.escape(left)}\b")
            right_re = re.compile(rf"\b{re.escape(right)}\b")
            for s in sentences:
                if left_re.search(s) and right_re.search(s):
                    total += 1
        return total

    def _visualize_embeddings(self, labels, vectors, out_path):
        """Reduce to 2D and save a scatter plot (PCA or t-SNE)."""
        arr = np.array(vectors)
        if arr.shape[0] <= 1:
            return None
        if arr.shape[0] < 4 or not self.RELATION_ENABLE_TSNE:
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(arr)
            title = "PCA of Query Tokens"
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, arr.shape[0]-1))
            coords = reducer.fit_transform(arr)
            title = "t-SNE of Query Tokens"

        plt.figure(figsize=(4.8, 4.0), dpi=120)
        x, y = coords[:, 0], coords[:, 1]
        plt.scatter(x, y, c='tab:blue')
        for i, lab in enumerate(labels):
            plt.annotate(lab, (x[i], y[i]), textcoords="offset points", xytext=(5, 3), fontsize=8)
        plt.title(title)
        plt.tight_layout()
        try:
            plt.savefig(out_path)
            plt.close()
            return out_path
        except Exception:
            plt.close()
            return None

    def _analyze_relations(self, query_text, content_words):
        """Analyze relations and compute similarity between phrase and its components, plus co-occurrence.
        Returns dict with keys: passed(bool), relations, relation_scores(list), outliers, coherence(optional), plot_path(optional)
        """
        relations = self._extract_relations_spacy(query_text)
        if not relations:
            return {'passed': True, 'relations': [], 'relation_scores': []}

        # Compute base coherence for content words for context/report
        coherence = None
        try:
            coherence = self._compute_domain_coherence(content_words)
        except Exception:
            coherence = None

        scores = []
        any_fail = False
        labels = []
        vectors = []

        for rel in relations:
            left = rel['left']
            right = rel['right']
            span = rel['span']

            # Embeddings
            span_vec = self.model.encode([span], show_progress_bar=False)[0]
            left_vec = self.model.encode([left], show_progress_bar=False)[0]
            right_vec = self.model.encode([right], show_progress_bar=False)[0]
            comp_avg_vec = (left_vec + right_vec) / 2.0

            # Similarities
            span_to_avg = float(cosine_similarity([span_vec], [comp_avg_vec])[0][0])
            left_right_sim = float(cosine_similarity([left_vec], [right_vec])[0][0])

            # Co-occurrence in corpus sentences
            coocc = self._count_cooccurrence(left, right)

            # Proper-noun heuristic: relax co-occurrence threshold for likely names with 'ni'
            right_lc = right.lower()
            cap_counts = self.global_token_case_counts.get(right_lc, {'cap': 0, 'lower': 0})
            looks_named = cap_counts.get('cap', 0) >= 3 or right_lc in {
                'ibarra','crisostomo','crisóstomo','maria','clara','damaso','salvi','tasio','basilio','sisa','elias','kapitan','tiago','tiyago'
            }

            if rel['pattern'] == 'ni' and looks_named and span_to_avg >= max(self.RELATION_SIM_THRESHOLD, 0.65):
                coocc_threshold = self.RELATION_COOCC_THRESHOLD_NAMED
            else:
                coocc_threshold = self.RELATION_COOCC_THRESHOLD

            passed = (span_to_avg >= self.RELATION_SIM_THRESHOLD) and (coocc >= coocc_threshold)
            if not passed:
                any_fail = True

            scores.append({
                'relation': rel,
                'span_to_components_sim': span_to_avg,
                'left_right_sim': left_right_sim,
                'cooccurrences': coocc,
                'coocc_threshold': coocc_threshold,
                'passed': passed
            })

            # For visualization
            for lab, vec in [(left, left_vec), (right, right_vec), (span, span_vec)]:
                labels.append(lab)
                vectors.append(vec)

        plot_path = None
        try:
            # Deduplicate identical labels by appending indices to keep readable
            disamb_labels = []
            seen = {}
            for lab in labels:
                if lab not in seen:
                    seen[lab] = 0
                    disamb_labels.append(lab)
                else:
                    seen[lab] += 1
                    disamb_labels.append(f"{lab}_{seen[lab]}")
            plot_path = self._visualize_embeddings(disamb_labels, vectors, out_path="relation_plot.png")
        except Exception:
            plot_path = None

        # Identify outliers using coherence if available
        outliers = []
        if coherence:
            overall = coherence['overall_avg']
            per_word = coherence['per_word_avg']
            outliers = [w for w, avg in per_word.items() if avg + self.DOMAIN_OUTLIER_DELTA < overall]

        return {
            'passed': not any_fail,
            'relations': relations,
            'relation_scores': scores,
            'outliers': outliers,
            'coherence': coherence,
            'plot_path': plot_path
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

    def _render_suggestions(self, title, items, border_style="cyan"):
        """Render suggestion lists in a consistent table format."""
        if not items:
            return

        suggestion_table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
            border_style=border_style,
            box=box.ROUNDED,
            expand=False
        )
        suggestion_table.add_column("#", style="bright_white", width=4, justify="center")
        suggestion_table.add_column("Suggestion", style="bright_green", justify="left")

        for idx, suggestion in enumerate(items, 1):
            suggestion_table.add_row(str(idx), suggestion)

        self.console.print(suggestion_table)

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
        """Display results with dynamic weight information"""
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
            self._render_suggestions("Recovery Suggestions", response.get('suggestions', []))
            return

        if result_type == 'semantic_grounding_rejected':
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("rejected")
            self.console.print(none_table)

            suggestions = response.get('suggestions', [])
            rejection_panel = Panel(
                f"{response.get('reason', 'Query concept not found in novel context.')}\n\n"
                "The system blocks hallucinated events by grounding every query "
                "in canonical passages, validated themes, and entity co-occurrences.",
                title="Semantic Grounding Validation",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(rejection_panel)
            self._render_suggestions("Recovery Suggestions", suggestions)
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
            self._render_suggestions("Recovery Suggestions", response.get('suggestions', []))
            return

        if result_type == 'semantic_validation_failed':
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("Block")
            self.console.print(none_table)

            avg_sim = response.get('avg_similarity', 0.0)
            min_coocc = response.get('min_cooccurrence', 0)
            word_pairs = response.get('word_pairs', [])
            reason = response.get('reason', 'Semantic validation failed')

            # Word pairs table
            if word_pairs:
                pairs_table = Table(
                    title="Word Pair Analysis",
                    show_header=True,
                    header_style="bold cyan",
                    border_style="cyan",
                    box=box.ROUNDED,
                    expand=False
                )
                pairs_table.add_column("Word 1", style="bright_white")
                pairs_table.add_column("Word 2", style="bright_white")
                pairs_table.add_column("Similarity", style="bright_yellow", justify="center")
                pairs_table.add_column("Co-occurrence", style="bright_magenta", justify="center")

                for pair in word_pairs:
                    sim_val = pair['similarity']
                    coocc_val = pair['cooccurrence']
                    
                    # Color code similarity
                    if sim_val >= 0.7:
                        sim_style = "bright_green"
                    elif sim_val >= 0.4:
                        sim_style = "yellow"
                    else:
                        sim_style = "red"
                    
                    # Color code co-occurrence
                    if coocc_val >= 3:
                        coocc_style = "bright_green"
                    elif coocc_val >= 1:
                        coocc_style = "yellow"
                    else:
                        coocc_style = "red"
                    
                    pairs_table.add_row(
                        pair['word1'],
                        pair['word2'],
                        f"[{sim_style}]{sim_val:.2f}[/{sim_style}]",
                        f"[{coocc_style}]{coocc_val}[/{coocc_style}]"
                    )
                
                self.console.print(pairs_table)

            summary_text = (
                f"{reason}\n\n"
                f"Validation Metrics:\n"
                f"  Average Similarity: {avg_sim:.2f} (threshold: {self.SEMANTIC_SIMILARITY_THRESHOLD})\n"
                f"  Minimum Co-occurrence: {min_coocc}\n"
                f"  Content Words: {', '.join(response.get('content_words', []))}"
            )

            validation_panel = Panel(
                summary_text,
                title="Semantic Query Validation Failed",
                style="red",
                box=box.ROUNDED
            )
            self.console.print(validation_panel)
            self._render_suggestions("Recovery Suggestions", response.get('suggestions', []))
            return

        if result_type == 'domain_incoherent':
            # Display rejection with similarity matrix and per-word stats
            none_table = Table(show_header=False, box=box.HEAVY, border_style="red", width=20)
            none_table.add_column("Result", style="bold red", justify="center")
            none_table.add_row("none")
            self.console.print(none_table)

            words = response['content_words']
            sim = response.get('similarity_matrix')
            per_word_avg = response.get('per_word_avg', {})
            overall = response.get('overall_avg', 0.0)
            outliers = response.get('outliers', [])
            relations = response.get('relations', [])
            relation_scores = response.get('relation_scores', [])
            plot_path = response.get('plot_path')

            # Similarity matrix table
            if sim is not None:
                sim_table = Table(title="Word Embedding Similarity Matrix", show_header=True, header_style="bold cyan", border_style="cyan", box=box.ROUNDED, expand=False)
                sim_table.add_column("Word", style="bright_white", justify="left")
                for w in words:
                    sim_table.add_column(w, style="bright_yellow", justify="center")

                for i, w in enumerate(words):
                    row_vals = []
                    for j in range(len(words)):
                        val = float(sim[i][j]) if i < len(sim) and j < len(sim) else 0.0
                        # Color by similarity strength
                        if i == j:
                            cell = "—"
                        else:
                            if val >= 0.70:
                                color = "bright_green"
                            elif val >= 0.50:
                                color = "yellow"
                            elif val >= 0.35:
                                color = "orange3"
                            else:
                                color = "red"
                            cell = f"[{color}]{val:.2f}[/{color}]"
                        row_vals.append(cell)
                    sim_table.add_row(w, *row_vals)

                self.console.print(sim_table)

            # Per-word averages table
            avg_table = Table(title="Per-Word Average Similarity", show_header=True, header_style="bold magenta", border_style="magenta", box=box.SIMPLE, expand=False)
            avg_table.add_column("Word", style="bright_white")
            avg_table.add_column("Avg Sim", style="bright_cyan", justify="center")
            avg_table.add_column("Status", style="bright_white", justify="center")

            for w in words:
                avg = per_word_avg.get(w.lower(), 0.0)
                status = "OK"
                style = "green"
                if avg < self.DOMAIN_COHERENCE_THRESHOLD or w.lower() in [o.lower() for o in outliers]:
                    status = "Outlier"
                    style = "red"
                avg_table.add_row(w, f"{avg:.2f}", f"[{style}]{status}[/{style}]")

            summary_lines = [
                f"Overall Avg Similarity: {overall:.2f} | Threshold: {self.DOMAIN_COHERENCE_THRESHOLD:.2f}",
                f"Outliers: {', '.join(outliers) if outliers else 'None'}"
            ]

            # Relation analysis table
            if relation_scores:
                rel_table = Table(title="Relation Consistency", show_header=True, header_style="bold blue", border_style="blue", box=box.SIMPLE, expand=False)
                rel_table.add_column("Relation", style="bright_white")
                rel_table.add_column("Sim(span vs comp)", style="bright_cyan", justify="center")
                rel_table.add_column("Sim(left-right)", style="bright_yellow", justify="center")
                rel_table.add_column("Co-occur", style="bright_magenta", justify="center")
                rel_table.add_column("Status", style="bright_white", justify="center")
                for sc in relation_scores:
                    rel = sc['relation']
                    status = "OK" if sc['passed'] else "Fail"
                    style = "green" if sc['passed'] else "red"
                    rel_table.add_row(
                        rel['span'],
                        f"{sc['span_to_components_sim']:.2f}",
                        f"{sc['left_right_sim']:.2f}",
                        f"{sc['cooccurrences']} (≥{sc.get('coocc_threshold', self.RELATION_COOCC_THRESHOLD)})",
                        f"[{style}]{status}[/{style}]"
                    )
                self.console.print(rel_table)
                summary_lines.append(
                    f"Relation threshold: sim>={self.RELATION_SIM_THRESHOLD:.2f} & coocc>={self.RELATION_COOCC_THRESHOLD}"
                )

            if plot_path:
                summary_lines.append(f"Embedding plot saved: {plot_path}")

            summary_panel = Panel(
                "\n".join(summary_lines),
                title="Domain Coherence Summary",
                style="red" if overall < self.DOMAIN_COHERENCE_THRESHOLD or outliers or any(not s['passed'] for s in relation_scores) else "green",
                box=box.ROUNDED
            )

            self.console.print(avg_table)
            self.console.print(summary_panel)
            self._render_suggestions("Recovery Suggestions", response.get('suggestions', []))
            return

        if result_type != 'success':
            error_panel = Panel(
                f"{response['message']}",
                title="No Results",
                style="yellow",
                box=box.ROUNDED
            )
            self.console.print(error_panel)
            self._render_suggestions("Recovery Suggestions", response.get('suggestions', []))

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

        header_text = Text(f"Results for: '{query}'", style="bold white")
        header_panel = Panel(
            Align.center(header_text),
            style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )
        self.console.print(header_panel)

        book_names = {'noli': 'Noli Me Tangere', 'elfili': 'El Filibusterismo'}
        book_colors = {'noli': 'bright_yellow', 'elfili': 'bright_magenta'}

        self._render_suggestions("Next Suggested Queries", response.get('next_queries', []), border_style="magenta")

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
                f"🔍 DYNAMIC DUAL-FORMULA CLEAR SYSTEM 🔍\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"MAIN RETRIEVAL: Dynamic weights based on sentence length\n"
                f"NEIGHBOR RETRIEVAL: Dynamic weights based on relative length\n"
                f"THEMATIC EXPLORATION: Dynamic weights based on meaning length\n"
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
    system = CleanNoliSystem()

    welcome_panel = Panel(
        Align.center(Text(
            "🔍 DYNAMIC DUAL-FORMULA CLEAR SYSTEM 🔍\n"
            "Noli Me Tangere & El Filibusterismo\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Main Retrieval: Length-Based Dynamic Weights\n"
            "Neighbor Retrieval: Relative Length-Based Weights\n"
            "Thematic Exploration: Meaning Length-Based Weights\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "XLM-RoBERTa | Official Tagalog Stopwords | Lexical Grounding",
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
                Align.center(Text("Thank you for using the Dynamic Dual-Formula CLEAR system!", style="bold green")),
                style="bright_green",
                box=box.ROUNDED
            )
            system.console.print(goodbye_panel)
            break

        if not user_input:
            system.console.print("[red]Please enter a valid query.[/red]")
            continue

        system.console.print(f"[dim]Processing '{user_input}' with dynamic length-based scoring...[/dim]")

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