"""
Data loading and preprocessing module.
"""
import pandas as pd
import numpy as np
import itertools
from sentence_transformers import SentenceTransformer
from .config import BOOKS_CONFIG, CORE_ENTITIES
from .utils import extract_words
from .errors import DataLoadingError

class DataLoader:
    """
    Handles loading of books, computing embeddings, and building vocabulary resources.
    """
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.books_data = {}
        self.global_passages = []
        self.global_passage_embeddings = []
        self.global_theme_texts = []
        self.global_theme_embeddings = []
        
        self.corpus_vocabulary = {}
        self.global_vocabulary = set()
        
        self.token_case_counts = {}
        self.global_token_case_counts = {}
        
        self.entity_list = []
        self.co_occurrence_matrix = None
        self.passage_embeddings_matrix = None
        self.theme_embeddings_matrix = None
        
        self.ready = False

    def load(self, query_analyzer):
        """Execute full loading pipeline."""
        self._load_books()
        self._compute_all_embeddings()
        self._build_corpus_vocabulary(query_analyzer)
        self._build_semantic_grounding_resources()
        self.ready = True
        return self

    def _load_books(self):
        """Load data for Noli Me Tangere and El Filibusterismo from CSVs."""
        for book_key, chapters_file, themes_file in BOOKS_CONFIG:
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
            except FileNotFoundError as e:
                raise DataLoadingError(f"Could not find critical data file: {e.filename}")

    def _compute_all_embeddings(self):
        """Compute embeddings for all books."""
        for book_key, book_data in self.books_data.items():
            chapters_df = book_data['chapters']
            themes_df = book_data['themes']

            # Prepare Chapter Text
            chapters_df['combined_text'] = (
                chapters_df['chapter_title'].astype(str) + " " +
                chapters_df['sentence_text'].astype(str)
            )
            chapters_df['sentence_word_count'] = (
                chapters_df['sentence_text'].astype(str).apply(lambda x: len(x.split()))
            )

            texts = chapters_df['combined_text'].tolist()
            book_data['embeddings'] = self.model.encode(texts, show_progress_bar=False)

            # Store Global Passages
            for row, embedding in zip(chapters_df.itertuples(index=False), book_data['embeddings']):
                self.global_passages.append({
                    'book_key': book_key,
                    'chapter_number': row.chapter_number,
                    'chapter_title': row.chapter_title,
                    'sentence_number': row.sentence_number,
                    'sentence_text': row.sentence_text
                })
                self.global_passage_embeddings.append(embedding)

            # Prepare Theme Text
            themes_df['theme_text'] = (
                themes_df['Tagalog Title'].astype(str) + " " +
                themes_df['Meaning'].astype(str)
            )

            theme_texts = themes_df['theme_text'].tolist()
            book_data['theme_embeddings'] = self.model.encode(theme_texts, show_progress_bar=False)
            self.global_theme_texts.extend(theme_texts)
            self.global_theme_embeddings.extend(book_data['theme_embeddings'])

    def _build_corpus_vocabulary(self, query_analyzer):
        """Build vocabulary from corpus and themes."""
        self.token_case_counts = {}
        self.global_token_case_counts = {}

        for book_key, book_data in self.books_data.items():
            vocabulary = set()

            for text in book_data['chapters']['sentence_text'].astype(str):
                # Vocabulary (lowercased)
                tokens_lower = extract_words(text.lower())
                vocabulary.update(tokens_lower)

                # Case counts for NER heuristics
                tokens_original = extract_words(str(text))
                for tok in tokens_original:
                    low = tok.lower()
                    is_cap = tok[:1].isupper()
                    
                    if book_key not in self.token_case_counts:
                        self.token_case_counts[book_key] = {}
                    d = self.token_case_counts[book_key].setdefault(low, {'cap': 0, 'lower': 0})
                    if is_cap: d['cap'] += 1
                    else: d['lower'] += 1

                    gd = self.global_token_case_counts.setdefault(low, {'cap': 0, 'lower': 0})
                    if is_cap: gd['cap'] += 1
                    else: gd['lower'] += 1

            for text in book_data['themes']['Meaning'].astype(str):
                vocabulary.update(extract_words(text.lower()))

            vocabulary -= query_analyzer.STOPWORDS
            self.corpus_vocabulary[book_key] = vocabulary
            self.global_vocabulary.update(vocabulary)

    def _build_semantic_grounding_resources(self):
        """Aggregate embeddings and prepare entity graphs."""
        embedding_dim = self.model.get_sentence_embedding_dimension()

        if self.global_passage_embeddings:
            self.passage_embeddings_matrix = np.vstack(self.global_passage_embeddings)
        else:
            self.passage_embeddings_matrix = np.empty((0, embedding_dim))
        
        # Clear list to save memory
        self.global_passage_embeddings = []

        if self.global_theme_embeddings:
            self.theme_embeddings_matrix = np.vstack(self.global_theme_embeddings)
        else:
            self.theme_embeddings_matrix = np.empty((0, embedding_dim))
        
        self.global_theme_embeddings = []

        self.entity_list = self._derive_entity_list()
        
        sentences = [entry['sentence_text'] for entry in self.global_passages]
        if self.entity_list and sentences:
            self.co_occurrence_matrix = self.build_co_occurrence(sentences, self.entity_list)
        else:
            self.co_occurrence_matrix = pd.DataFrame()

    def _derive_entity_list(self):
        """Construct entity inventory using case statistics."""
        entities = set(CORE_ENTITIES)
        for token, counts in self.global_token_case_counts.items():
            cap_count = counts.get('cap', 0)
            lower_count = counts.get('lower', 0)
            if cap_count >= 3 and cap_count >= lower_count and len(token) > 2 and token.isalpha():
                entities.add(token)
        return sorted(entities)
    
    def extract_entities(self, text, spacy_nlp=None):
        """
        Simple rule-based entity extraction.
        Combines capitalization heuristics with curated entity vocabulary.
        """
        if not text:
            return []

        tokens = extract_words(str(text))
        entities = []
        entity_set = set(self.entity_list)
        
        for token in tokens:
            token_lower = token.lower()
            if token_lower in entity_set and token_lower not in entities:
                entities.append(token_lower)

        # Attempt spaCy NER if available
        if spacy_nlp:
            try:
                doc = spacy_nlp(str(text))
                for ent in doc.ents:
                    ent_lower = ent.text.lower()
                    if ent_lower in entity_set and ent_lower not in entities:
                        entities.append(ent_lower)
            except Exception:
                pass

        return entities

    def build_co_occurrence(self, sentences, entities):
        """Count entity pair co-occurrences at the sentence level."""
        entity_order = sorted(set(entities))
        matrix = pd.DataFrame(0, index=entity_order, columns=entity_order, dtype=int)
        entity_index = set(entity_order)

        for sentence in sentences:
            # We use self.extract_entities here without spaCy for speed in bulk build
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

    def get_passage_id(self, chapter_num, sentence_num):
        """Create unique identifier for passages."""
        return (int(chapter_num), int(sentence_num))
