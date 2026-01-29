"""
Query analysis and validation module.
"""
from .utils import extract_words

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
            return tagalog_stops
        except ImportError:
            # Fallback
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
