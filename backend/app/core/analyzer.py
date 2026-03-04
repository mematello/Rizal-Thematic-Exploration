import re
try:
    from stopwordsiso import stopwords
except ImportError:
    stopwords = None

WORD_PATTERN = re.compile(r'[0-9a-zA-ZÀ-ÿñÑ]+(?:-[0-9a-zA-ZÀ-ÿñÑ]+)*')

def extract_words(text):
    return WORD_PATTERN.findall(text)

class QueryAnalyzer:
    def __init__(self):
        self.STOPWORDS = self._load_official_stopwords()

    def _load_official_stopwords(self):
        try:
            if stopwords:
                # Use the library's Tagalog stopword list
                return set(stopwords('tl'))
        except Exception as e:
            print(f"Error loading stopwords: {e}")
        # If library fails, return an empty set to avoid total failure, 
        # but the primary directive is to use the library.
        return set()

    def get_word_frequency(self, word, lang='tl'):
        try:
            from wordfreq import word_frequency
            return word_frequency(word.lower(), lang)
        except Exception:
            return 0.0

    def analyze_query_words(self, query):
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
                'semantic_weight': semantic_weight
            })
        return analysis

    def get_stopword_ratio(self, query):
        words = extract_words(query)
        if not words:
            return 0.0
        stopword_count = sum(1 for w in words if w.lower() in self.STOPWORDS)
        return stopword_count / len(words)
