import re

class TagalogStemmer:
    """
    A lightweight, regex-based stemming utility for Tagalog,
    optimized specifically for generating lexical roots for TF-IDF matching.
    """
    
    def __init__(self):
        # Order of stripping:
        # Pre-fix -> Suffix -> Infix -> Recuplication
        self.prefixes = sorted([
            'pinag', 'nakipag', 'pakikipag', 'makipag', 'ipag',
            'nagpa', 'magpa', 'pagpa', 'pagka', 'naka', 'maka',
            'nang', 'mang', 'pang',
            'nag', 'mag', 'pag',
            'nam', 'mam', 'pam',
            'nan', 'man', 'pan',
            'ika', 'ipa', 'isa',
            'na', 'ma', 'pa', 'ka', 'i', 'um', 'in'
        ], key=len, reverse=True)
        
        self.suffixes = sorted(['hin', 'han', 'in', 'an'], key=len, reverse=True)
        self.infixes = ['um', 'in']
        
        self.vowels = 'aeiou'

    def stem(self, word: str) -> str:
        word = word.lower().strip()
        
        # 1. Reduplication (e.g., araw-araw -> araw, gabi-gabi -> gabi)
        if '-' in word:
            parts = word.split('-')
            if len(parts) == 2 and parts[0] == parts[1]:
                word = parts[0]
            else:
                # remove hyphens for prefix handling
                word = word.replace('-', '')
                
        original_word = word
        
        # 3. Strip Infixes (-in-, -um-)
        for infix in self.infixes:
            if len(word) > 4:
                if word.startswith(infix) and word[0] in self.vowels:
                     word = word[len(infix):]
                     break
                elif len(word) > len(infix)+1 and word[1:1+len(infix)] == infix and word[0] not in self.vowels:
                     word = word[0] + word[1+len(infix):]
                     break
                     
        # 4. Strip Prefixes
        for prefix in self.prefixes:
            if word.startswith(prefix) and len(word) - len(prefix) >= 3:
                word = word[len(prefix):]
                break
                
        # 5. Strip Suffixes (-in, -an, etc.)
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break

        # Reduplication on Root (like lalakad -> lakad, titingin -> tingin)
        if len(word) > 4 and word[0:2] == word[2:4]: # e.g. baba (babae? no)
            pass
        elif len(word) > 3 and word[0] not in self.vowels and word[0:2] == word[0] + word[2]:
            # lalakad -> l-a-lakad -> lakad? word[0]='l', word[1]='a', word[2]='l', word[3]='a'.
            if word[0:2] == word[2:4]:
                word = word[2:]

        if len(word) <= 2:
            return original_word
            
        return word

    def stem_sentence(self, sentence: str) -> str:
        """Stems an entire sentence."""
        # Replace punctuation with spaces
        clean_text = re.sub(r'[^\w\s]', ' ', sentence.lower())
        words = clean_text.split()
        return " ".join([self.stem(w) for w in words])
