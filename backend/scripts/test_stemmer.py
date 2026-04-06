from app.core.tagalog_stemmer import TagalogStemmer

def test_stemmer():
    stemmer = TagalogStemmer()
    
    test_cases = {
        "punongkahoy": "kahoy",
        "inumaga": "umaga",
        "araw-araw": "araw",
        "kumain": "kain",
        "pagkamatay": "matay", # matay/patay are roots
        "basahin": "basa",
        "naglalakad": "lakad", # wait, "nag" -> "lalakad", it's a bit harder, but let's test basic
        "masaya": "saya"
    }
    
    print("Test Cases:")
    for word, expected in test_cases.items():
        stemmed = stemmer.stem(word)
        print(f"{word} -> {stemmed} (Expected: {expected})")
        
if __name__ == "__main__":
    test_stemmer()
