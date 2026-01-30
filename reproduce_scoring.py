
import sys
import numpy as np

# Add backend to path
sys.path.append('backend')
from app.core.analyzer import QueryAnalyzer, extract_words

def verify_scoring():
    analyzer = QueryAnalyzer()
    
    query = "Edukasyon"
    sentence = "Ang edukasyon ay mahalaga para sa lahat."
    
    # 1. Analyze Query
    print(f"\n--- Analyzing Query: '{query}' ---")
    analysis = analyzer.analyze_query_words(query)
    for item in analysis:
        print(f"Word: {item['word']}, Stopword: {item['is_stopword']}, Weight: {item['semantic_weight']}")
        
    # 2. Extract Words from Sentence
    print(f"\n--- Extracting Words from Sentence ---")
    words = extract_words(sentence.lower())
    print(f"Extracted: {words}")
    
    # 3. Compute Lexical Score (Manual)
    sentence_words = set(words)
    matched_weight = 0.0
    total_weight = sum(item['semantic_weight'] for item in analysis)
    
    print(f"\nTotal Weight: {total_weight}")
    
    for item in analysis:
        if item['word'].lower() in sentence_words:
            print(f"MATCH: {item['word']}")
            matched_weight += item['semantic_weight']
        else:
            print(f"MISS: {item['word']}")
            
    score = matched_weight / total_weight if total_weight > 0 else 0
    print(f"Lexical Score: {score}")

if __name__ == "__main__":
    verify_scoring()
