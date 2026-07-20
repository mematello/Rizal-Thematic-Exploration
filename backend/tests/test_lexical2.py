import json
from app.core._legacy_nlp import is_stopword, analyze_semantic_weight
from app.core.engine import get_engine, extract_words

buod = "Dahil sa paggigitgitan, natapakan ng tinyente ang laylayan ng damit ni Donya Victorina dahilan kung bakit nagalit ito."
full = "Ipatawad po ninyo; ngunit sa akala ko'y inyong pinakamahal ang aking ama; ¿maaari po bang sabihin ninyo sa akin kung ano ang kanyang kinahinatnan? ang tanong ni Ibarra na siya'y minamasdan. Bakit? hindi po ba ninyo nalalaman? ang tanong ng militar. Itinanong ko kay Kapitang Tiago ay sumagot sa aking hindi niya sasabihin kung di bukas na."

eng = get_engine()

# Mock query analysis
words = extract_words(buod.lower())
query_analysis = []
for w in words:
    is_stop = is_stopword(w)
    weight = analyze_semantic_weight(w)
    query_analysis.append({"word": w, "is_stopword": is_stop, "semantic_weight": weight})

print("Query Analysis:")
for q in query_analysis:
    print(f"  {q['word']}: stop={q['is_stopword']}, weight={q['semantic_weight']}")

# Call _compute_lexical_score
score = eng._compute_lexical_score(buod, full, query_analysis, 0.0)
print(f"Computed Lexical Score: {score}")

