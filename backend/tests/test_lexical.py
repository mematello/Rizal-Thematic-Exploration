from app.core.engine import get_engine, extract_words
import numpy as np

buod = "Dahil sa paggigitgitan, natapakan ng tinyente ang laylayan ng damit ni Donya Victorina dahilan kung bakit nagalit ito."
full = "Ipatawad po ninyo; ngunit sa akala ko'y inyong pinakamahal ang aking ama; ¿maaari po bang sabihin ninyo sa akin kung ano ang kanyang kinahinatnan? ang tanong ni Ibarra na siya'y minamasdan. Bakit? hindi po ba ninyo nalalaman? ang tanong ng militar. Itinanong ko kay Kapitang Tiago ay sumagot sa aking hindi niya sasabihin kung di bukas na."

w1 = set(extract_words(buod.lower()))
w2 = set(extract_words(full.lower()))

print(f"w1: {w1}")
print(f"w2: {w2}")
print(f"Intersection: {w1 & w2}")

eng = get_engine()
score1 = eng._compute_simple_lexical(buod, full)
score2 = eng._compute_simple_lexical(full, buod)

print(f"Option 1 (buod, full): {score1}")
print(f"Option 2 (full, buod): {score2}")

# Let's test TF-IDF or Jaccard
jaccard = len(w1 & w2) / len(w1 | w2)
print(f"Jaccard: {jaccard}")

# Removed stopword cleanly

