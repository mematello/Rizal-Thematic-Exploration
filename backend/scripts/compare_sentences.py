
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_sentences():
    model_name = os.getenv('BERT_MODEL_NAME', 'sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    
    if len(sys.argv) > 2:
        s1 = sys.argv[1]
        s2 = sys.argv[2]
    else:
        s1 = "edukasyon"
        s2 = "Isinantabi niya ang kanyang pag-aaral."
        print(f"Usage: python compare_sentences.py \"s1\" \"s2\"")
        print(f"Using defaults: '{s1}' vs '{s2}'\n")

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    words1 = s1.split()
    words2 = s2.split()

    vecs1 = model.encode(words1)
    vecs2 = model.encode(words2)

    header = " " * 15 + "".join([f"{w:>12}" for w in words2])
    print(header)
    print("-" * len(header))

    for i, w1 in enumerate(words1):
        label = (w1[:12] + '..') if len(w1) > 14 else w1
        row = f"{label:<15}|"
        for j, w2 in enumerate(words2):
            sim = cosine_similarity(vecs1[i], vecs2[j])
            row += f" {sim:10.4f}"
        print(row)

    v_s1 = model.encode(s1)
    v_s2 = model.encode(s2)
    overall = cosine_similarity(v_s1, v_s2)
    print(f"\nOverall Sentence Similarity: {overall:.4f} ({round(overall*100)}%)")

if __name__ == "__main__":
    compare_sentences()
