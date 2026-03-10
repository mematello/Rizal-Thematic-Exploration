
import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env to get model name
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))
load_dotenv(env_path)

# Force offline mode
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_sentences():
    model_name = os.getenv('BERT_MODEL_NAME', 'sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    
    # Use arguments if provided, else defaults
    if len(sys.argv) > 2:
        s1 = sys.argv[1]
        s2 = sys.argv[2]
    else:
        # Defaults for demonstration
        s1 = "edukasyon"
        s2 = "Isinantabi niya ang kanyang pag-aaral."
        print("Usage: python compare_sentences.py \"sentence 1\" \"sentence 2\"")
        print(f"Using defaults: '{s1}' vs '{s2}'\n")

    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    words1 = s1.split()
    words2 = s2.split()

    print(f"\nComparing '{s1}'")
    print(f"with '{s2}'\n")

    # Encode words
    vecs1 = model.encode(words1)
    vecs2 = model.encode(words2)

    # Build similarity matrix
    header = " " * 15 + "".join([f"{w:>12}" for w in words2])
    print(header)
    print("-" * len(header))

    for i, w1 in enumerate(words1):
        # Truncate long words for display
        label = (w1[:12] + '..') if len(w1) > 14 else w1
        row = f"{label:<15}|"
        for j, w2 in enumerate(words2):
            sim = cosine_similarity(vecs1[i], vecs2[j])
            row += f" {sim:10.4f}"
        print(row)

    # Overall Similarity
    v_s1 = model.encode(s1)
    v_s2 = model.encode(s2)
    overall = cosine_similarity(v_s1, v_s2)
    print(f"\nOverall Sentence Similarity: {overall:.4f} ({round(overall*100)}%)")

if __name__ == "__main__":
    compare_sentences()
