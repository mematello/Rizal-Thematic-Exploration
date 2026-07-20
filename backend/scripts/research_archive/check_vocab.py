import pandas as pd
import os

vocab = set()
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_dir = os.path.join(base_dir, 'csvFiles')
files = ['noli_chapters.csv', 'elfili_chapters.csv', 'fullversion_noli.csv', 'fullversion_elfili.csv']

import re
def extract_words(text):
    return re.findall(r'\b[a-z0-9]+\b', text.lower())

for f in files:
    path = os.path.join(csv_dir, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'sentence_text' in df.columns:
            for text in df['sentence_text'].dropna():
                words = extract_words(str(text))
                vocab.update(words)

print("IS LOVE IN VOCAB?", "love" in vocab)
