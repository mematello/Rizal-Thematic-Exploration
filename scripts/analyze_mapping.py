import pandas as pd
import numpy as np

# Load CSVs
full_noli = pd.read_csv('csvFiles/fullversion_noli.csv').dropna(subset=['sentence_text'])
buod_noli = pd.read_csv('csvFiles/noli_chapters.csv').dropna(subset=['sentence_text'])

# 1. Noli Chapter 64 Problem
print("--- Noli Chapter Comparison ---")
f63 = full_noli[full_noli.chapter_number == 63]
b64 = buod_noli[buod_noli.chapter_number == 64]

print(f"Full 63 Title: {f63.chapter_title.unique()}")
print(f"Full 63 Last Sentences: {f63.sentence_text.tail(2).tolist()}")
print(f"Buod 64 Title: {b64.chapter_title.unique()}")
print(f"Buod 64 First Sentences: {b64.sentence_text.head(2).tolist()}")

# 2. Structural Signals (Dialogue markers, Sentence lengths)
print("\n--- Structural Signals (Noli Full) ---")
has_quotes = full_noli.sentence_text.str.contains('"', regex=False).mean()
print(f"Quote density (approx sentences with quotes): {has_quotes:.2%}")
lengths = full_noli.sentence_text.str.len()
print(f"Sentence length stats: Mean={lengths.mean():.1f}, Std={lengths.std():.1f}, Max={lengths.max()}")

# 3. Large Chapter Analysis (Mapping Windows)
print("\n--- Large Chapter Ratio Analysis ---")
def check_ratio(full_df, buod_df, book, chap_num):
    f_count = len(full_df[full_df.chapter_number == chap_num])
    b_count = len(buod_df[buod_df.chapter_number == chap_num])
    if b_count > 0:
        print(f"{book} Chap {chap_num}: Full={f_count}, Buod={b_count}, Ratio={f_count/b_count:.2f}:1")

check_ratio(full_noli, buod_noli, 'Noli', 32)
check_ratio(full_noli, buod_noli, 'Noli', 40)
check_ratio(full_noli, buod_noli, 'Noli', 55)

# 4. Character Aliases vs characterData.ts
# Read characterData.ts for aliases
with open('frontend/lib/characterData.ts', 'r', encoding='utf-8') as f:
    char_data = f.read()

print("\n--- Alias Check ---")
if 'Juan Crisostomo Ibarra y Magsalin' in char_data:
    print("Found 'Juan Crisostomo Ibarra y Magsalin' in characterData.ts")
else:
    print("NOT FOUND: 'Juan Crisostomo Ibarra y Magsalin' in characterData.ts")
