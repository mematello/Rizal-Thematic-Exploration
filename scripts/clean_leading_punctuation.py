import pandas as pd
import re

files = [
    '/Users/marcusoliver/Desktop/Rizal-Thematic-Exploration/csvFiles/fullversion_elfili.csv',
    '/Users/marcusoliver/Desktop/Rizal-Thematic-Exploration/csvFiles/fullversion_noli.csv'
]

def clean_text(text):
    if pd.isna(text):
        return text
    # Remove any leading non-letter characters (including Spanish chars)
    # ^ : start of string
    # [^...] : matches characters NOT in the list
    # The list contains standard letters + Spanish accented letters
    # + : one or more times
    return re.sub(r'^[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+', '', str(text)).strip()

if __name__ == "__main__":
    for filepath in files:
        print(f"Loading {filepath}...")
        df = pd.read_csv(filepath)
        
        old_sentences = df['sentence_text'].copy()
        
        df['sentence_text'] = df['sentence_text'].apply(clean_text)
        
        changed = (old_sentences != df['sentence_text']).sum()
        print(f"Cleaned {changed} sentences containing leading punctuation in {filepath}.")
        
        df.to_csv(filepath, index=False)
        print(f"Saved cleaned CSV: {filepath}\n")
