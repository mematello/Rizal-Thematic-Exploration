import pandas as pd
import re
import os

filepath = '/Users/marcusoliver/Desktop/Rizal-Thematic-Exploration/csvFiles/fullversion_elfili.csv'

if __name__ == "__main__":
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Store old values for comparison
    old_sentences = df['sentence_text'].copy()
    
    def clean_text(text):
        if pd.isna(text):
            return text
        # Remove leading em-dashes (—) and optional whitespace
        return re.sub(r'^—\s*', '', str(text))
    
    df['sentence_text'] = df['sentence_text'].apply(clean_text)
    
    # Count how many changed
    changed = (old_sentences != df['sentence_text']).sum()
    print(f"Cleaned {changed} sentences containing leading em-dashes.")
    
    df.to_csv(filepath, index=False)
    print("Saved cleaned CSV.")
