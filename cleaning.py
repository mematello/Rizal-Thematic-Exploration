import pandas as pd
import re

def clean_chapter_text(text):
    """
    Remove patterns like '14 noypi.com.ph/noli-me-tangere-buod' from text
    """
    if pd.isna(text):
        return text
    
    # Pattern to match: number followed by 'noypi.com.ph/noli-me-tangere-buod'
    pattern = r'\d+\s*noypi\.com\.ph/noli-me-tangere-buod'
    
    # Remove all occurrences of this pattern
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text.strip()

# Read the CSV file
df = pd.read_csv('noli_corpus.csv')  # Replace with your CSV file path

# Apply the cleaning function to the chapter_text column
df['chapter_text'] = df['chapter_text'].apply(clean_chapter_text)

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_noli_corpus.csv', index=False)

print("Cleaning completed! Saved to 'cleaned_file.csv'")