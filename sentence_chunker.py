import pandas as pd
import re
from typing import List

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling common abbreviations and edge cases.
    Designed to work well with Filipino and Spanish text patterns.
    """
    if not text or pd.isna(text):
        return []
    
    # Clean the text
    text = str(text).strip()
    
    # Common abbreviations that shouldn't trigger sentence breaks
    abbreviations = [
        r'Dr\.', r'Sr\.', r'Sra\.', r'Don\.', r'Doña\.', 
        r'St\.', r'Sto\.', r'Sta\.', r'P\.', r'pp\.',
        r'etc\.', r'vs\.', r'i\.e\.', r'e\.g\.',
        r'Jr\.', r'Sr\.', r'Mr\.', r'Mrs\.', r'Ms\.'
    ]
    
    # Replace abbreviations temporarily
    temp_replacements = {}
    for i, abbrev in enumerate(abbreviations):
        placeholder = f"__ABBREV_{i}__"
        text = re.sub(abbrev, placeholder, text, flags=re.IGNORECASE)
        temp_replacements[placeholder] = abbrev.replace(r'\.', '.')
    
    # Split on sentence-ending punctuation followed by whitespace and capital letter
    # This handles periods, exclamation marks, and question marks
    sentences = re.split(r'([.!?]+)\s+(?=[A-Z])', text)
    
    # Reconstruct sentences by combining text with punctuation
    reconstructed = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            sentence = sentences[i] + sentences[i+1]
        else:
            sentence = sentences[i]
        reconstructed.append(sentence.strip())
    
    # Handle the last part if it doesn't end with punctuation
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        reconstructed.append(sentences[-1].strip())
    
    # Restore abbreviations
    for placeholder, original in temp_replacements.items():
        reconstructed = [sent.replace(placeholder, original) for sent in reconstructed]
    
    # Filter out empty sentences and very short ones (likely artifacts)
    sentences = [sent for sent in reconstructed if len(sent.strip()) > 3]
    
    return sentences

def process_csv_to_sentences(input_file: str, output_file: str):
    """
    Process a CSV file to chunk chapter_text into sentences.
    """
    try:
        # Read the CSV
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
        
        # Verify required columns exist
        required_cols = ['book_title', 'chapter_number', 'chapter_title', 'chapter_text', 'hs_thematic_explanation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
        
        # Create new dataframe for sentence-level data
        sentence_rows = []
        
        for idx, row in df.iterrows():
            chapter_text = row['chapter_text']
            sentences = split_into_sentences(chapter_text)
            
            print(f"Processing Chapter {row['chapter_number']}: '{row['chapter_title'][:50]}...' -> {len(sentences)} sentences")
            
            # Create a row for each sentence
            for sent_idx, sentence in enumerate(sentences, 1):
                sentence_row = {
                    'book_title': row['book_title'],
                    'chapter_number': row['chapter_number'],
                    'chapter_title': row['chapter_title'],
                    'sentence_number': sent_idx,
                    'sentence_text': sentence
                }
                sentence_rows.append(sentence_row)
        
        # Create new dataframe
        sentences_df = pd.DataFrame(sentence_rows)
        
        # Save to new CSV
        sentences_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nSaved {len(sentences_df)} sentences to {output_file}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Original chapters: {len(df)}")
        print(f"Total sentences: {len(sentences_df)}")
        print(f"Average sentences per chapter: {len(sentences_df)/len(df):.1f}")
        
        return sentences_df
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
        return None
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return None

def main():
    """
    Main function to process both CSV files.
    """
    csv_files = [
        'ELFILI_chapters_with_thematic.csv',
        'NOLI_chapters_with_thematic.csv'
    ]
    
    for csv_file in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing {csv_file}")
        print('='*50)
        
        # Create output filename
        output_file = csv_file.replace('.csv', '_sentences.csv')
        
        # Process the file
        result_df = process_csv_to_sentences(csv_file, output_file)
        
        if result_df is not None:
            # Show sample of the output
            print(f"\nSample of output data:")
            print(result_df[['book_title', 'chapter_number', 'sentence_number', 'sentence_text']].head(3).to_string(index=False))

if __name__ == "__main__":
    main()