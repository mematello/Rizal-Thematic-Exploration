import os
import pandas as pd

def count():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/
    root_dir = os.path.dirname(base_dir) # Rizal-Thematic-Exploration/
    csv_dir = os.path.join(root_dir, 'csvFiles')
    
    files = [
        'noli_chapters.csv', 'elfili_chapters.csv',
        'fullversion_noli.csv', 'fullversion_elfili.csv'
    ]
    train_sentences = []
    
    log_file = open("sentence_count_log.txt", "w", encoding="utf-8")
    
    for filename in files:
        file_path = os.path.join(csv_dir, filename)
        if not os.path.exists(file_path):
            log_file.write(f"File not found: {file_path}\n")
            continue
            
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            if 'sentence_text' not in df.columns:
                log_file.write(f"Key 'sentence_text' not found in {filename}.\n")
                continue
                
            texts = df['sentence_text'].dropna().astype(str).tolist()
            texts = [t.strip() for t in texts if len(t.strip()) > 10]
            train_sentences.extend(texts)
            log_file.write(f"Loaded {len(texts)} from {filename}\n")
        except Exception as e:
            log_file.write(f"Error: {e}\n")

    log_file.write(f"Total sentences before deduplication: {len(train_sentences)}\n")
    train_sentences = list(set(train_sentences))
    log_file.write(f"Total unique training sentences: {len(train_sentences)}\n")
    log_file.close()

if __name__ == '__main__':
    count()
