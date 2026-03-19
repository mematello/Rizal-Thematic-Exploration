import sys
import os
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

# Append backend root to path so we can import app modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import engine
from app.core.config import get_settings
import pickle

def index_themes():
    settings = get_settings()
    
    model = SentenceTransformer(settings.BERT_MODEL_NAME)
    
    csv_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'csvFiles')
    
    books = [
        ('noli', 'noli_themes.csv'),
        ('elfili', 'elfili_themes.csv')
    ]
    
    theme_bank = {'noli': [], 'elfili': []}
    total_examples = 0
    
    for book_key, theme_filename in books:
        theme_path = os.path.join(csv_dir, theme_filename)
        if not os.path.exists(theme_path):
            print(f"File not found: {theme_path}")
            continue
            
        print(f"Processing {book_key} themes from {theme_filename}...")
        try:
            themes_df = pd.read_csv(theme_path)
            themes_df.columns = themes_df.columns.str.strip()
        except pd.errors.EmptyDataError:
            print(f"CSV empty or invalid: {theme_path}")
            continue
            
        meanings = []
        labels = []
        
        for _, row in themes_df.iterrows():
            theme_label = str(row.get('Theme', row.get('Tagalog Title', ''))).strip()
            example_text = str(row.get('Example', row.get('Meaning', ''))).strip()
            
            if not theme_label or not example_text or str(theme_label) == 'nan' or str(example_text) == 'nan':
                continue
                
            meanings.append(example_text)
            labels.append(theme_label)
            
        if meanings:
            print(f"Computing embeddings for {len(meanings)} rows in {book_key}...")
            embeddings = model.encode(meanings, show_progress_bar=False)
            
            for i in range(len(meanings)):
                theme_bank[book_key].append({
                    "label": labels[i],
                    "evidence": meanings[i],
                    "book": book_key,
                    "embedding": embeddings[i]
                })
            total_examples += len(meanings)
        
    print(f"Total structured examples embedded: {total_examples}")
    
    # Save the theme bank to a pickle file
    backend_data_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'data')
    if not os.path.exists(backend_data_dir):
        os.makedirs(backend_data_dir)
        
    pickle_path = os.path.join(backend_data_dir, 'theme_bank.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(theme_bank, f)
        
    print(f"Saved theme bank to {pickle_path}")

if __name__ == "__main__":
    index_themes()
