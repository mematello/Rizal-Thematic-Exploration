import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

# Append backend root to path so we can import app modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import engine, Theme
from app.core.config import get_settings
import pickle

def index_themes():
    settings = get_settings()
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    
    model = SentenceTransformer(settings.BERT_MODEL_NAME)
    
    csv_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'csvFiles')
    
    books = [
        ('noli', 'noli_themes.csv'),
        ('elfili', 'elfili_themes.csv')
    ]
    
    themes_data = {}
    
    for book_key, theme_filename in books:
        theme_path = os.path.join(csv_dir, theme_filename)
        if not os.path.exists(theme_path):
            print(f"File not found: {theme_path}")
            continue
            
        print(f"Processing {book_key} themes from {theme_filename}...")
        try:
            # For ~552 theme CSV rows
            themes_df = pd.read_csv(theme_path)
            themes_df.columns = themes_df.columns.str.strip()
        except pd.errors.EmptyDataError:
            print(f"CSV empty or invalid: {theme_path}")
            continue
            
        for _, row in themes_df.iterrows():
            theme_label = str(row.get('Theme', row.get('Tagalog Title', ''))).strip()
            example_text = str(row.get('Example', row.get('Meaning', ''))).strip()
            
            if not theme_label or not example_text or str(theme_label) == 'nan' or str(example_text) == 'nan':
                continue
                
            if theme_label not in themes_data:
                themes_data[theme_label] = []
            themes_data[theme_label].append(example_text)
            
    # Embed themes
    theme_bank = {}
    total_examples = 0
    
    for theme, examples in themes_data.items():
        print(f"Computing embeddings for theme '{theme}' ({len(examples)} examples)...")
        embeddings = model.encode(examples, show_progress_bar=False)
        theme_bank[theme] = embeddings
        total_examples += len(examples)
        
    print(f"Total themes: {len(theme_bank)}, Total examples embedded: {total_examples}")
    
    # Save the theme bank to a pickle file
    backend_data_dir = os.path.join(os.path.dirname(__file__), '..', 'app', 'data')
    if not os.path.exists(backend_data_dir):
        os.makedirs(backend_data_dir)
        
    pickle_path = os.path.join(backend_data_dir, 'theme_bank.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(theme_bank, f)
        
    print(f"Saved theme bank to {pickle_path}")
    session.close()

if __name__ == "__main__":
    index_themes()
