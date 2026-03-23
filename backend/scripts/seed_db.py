import sys
import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer

# Append backend root to path so we can import app modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.models.database import Sentence, Base, Theme
from app.core.config import get_settings

def seed_db():
    settings = get_settings()
    engine = create_engine(settings.DATABASE_URL)
    
    # Create tables and extension if they don't exist
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    # Load Model with DAPT preference (match engine.py logic)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dapt_path = os.path.join(base_path, 'models', 'rizal-xlm-r-dapt')
    
    if os.path.exists(dapt_path):
        print(f"Using DAPT model for seeding from {dapt_path}")
        model = SentenceTransformer(dapt_path)
    else:
        print(f"DAPT model not found at {dapt_path}. Using base model as fallback.")
        model = SentenceTransformer(settings.BERT_MODEL_NAME)

    # Clear existing data
    print("Clearing existing sentence and theme data...")
    session.query(Sentence).delete()
    session.query(Theme).delete()
    session.commit()

    # Data paths (assuming script is run from backend/)
    # Adjusted to point to root project csvFiles/
    csv_dir = os.path.join(os.path.dirname(__file__), '../../csvFiles')
    
    books = [
        ('noli', 'noli_chapters.csv', 'noli_themes.csv'),
        ('elfili', 'elfili_chapters.csv', 'elfili_themes.csv')
    ]

    for book_key, filename, theme_filename in books:
        file_path = os.path.join(csv_dir, filename)
        theme_path = os.path.join(csv_dir, theme_filename)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        # --- Process Sentences ---
        print(f"Processing {book_key} sentences from {filename}...")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        
        entries = []
        texts = []
        
        for idx, row in df.iterrows():
            text_content = str(row['sentence_text']).strip()
            if not text_content:
                continue
            
            combined_text = f"{row['chapter_title']} {text_content}"
            texts.append(combined_text)
            
            entries.append({
                'book': book_key,
                'chapter_number': int(row['chapter_number']),
                'chapter_title': str(row['chapter_title']),
                'sentence_index': idx,
                'sentence_text': text_content
            })
            
        print(f"Computing embeddings for {len(texts)} sentences...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        db_objects = []
        for i, entry in enumerate(entries):
            embedding_list = embeddings[i].tolist()
            db_obj = Sentence(
                book=entry['book'],
                chapter_number=entry['chapter_number'],
                chapter_title=entry['chapter_title'],
                sentence_index=entry['sentence_index'],
                sentence_text=entry['sentence_text'],
                embedding=embedding_list
            )
            db_objects.append(db_obj)
        
        print(f"Inserting {len(db_objects)} sentences...")
        session.bulk_save_objects(db_objects)
        
        # --- Process Themes ---
        if os.path.exists(theme_path):
            from app.models.database import Theme # Import here to ensure it's available
            print(f"Processing {book_key} themes from {theme_filename}...")
            themes_df = pd.read_csv(theme_path)
            themes_df.columns = themes_df.columns.str.strip()
            
            theme_entries = []
            theme_texts = []
            
            for _, row in themes_df.iterrows():
                tagalog = str(row.get('Tagalog Title', '')).strip()
                meaning = str(row.get('Meaning', '')).strip()
                if not tagalog or not meaning:
                    continue
                    
                theme_texts.append(meaning)
                theme_entries.append({
                    'book': book_key,
                    'tagalog_title': tagalog,
                    'meaning': meaning
                })
                
            print(f"Computing embeddings for {len(theme_texts)} themes...")
            theme_embeddings = model.encode(theme_texts, show_progress_bar=True)
            
            theme_db_objects = []
            for i, entry in enumerate(theme_entries):
                theme_obj = Theme(
                    book=entry['book'],
                    tagalog_title=entry['tagalog_title'],
                    meaning=entry['meaning'],
                    embedding=theme_embeddings[i].tolist()
                )
                theme_db_objects.append(theme_obj)
                
            print(f"Inserting {len(theme_db_objects)} themes...")
            session.bulk_save_objects(theme_db_objects)

        session.commit()
        print(f"Finished {book_key}.")

    session.close()

if __name__ == "__main__":
    seed_db()
