import sys
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import logging
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load .env explicitly
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

from app.models.database import Sentence, Base, engine
from app.core.tagalog_parser import TagalogRoleParser
from app.core.config import get_settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SessionLocal = sessionmaker(bind=engine)

def seed_full_db():
    session = SessionLocal()
    parser = TagalogRoleParser()
    settings = get_settings()
    
    # Model Selection - Skipping DAPT for now as it's too slow on CPU
    model_name = settings.BERT_MODEL_NAME
    logger.info(f"Using model {model_name} for seeding...")
    model = SentenceTransformer(model_name)

    csv_files = {
        'fullversion_noli.csv': 'Noli Me Tangere',
        'fullversion_elfili.csv': 'El Filibusterismo'
    }
    
    csv_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'csvFiles')

    for csv_file, book_title in csv_files.items():
        file_path = os.path.join(csv_dir, csv_file)
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
            
        logger.info(f"Processing {book_title} from {csv_file}...")
        df = pd.read_csv(file_path)
        
        # Ensure we don't duplicate
        session.query(Sentence).filter(Sentence.book == book_title, Sentence.source_type == 'full').delete()
        session.commit()
        
        batch_size = 50 # Reduced batch size for stability
        sentences_to_add = []
        
        for index, row in df.iterrows():
            text = row['sentence_text']
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Parse roles
            structured_info = parser.structured_string(text)
            
            # Combine for embedding
            combined_text = f"{text} [SEP] {structured_info}"
            
            # Embed
            embedding = model.encode(combined_text, show_progress_bar=False).tolist()
            
            sentence_obj = Sentence(
                book=book_title,
                chapter_number=row['chapter_number'],
                chapter_title=row['chapter_title'],
                sentence_index=row['sentence_number'],
                sentence_text=text,
                source_type='full',
                embedding=embedding
            )
            sentences_to_add.append(sentence_obj)
            
            if len(sentences_to_add) >= batch_size:
                session.bulk_save_objects(sentences_to_add)
                session.commit()
                sentences_to_add = []
                if (index + 1) % 500 == 0:
                    logger.info(f"  Processed {index + 1} sentences...")
                
        if sentences_to_add:
            session.bulk_save_objects(sentences_to_add)
            session.commit()
            
        logger.info(f"Finished seeding {book_title}")

    session.close()

if __name__ == "__main__":
    seed_full_db()
