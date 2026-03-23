import os
import pandas as pd
import sys
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, datasets, losses
from torch.utils.data import DataLoader
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def train_dapt():
    # 1. Setup Paths
    # Current script is in backend/scripts/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # backend/
    root_dir = os.path.dirname(base_dir) # Rizal-Thematic-Exploration/
    csv_dir = os.path.join(root_dir, 'csvFiles')
    output_path = os.path.join(base_dir, 'model_output', 'rizal-xlm-r')
    
    # Creates output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # 2. Load Corpus
    files = [
        'noli_chapters.csv', 'elfili_chapters.csv',
        'fullversion_noli.csv', 'fullversion_elfili.csv'
    ]
    train_sentences = []
    
    logging.info(f"Loading corpus from {csv_dir}...")
    
    for filename in files:
        file_path = os.path.join(csv_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            # Ensure column names are clean
            df.columns = df.columns.str.strip()
            
            if 'sentence_text' not in df.columns:
                logging.warning(f"Key 'sentence_text' not found in {filename}. Columns: {df.columns}")
                continue
                
            # Collect unique sentences to avoid overexpression of repetitive phrases
            # Although for DAPT, frequency might matter? TSDAE usually works on unique sentences or full corpus.
            # Let's keep them as is but filter empties.
            texts = df['sentence_text'].dropna().astype(str).tolist()
            texts = [t.strip() for t in texts if len(t.strip()) > 10] # Filter very short noise
            
            logging.info(f"Loaded {len(texts)} sentences from {filename}")
            train_sentences.extend(texts)
            
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")

    logging.info(f"Total sentences before deduplication: {len(train_sentences)}")
    
    # Deduplicate sentences
    train_sentences = list(set(train_sentences))
    logging.info(f"Total unique training sentences: {len(train_sentences)}")
    
    if not train_sentences:
        logging.error("No training data found. Aborting.")
        return

    # 3. Initialize Model
    # We use the same base model as before
    model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    logging.info(f"Loading base model: {model_name}")
    model = SentenceTransformer(model_name)

    # 4. Create TSDAE Dataset
    # TSDAE adds noise (deletion) to input sentences and asks model to reconstruct original
    logging.info("Preparing TSDAE dataset...")
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    
    # DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    # 5. Define Loss
    logging.info("Defining DenoisingAutoEncoderLoss...")
    # Removing decoder_name_or_path to let it default (often safer)
    train_loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    # 6. Train
    logging.info("Starting training...")
    # 1-3 epochs is usually enough for DAPT on small-medium corpus
    num_epochs = 3 
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,
        output_path=output_path,
        save_best_model=True
    )
    
    logging.info(f"Training finished. Model saved to {output_path}")
    logging.info("IMPORTANT: You must now update your .env file to point to this path and RE-SEED your database.")

if __name__ == "__main__":
    train_dapt()
