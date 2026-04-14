"""
train_buod_model.py
Trains rizal-xlm-r-buod using TSDAE on the SUMMARY (buod) corpus only.
Corpus: noli_chapters.csv + elfili_chapters.csv
Output: backend/model_output/rizal-xlm-r-buod/
"""
import os
import sys
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import datasets, losses
from torch.utils.data import DataLoader

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

import nltk
for pkg in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{pkg}')
    except LookupError:
        nltk.download(pkg)


def train_buod_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
    root_dir = os.path.dirname(base_dir)                                     # project root
    csv_dir  = os.path.join(root_dir, 'csvFiles')
    output_path = os.path.join(base_dir, 'model_output', 'rizal-xlm-r-buod')
    os.makedirs(output_path, exist_ok=True)

    # Only summary / buod files
    buod_files = ['noli_chapters.csv', 'elfili_chapters.csv']

    train_sentences = []
    logging.info(f"Loading BUOD corpus from {csv_dir}...")

    for filename in buod_files:
        file_path = os.path.join(csv_dir, filename)
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            if 'sentence_text' not in df.columns:
                logging.warning(f"No 'sentence_text' column in {filename}")
                continue
            texts = df['sentence_text'].dropna().astype(str).tolist()
            texts = [t.strip() for t in texts if len(t.strip()) > 10]
            logging.info(f"  Loaded {len(texts)} sentences from {filename}")
            train_sentences.extend(texts)
        except Exception as e:
            logging.error(f"Error reading {filename}: {e}")

    logging.info(f"Total before dedup: {len(train_sentences)}")
    train_sentences = list(set(train_sentences))
    logging.info(f"Total unique BUOD sentences: {len(train_sentences)}")

    if not train_sentences:
        logging.error("No training data. Aborting.")
        return

    base_model_name = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
    logging.info(f"Loading base model: {base_model_name}")
    model = SentenceTransformer(base_model_name)

    logging.info("Preparing TSDAE dataset...")
    train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

    logging.info("Defining DenoisingAutoEncoderLoss...")
    train_loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    logging.info("Starting BUOD model training (3 epochs)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,
        output_path=output_path,
        save_best_model=True
    )

    logging.info(f"SUCCESS: BUOD model saved to {output_path}")
    logging.info("Next: copy/rename to backend/app/models/rizal-xlm-r-buod")
    logging.info("Then run: poetry run python scripts/seed_buod_embeddings.py")


if __name__ == "__main__":
    train_buod_model()
