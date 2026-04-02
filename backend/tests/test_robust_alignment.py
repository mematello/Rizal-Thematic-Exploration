import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add backend to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.robust_aligner import RobustAligner
from sentence_transformers import SentenceTransformer

def test_robust_alignment():
    print("Testing Robust Aligner...")
    
    # Load a small sample of sentences
    # For testing, we'll just use the first 20 sentences of Noli Chapter 1
    csv_path = Path(__file__).resolve().parent.parent.parent / "csvFiles" / "noli_chapters.csv"
    if not csv_path.exists():
        print(f"CSV not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    ch1_full = df[(df['book_title'] == 'Noli Me Tangere') & (df['chapter_number'] == 1)]['sentence_text'].tolist()
    
    if not ch1_full:
        print("No sentences found for Chapter 1")
        return
        
    # Take first 50 sentences as full text
    full_text = ch1_full[:50]
    
    # Create a simulated "buod" (summary)
    # 1. Paraphrase of first few sentences
    # 2. Mention of a character (Tiago)
    # 3. Mention of another character (Isabel)
    buod = [
        "May malaking handaan sa bahay ni Kapitan Tiago sa Kalye Anluwage.",
        "Si Tiya Isabel ang tumanggap sa mga panauhin.",
        "Nagkaroon ng mainit na sagutan sa pagitan ni Padre Damaso at Tinyente Guevarra."
    ]
    
    print(f"Full text sentences: {len(full_text)}")
    print(f"Buod sentences: {len(buod)}")
    
    # Characters list
    tauhan = ["Kapitan Tiago", "Tiya Isabel", "Padre Damaso", "Tinyente Guevarra", "Don Santiago de los Santos"]
    
    # Initialize model (using a small one for speed if possible, or the one from settings)
    # For the sake of this test, we'll use a small model
    print("Loading model...")
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Encoding sentences...")
    buod_embs = model.encode(buod)
    full_embs = model.encode(full_text)
    
    # Initialize Aligner
    aligner = RobustAligner(tauhan_list=tauhan)
    
    # Run Alignment
    print("Running alignment...")
    results = aligner.align(
        buod_sentences=buod,
        full_sentences=full_text,
        buod_embeddings=buod_embs,
        full_embeddings=full_embs
    )
    
    print("\nAlignment Results:")
    for res in results:
        print(f"Buod [{res.buod_index}]: {res.buod_text}")
        print(f"  Mapped to Window: {res.best_window_start} - {res.best_window_end}")
        print(f"  Center Sentence: {res.best_center_sentence} ({full_text[res.best_center_sentence][:50]}...)")
        print(f"  Scores: Lex={res.lexical_score:.2f}, Sem={res.semantic_score:.2f}, Pos={res.position_score:.2f}, Tau={res.tauhan_score:.2f}, Final={res.final_score:.2f}")
        print("-" * 20)

    # Simple validations
    assert len(results) == len(buod)
    assert results[0].best_window_start <= results[1].best_window_start
    assert results[1].best_window_start <= results[2].best_window_start
    print("\nTest passed: Monotonicity and coverage confirmed.")

if __name__ == "__main__":
    test_robust_alignment()
