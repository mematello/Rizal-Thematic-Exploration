# Noli Me Tangere Semantic Search Pipeline
# High-accuracy multilingual semantic search for Tagalog literature

# Install required packages
!pip install -q sentence-transformers torch transformers scikit-learn pandas numpy

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CONFIGURATION PARAMETERS
# ========================================

# Configurable similarity thresholds
THEME_THRESHOLD = 0.50      # Minimum similarity to consider keyword related to theme
CHAPTER_THRESHOLD = 0.35    # Minimum similarity to return chapter results

# Model configuration
# Using multilingual-E5-large for strong Tagalog/English performance
MODEL_NAME = "intfloat/multilingual-e5-large"  

print("🔧 Configuration:")
print(f"   Theme threshold: {THEME_THRESHOLD} (adjust between 0.45-0.60)")
print(f"   Chapter threshold: {CHAPTER_THRESHOLD} (adjust between 0.30-0.40)")
print(f"   Model: {MODEL_NAME}")
print()

# ========================================
# DATA LOADING AND VALIDATION
# ========================================

def load_and_validate_data():
    """Load CSV files with comprehensive error handling."""
    try:
        # Load themes CSV
        if not os.path.exists('noli_themes.txt'):
            raise FileNotFoundError("themes CSV file 'noli_themes.txt' not found")
        
        themes_df = pd.read_csv('noli_themes.txt')
        required_theme_cols = ['English Title', 'Tagalog Title', 'Meaning']
        
        if not all(col in themes_df.columns for col in required_theme_cols):
            raise ValueError(f"themes CSV missing required columns: {required_theme_cols}")
        
        # Load chapters CSV  
        if not os.path.exists('noli_chapters.txt'):
            raise FileNotFoundError("chapters CSV file 'noli_chapters.txt' not found")
            
        chapters_df = pd.read_csv('noli_chapters.txt')
        required_chapter_cols = ['book_title', 'chapter_number', 'chapter_title', 
                                'sentence_number', 'sentence_text']
        
        if not all(col in chapters_df.columns for col in required_chapter_cols):
            raise ValueError(f"chapters CSV missing required columns: {required_chapter_cols}")
        
        # Clean and validate data
        themes_df = themes_df.dropna(subset=['Meaning']).reset_index(drop=True)
        chapters_df = chapters_df.dropna(subset=['sentence_text']).reset_index(drop=True)
        
        # Normalize text (minimal cleaning while preserving original content)
        themes_df['meaning_clean'] = themes_df['Meaning'].str.strip()
        chapters_df['sentence_clean'] = chapters_df['sentence_text'].str.strip()
        
        print(f"✅ Data loaded successfully:")
        print(f"   Themes: {len(themes_df)} entries")
        print(f"   Chapters: {len(chapters_df)} sentences")
        print()
        
        return themes_df, chapters_df
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

# ========================================
# MODEL INITIALIZATION
# ========================================

def initialize_model():
    """Initialize the multilingual sentence transformer model."""
    try:
        print("🤖 Loading multilingual model (this may take a few minutes)...")
        model = SentenceTransformer(MODEL_NAME)
        
        # Test model with sample text
        test_embedding = model.encode("Test sentence", convert_to_tensor=True)
        print(f"   Model loaded successfully (embedding dim: {test_embedding.shape[0]})")
        print(f"   Chosen for strong Tagalog/English multilingual performance")
        print()
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("   Falling back to smaller model...")
        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("   Fallback model loaded successfully")
            return model
        except:
            raise Exception("Failed to load any suitable model")

# ========================================
# EMBEDDING GENERATION AND CACHING
# ========================================

def generate_embeddings(model, themes_df, chapters_df):
    """Generate and cache embeddings for themes and chapters."""
    print("🧠 Generating embeddings...")
    
    try:
        # Generate theme embeddings (from Meaning column)
        theme_texts = themes_df['meaning_clean'].tolist()
        print(f"   Processing {len(theme_texts)} theme meanings...")
        theme_embeddings = model.encode(theme_texts, convert_to_tensor=True)
        
        # Generate chapter sentence embeddings
        chapter_texts = chapters_df['sentence_clean'].tolist()
        print(f"   Processing {len(chapter_texts)} chapter sentences...")
        chapter_embeddings = model.encode(chapter_texts, convert_to_tensor=True)
        
        print("   ✅ All embeddings generated successfully")
        print()
        
        return theme_embeddings, chapter_embeddings
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {str(e)}")
        raise

# ========================================
# SEARCH PIPELINE
# ========================================

def semantic_search(query, model, themes_df, chapters_df, theme_embeddings, chapter_embeddings):
    """
    Main semantic search function implementing conditional logic.
    
    Args:
        query: User search keyword/phrase
        model: Sentence transformer model
        themes_df: Themes dataframe
        chapters_df: Chapters dataframe
        theme_embeddings: Pre-computed theme embeddings
        chapter_embeddings: Pre-computed chapter embeddings
    
    Returns:
        List of result tuples (tagalog_title, chapter_title, sentence_number, sentence_text)
    """
    
    # Input validation and normalization
    if not query or not query.strip():
        print("❌ Please provide a valid search keyword")
        return []
    
    query = query.strip().lower()
    print(f"🔍 Searching for: '{query}'")
    
    try:
        # Generate query embedding
        query_embedding = model.encode([query], convert_to_tensor=True)
        
        # Step 1: Calculate theme similarities
        theme_similarities = cosine_similarity(
            query_embedding.cpu().numpy(), 
            theme_embeddings.cpu().numpy()
        )[0]
        
        best_theme_idx = np.argmax(theme_similarities)
        best_theme_score = theme_similarities[best_theme_idx]
        best_theme_tagalog = themes_df.iloc[best_theme_idx]['Tagalog Title']
        
        print(f"   Best theme match: '{best_theme_tagalog}' (similarity: {best_theme_score:.3f})")
        
        # Step 2: Calculate chapter similarities
        chapter_similarities = cosine_similarity(
            query_embedding.cpu().numpy(),
            chapter_embeddings.cpu().numpy()
        )[0]
        
        # Get top chapter matches
        top_chapter_indices = np.argsort(chapter_similarities)[::-1][:3]
        top_chapter_scores = chapter_similarities[top_chapter_indices]
        
        # Check if top chapter result meets minimum threshold
        if len(top_chapter_scores) == 0 or top_chapter_scores[0] < CHAPTER_THRESHOLD:
            print(f"   No relevant results found (top similarity: {top_chapter_scores[0] if len(top_chapter_scores) > 0 else 0:.3f} < {CHAPTER_THRESHOLD})")
            return []
        
        # Step 3: Apply conditional logic
        results = []
        
        if best_theme_score >= THEME_THRESHOLD:
            # Keyword is related to a theme - include Tagalog Title
            print(f"   ✅ Theme-related search (using theme: '{best_theme_tagalog}')")
            
            for i, (idx, score) in enumerate(zip(top_chapter_indices, top_chapter_scores)):
                if score >= CHAPTER_THRESHOLD:
                    row = chapters_df.iloc[idx]
                    results.append((
                        best_theme_tagalog,
                        row['chapter_title'], 
                        row['sentence_number'],
                        row['sentence_text']
                    ))
                    print(f"      Result {i+1}: Chapter similarity {score:.3f}")
        else:
            # Keyword not related to any theme - chapter-only search
            print(f"   📖 Chapter-only search (theme similarity {best_theme_score:.3f} < {THEME_THRESHOLD})")
            
            for i, (idx, score) in enumerate(zip(top_chapter_indices, top_chapter_scores)):
                if score >= CHAPTER_THRESHOLD:
                    row = chapters_df.iloc[idx]
                    results.append((
                        "",  # Empty Tagalog Title for non-theme searches
                        row['chapter_title'],
                        row['sentence_number'], 
                        row['sentence_text']
                    ))
                    print(f"      Result {i+1}: Chapter similarity {score:.3f}")
        
        print(f"   📋 Returning {len(results)} results")
        return results
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []

# ========================================
# OUTPUT FORMATTING
# ========================================

def format_results(results):
    """Format search results for display."""
    if not results:
        print("\n📝 No relevant results found.")
        return
    
    print(f"\n📋 Search Results ({len(results)} found):")
    print("=" * 80)
    
    for i, (tagalog_title, chapter_title, sentence_number, sentence_text) in enumerate(results, 1):
        # Format with or without Tagalog Title based on search type
        if tagalog_title:
            print(f"{i}. {tagalog_title} | {chapter_title} | Sentence {sentence_number}")
        else:
            print(f"{i}. {chapter_title} | Sentence {sentence_number}")
        
        # Word wrap long sentences
        wrapped_text = sentence_text
        if len(wrapped_text) > 100:
            wrapped_text = wrapped_text[:97] + "..."
        
        print(f"   └─ {wrapped_text}")
        print()

# ========================================
# MAIN EXECUTION
# ========================================

def main():
    """Main execution function."""
    print("🚀 Noli Me Tangere Semantic Search Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Load data
        themes_df, chapters_df = load_and_validate_data()
        
        # Initialize model
        model = initialize_model()
        
        # Generate embeddings (cached in memory)
        theme_embeddings, chapter_embeddings = generate_embeddings(model, themes_df, chapters_df)
        
        print("🎯 System ready! You can now search for keywords.")
        print()
        
        # Interactive search loop
        while True:
            try:
                keyword = input("Enter search keyword (or 'exit'/'quit' to stop): ").strip()
                
                if keyword.lower() in ['exit', 'quit', '']:
                    print("👋 Thank you for using the semantic search system!")
                    break
                
                # Perform search
                results = semantic_search(
                    keyword, model, themes_df, chapters_df, 
                    theme_embeddings, chapter_embeddings
                )
                
                # Format and display results
                format_results(results)
                print("-" * 80)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Search interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Search error: {str(e)}")
                print("Please try again with a different keyword.")
                print()
    
    except Exception as e:
        print(f"💥 Fatal error: {str(e)}")
        print("Please check your CSV files and try again.")

# ========================================
# DEMO SEARCHES (for testing)
# ========================================

def run_demo_searches():
    """Run demonstration searches with various keywords."""
    demo_keywords = [
        "pag-ibig",           # Love (Tagalog) - should match theme
        "edukasyon",          # Education (Tagalog) - should match theme  
        "Ibarra",             # Character name - should be chapter-only
        "simbahan",           # Church (Tagalog) - could match religion theme
        "kalayaan",           # Freedom (Tagalog) - should match theme
        "random_word_xyz"     # Non-existent - should return no results
    ]
    
    print("🎪 Running demonstration searches...")
    print("=" * 60)
    
    for keyword in demo_keywords:
        print(f"\nDemo search: '{keyword}'")
        print("-" * 30)
        
        results = semantic_search(
            keyword, model, themes_df, chapters_df,
            theme_embeddings, chapter_embeddings
        )
        
        if results:
            for i, result in enumerate(results[:2], 1):  # Show first 2 results
                tagalog_title, chapter_title, sentence_number, sentence_text = result
                if tagalog_title:
                    print(f"  {i}. {tagalog_title} | {chapter_title} | #{sentence_number}")
                else:
                    print(f"  {i}. {chapter_title} | #{sentence_number}")
                print(f"     └─ {sentence_text[:80]}...")
        else:
            print("  No results found.")
        print()

# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    # Load data and initialize system
    themes_df, chapters_df = load_and_validate_data()
    model = initialize_model()
    theme_embeddings, chapter_embeddings = generate_embeddings(model, themes_df, chapters_df)
    
    # Uncomment the next line to run demo searches first
    # run_demo_searches()
    
    # Start interactive search
    main()