import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    TrainingArguments, Trainer,
    pipeline
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
import pickle
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class RizalThematicDataset(Dataset):
    """Custom dataset for chapter text to thematic explanation pairs"""
    
    def __init__(self, texts: List[str], explanations: List[str], tokenizer, max_length=512):
        self.texts = texts
        self.explanations = explanations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        explanation = self.explanations[idx]
        
        # Tokenize input text
        input_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target explanation
        target_encoding = self.tokenizer(
            explanation,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

class RizalThematicGenerator:
    """Main class for the Rizal novels thematic explanation generator"""
    
    def __init__(self, model_name='t5-small'):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # For keyword search
        self.vectorizer = None
        self.chapter_vectors = None
        self.chapters_data = []
        
        # Translation pipeline for multilingual support
        try:
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-tl-en")
        except:
            print("Translation model not available, using text as-is")
            self.translator = None
    
    def load_and_preprocess_data(self, csv_files: List[str]) -> Tuple[List[str], List[str]]:
        """Load and preprocess CSV files"""
        all_texts = []
        all_explanations = []
        
        for csv_file in csv_files:
            print(f"Loading {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # Clean column names (remove whitespace)
            df.columns = df.columns.str.strip()
            
            # Combine chapter title and text for richer input
            for _, row in df.iterrows():
                if pd.notna(row['chapter_text']) and pd.notna(row['hs_thematic_explanation']):
                    # Create input text combining title and content
                    input_text = f"Chapter: {row['chapter_title']} Content: {row['chapter_text'][:1000]}"  # Limit length
                    
                    # Clean text
                    input_text = self.clean_text(input_text)
                    explanation = self.clean_text(row['hs_thematic_explanation'])
                    
                    if len(input_text.strip()) > 50 and len(explanation.strip()) > 20:  # Filter short texts
                        all_texts.append(input_text)
                        all_explanations.append(explanation)
                        
                        # Store for keyword search
                        self.chapters_data.append({
                            'book_title': row['book_title'],
                            'chapter_number': row['chapter_number'],
                            'chapter_title': row['chapter_title'],
                            'chapter_text': row['chapter_text'],
                            'hs_thematic_explanation': row['hs_thematic_explanation'],
                            'search_text': f"{row['chapter_title']} {row['chapter_text']}"
                        })
        
        print(f"Loaded {len(all_texts)} training examples")
        return all_texts, all_explanations
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Filipino characters
        text = re.sub(r'[^\w\s\-\.,!?;:\'"()]', ' ', text)
        return text.strip()
    
    def prepare_training_data(self, texts: List[str], explanations: List[str]) -> RizalThematicDataset:
        """Prepare dataset for training"""
        # Add task prefix for T5
        prefixed_texts = [f"explain theme: {text}" for text in texts]
        return RizalThematicDataset(prefixed_texts, explanations, self.tokenizer)
    
    def train_model(self, csv_files: List[str], output_dir='./rizal_model', epochs=3, batch_size=4):
        """Train the model on the dataset"""
        print("Loading and preprocessing data...")
        texts, explanations = self.load_and_preprocess_data(csv_files)
        
        # Prepare dataset
        dataset = self.prepare_training_data(texts, explanations)
        
        # Split data (80% train, 20% eval)
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save search data
        self.build_search_index()
        with open(os.path.join(output_dir, 'search_data.pkl'), 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'chapter_vectors': self.chapter_vectors,
                'chapters_data': self.chapters_data
            }, f)
        
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir='./rizal_model'):
        """Load trained model and search data"""
        print(f"Loading model from {model_dir}")
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
        
        # Load search data
        search_data_path = os.path.join(model_dir, 'search_data.pkl')
        if os.path.exists(search_data_path):
            with open(search_data_path, 'rb') as f:
                search_data = pickle.load(f)
                self.vectorizer = search_data['vectorizer']
                self.chapter_vectors = search_data['chapter_vectors']
                self.chapters_data = search_data['chapters_data']
            print("Search index loaded successfully")
        else:
            print("Search data not found. Run training first or call build_search_index()")
    
    def build_search_index(self):
        """Build TF-IDF search index for chapters"""
        print("Building search index...")
        if not self.chapters_data:
            print("No chapter data available. Load data first.")
            return
        
        # Extract search texts
        search_texts = [chapter['search_text'] for chapter in self.chapters_data]
        
        # Build TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # Keep Filipino words
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.chapter_vectors = self.vectorizer.fit_transform(search_texts)
        print(f"Search index built with {len(search_texts)} chapters")
    
    def translate_to_english(self, text: str) -> str:
        """Translate Tagalog text to English for better search"""
        if self.translator is None:
            return text
        
        try:
            # Simple heuristic: if text contains Filipino words, attempt translation
            filipino_indicators = ['ang', 'ng', 'sa', 'mga', 'kay', 'ni', 'si', 'mga', 'at']
            if any(word in text.lower() for word in filipino_indicators):
                result = self.translator(text)
                return result[0]['translation_text'] if result else text
        except:
            pass
        
        return text
    
    def search_chapters(self, query: str, top_k=3) -> List[Dict]:
        """Search for relevant chapters based on keyword query"""
        if self.vectorizer is None or self.chapter_vectors is None:
            print("Search index not built. Call build_search_index() first.")
            return []
        
        # Translate query if needed
        english_query = self.translate_to_english(query)
        combined_query = f"{query} {english_query}"
        
        # Vectorize query
        query_vector = self.vectorizer.transform([combined_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.chapter_vectors).flatten()
        
        # Get top results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                chapter = self.chapters_data[idx].copy()
                chapter['similarity'] = similarities[idx]
                results.append(chapter)
        
        return results
    
    def generate_thematic_explanation(self, chapter_text: str, max_length=200) -> str:
        """Generate thematic explanation for given chapter text"""
        # Prepare input
        input_text = f"explain theme: Chapter content: {chapter_text[:800]}"  # Limit input length
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=4,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation
    
    def process_user_query(self, query: str) -> Dict:
        """Main function to process user keyword query and return thematic explanation"""
        print(f"Processing query: '{query}'")
        
        # Search for relevant chapters
        relevant_chapters = self.search_chapters(query, top_k=2)
        
        if not relevant_chapters:
            return {
                'query': query,
                'found_chapters': [],
                'generated_explanation': "No relevant chapters found for the given keywords.",
                'success': False
            }
        
        # Combine text from top chapters
        combined_text = ""
        chapter_info = []
        
        for chapter in relevant_chapters:
            combined_text += f"{chapter['chapter_title']}: {chapter['chapter_text'][:500]} "
            chapter_info.append({
                'book': chapter['book_title'],
                'chapter': chapter['chapter_number'],
                'title': chapter['chapter_title'],
                'similarity': round(chapter['similarity'], 3)
            })
        
        # Generate explanation
        generated_explanation = self.generate_thematic_explanation(combined_text)
        
        return {
            'query': query,
            'found_chapters': chapter_info,
            'generated_explanation': generated_explanation,
            'success': True
        }

# Example usage and main execution
def main():
    # Initialize generator
    generator = RizalThematicGenerator(model_name='t5-small')
    
    # CSV file paths (update these to your actual file paths)
    csv_files = [
        'NOLI_chapters_with_thematic.csv',  # Replace with actual file name
        'ELFILI_chapters_with_thematic.csv'  # Replace with actual file name
    ]
    
    # Check if model exists, if not train it
    model_dir = './rizal_model'
    if not os.path.exists(model_dir):
        print("Training new model...")
        generator.train_model(csv_files, output_dir=model_dir, epochs=3, batch_size=2)
    else:
        print("Loading existing model...")
        generator.load_model(model_dir)
    
    # Interactive query processing
    print("\n=== Rizal Novels Thematic Explanation Generator ===")
    print("Enter keywords in English or Tagalog to get thematic explanations")
    print("Examples: 'education kolonyalismo', 'social class inequality', 'Jose Rizal nationalism'")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Enter your keywords: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # Process query
        result = generator.process_user_query(user_input)
        
        print(f"\n--- Results for: '{result['query']}' ---")
        
        if result['success']:
            print("Found chapters:")
            for chapter in result['found_chapters']:
                print(f"  - {chapter['book']}, Chapter {chapter['chapter']}: {chapter['title']} (similarity: {chapter['similarity']})")
            
            print(f"\nGenerated Thematic Explanation:")
            print(f"{result['generated_explanation']}")
        else:
            print(result['generated_explanation'])
        
        print("-" * 50)

if __name__ == "__main__":
    main()