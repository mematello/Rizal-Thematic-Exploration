import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

def build_theme_keywords():
    base_dir = Path(__file__).resolve().parent.parent.parent
    backend_data_dir = base_dir / "backend" / "app" / "data"
    
    # Simple tagalog stop words loosely defined, but mostly letting TF-IDF do the heavy lifting
    stop_words = ["ang", "ng", "mga", "sa", "na", "at", "ay", "ito", "ni", "si", "para", "kung", "o", "kay", "ba", "pa", "kanya", "kanila", "sila", "siya", "ako", "kami", "tayo"]
    
    for book in ['noli', 'elfili']:
        csv_path = base_dir / "csvFiles" / f"{book}_themes.csv"
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Group meanings by Tagalog Title
        theme_texts = {}
        for _, row in df.iterrows():
            theme = row['Tagalog Title']
            meaning = str(row['Meaning'])
            if theme not in theme_texts:
                theme_texts[theme] = []
            theme_texts[theme].append(meaning)
            
        theme_names = list(theme_texts.keys())
        corpus = [" ".join(texts) for texts in theme_texts.values()]
        
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=stop_words)
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError as e:
            print(f"Error vectorizing {book}: {e}")
            continue
            
        feature_names = vectorizer.get_feature_names_out()
        
        theme_keywords = {}
        for i, theme in enumerate(theme_names):
            row = tfidf_matrix.getrow(i).toarray()[0]
            # Get top 25 indices
            top_indices = row.argsort()[-25:][::-1]
            keywords = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            theme_keywords[theme] = keywords
            
        output_path = backend_data_dir / f"theme_keywords_{book}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(theme_keywords, f, ensure_ascii=False, indent=2)
            
        print(f"Generated {output_path.name}: keywords for {len(theme_names)} themes.")

if __name__ == "__main__":
    build_theme_keywords()
