# ===============================================
# Thesis Project: Search Engine with BERT (Tagalog + English)
# Using XLM-R for better multilingual accuracy
# Google Colab Notebook
# ===============================================

# Install dependencies
!pip install transformers sentence-transformers torch pandas scikit-learn ipywidgets

# Import libraries
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import ipywidgets as widgets
from IPython.display import display, HTML

# ===============================================
# Load Data
# ===============================================
themes_path = "/content/drive/MyDrive/THESIS RIZAL - FILES/THEMES OF NOLI AND ELFILI/Elfilithemes.csv"
chapters_path = "/content/drive/MyDrive/THESIS RIZAL - FILES/THEMES OF NOLI AND ELFILI/ELFILI_chapters_with_thematic_sentences.csv"
talasalitaan_path = "/content/drive/MyDrive/THESIS RIZAL - FILES/THEMES OF NOLI AND ELFILI/elfili_talasalitaan_cleaned_updated.csv"

# Load CSVs
themes_df = pd.read_csv(themes_path)
chapters_df = pd.read_csv(chapters_path)
talasalitaan_df = pd.read_csv(talasalitaan_path)

# ===============================================
# Build Talasalitaan Dictionary
# ===============================================
# Dictionary mapping deep_word -> modern_word
talasalitaan_dict = dict(zip(
    talasalitaan_df["deep_word"].str.lower(),
    talasalitaan_df["modern_word"]
))

# ===============================================
# Prepare Search Dataset
# ===============================================
# Combine relevant text fields for search (themes + sentences)
themes_df["combined_text"] = (
    themes_df["English Title"].fillna("") + " | " +
    themes_df["Tagalog Title"].fillna("") + " | " +
    themes_df["Meaning"].fillna("")
)

chapters_df["combined_text"] = (
    "Book: " + chapters_df["book_title"].astype(str) +
    " | Chapter " + chapters_df["chapter_number"].astype(str) + ": " +
    chapters_df["chapter_title"].astype(str) +
    " | Sentence: " + chapters_df["sentence_text"].astype(str)
)

# Merge both datasets for embedding
all_data = pd.concat([
    pd.DataFrame({"source": "Theme", "text": themes_df["combined_text"]}),
    pd.DataFrame({"source": "Chapter", "text": chapters_df["combined_text"]})
], ignore_index=True)

# ===============================================
# Load Multilingual Model (Tagalog + English)
# ===============================================
# Best choice for multilingual semantic search
model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")

# Precompute embeddings for dataset
print("Encoding dataset... this may take a while.")
data_embeddings = model.encode(all_data["text"].tolist(),
                               convert_to_tensor=True,
                               show_progress_bar=True)

# ===============================================
# Search Function with Talasalitaan Expansion
# ===============================================
def expand_query(query):
    """Expand query using talasalitaan if deep_word is detected."""
    words = query.lower().split()
    expanded_words = []
    for w in words:
        if w in talasalitaan_dict:
            expanded_words.append(w)
            expanded_words.append(talasalitaan_dict[w])
        else:
            expanded_words.append(w)
    return " ".join(expanded_words)

def semantic_search(query, top_k=5):
    expanded_query = expand_query(query)
    query_embedding = model.encode(expanded_query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, data_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1]), 1):
        results.append({
            "rank": rank,
            "text": all_data.iloc[idx.item()]["text"],
            "source": all_data.iloc[idx.item()]["source"],
            "score": float(score)
        })
    return results, expanded_query

# ===============================================
# Interactive Search Bar
# ===============================================
search_box = widgets.Text(
    description='',
    placeholder='Type a query...',
    layout=widgets.Layout(width='60%')
)

output = widgets.Output()

def on_search_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        query = change['new']
        output.clear_output()
        if query.strip() == "":
            return
        results, expanded_query = semantic_search(query, top_k=5)
        with output:
            display(HTML(f"<h3>Top results for query: '<b>{query}</b>'</h3>"))
            if expanded_query.lower() != query.lower():
                display(HTML(f"<p><i>Expanded query used:</i> {expanded_query}</p>"))
            for r in results:
                display(HTML(
                    f"<div style='border:1px solid #ccc; padding:10px; margin:8px; border-radius:8px;'>"
                    f"<b>Rank:</b> {r['rank']}<br>"
                    f"<b>Source:</b> {r['source']}<br>"
                    f"<b>Relevance Score:</b> {r['score']:.4f}<br><br>"
                    f"<b>Text:</b> {r['text']}"
                    f"</div>"
                ))

search_box.observe(on_search_change)

print("Enter a query below:")
display(search_box, output)
