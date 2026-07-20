import pandas as pd
import json

buod_df = pd.read_csv(r"c:\Users\ianku\Documents\VS Code\Rizal-Thematic-Exploration\csvFiles\noli_chapters.csv")
full_df = pd.read_csv(r"c:\Users\ianku\Documents\VS Code\Rizal-Thematic-Exploration\csvFiles\fullversion_noli.csv")

# Filter for Chapter 1
buod_ch1 = buod_df[buod_df['chapter_number'] == 1].sort_values('sentence_number')
full_ch1 = full_df[full_df['chapter_number'] == 1].sort_values('sentence_number')

out = {
    "buod": [{"id": row['sentence_number'], "text": row['sentence_text']} for _, row in buod_ch1.iterrows()],
    "full": [{"id": row['sentence_number'], "text": row['sentence_text']} for _, row in full_ch1.iterrows()]
}

with open(r"c:\Users\ianku\Documents\VS Code\Rizal-Thematic-Exploration\backend\ch1_sentences.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("Done")
