import pandas as pd

# Load the two CSV files
df1 = pd.read_csv("cleaned_elfili_corpus.csv")   # book_title,chapter_number,chapter_title,chapter_text
df2 = pd.read_csv("complete_chapters.csv")   # chapter_number,hs_thematic_explanation

# Merge them on chapter_number
merged_df = pd.merge(df1, df2, on="chapter_number", how="left")

# Ensure correct column order
merged_df = merged_df[[
    "book_title",
    "chapter_number",
    "chapter_title",
    "chapter_text",
    "hs_thematic_explanation"
]]

# Save to new CSV
merged_df.to_csv("ELFILI_chapters_with_thematic.csv", index=False)

print("✅ New CSV created: chapters_with_thematic.csv")
