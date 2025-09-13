import pandas as pd

# Load CSVs
first_df = pd.read_csv("ENRICHING_ELFILI.csv")   # has complete 1–39
second_df = pd.read_csv("enhanced_elfili_chapters.csv") # has missing but updated content

# Merge, prioritizing second_df content
merged_df = pd.concat([second_df, first_df]).drop_duplicates(subset=["chapter_number"], keep="first")

# Sort by chapter_number to restore order
merged_df = merged_df.sort_values(by="chapter_number")

# Save final CSV
merged_df.to_csv("complete_chapters.csv", index=False)

print("Complete chapters saved to complete_chapters.csv")
