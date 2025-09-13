import pandas as pd

# Load the CSV file
df = pd.read_csv("cleaned_elfili_corpus.csv")

# Extract only the chapter_title column
chapter_titles = df["chapter_title"]

# Print them
print(chapter_titles)

# If you want to save the extracted titles into a new CSV
chapter_titles.to_csv("elfili_chapter_titles.csv", index=False, header=["chapter_title"])
