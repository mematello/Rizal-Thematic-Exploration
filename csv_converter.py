import re
import pandas as pd

# Input text (you can replace this with reading from a file)
with open("noli_themes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Regex to capture Tagalog (English) + meaning
pattern = r"([A-Za-zÁÉÍÓÚáéíóúñÑ\s\-\’,]+)\s*\(([A-Za-z\s\-\’,]+)\)\n(.*?)(?=\n[A-Za-zÁÉÍÓÚáéíóúñÑ\s\-\’,]+\s*\(|\Z)"
matches = re.findall(pattern, text, re.S)

# Reformat for CSV
rows = []
for tagalog, english, meaning in matches:
    rows.append({
        "English Title": english.strip(),
        "Tagalog Title": tagalog.strip(),
        "Meaning": meaning.strip().replace("\n", " ")
    })

# Create DataFrame
df = pd.DataFrame(rows, columns=["English Title", "Tagalog Title", "Meaning"])

# Save to CSV
df.to_csv("noli_themes.csv", index=False, encoding="utf-8-sig")

print("CSV file created: noli_themes.csv")
