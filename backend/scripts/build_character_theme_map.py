import json
import pandas as pd
import re
from pathlib import Path

# Characters who are absent from the themes CSV Meaning rows but play
# notable narrative roles. These are merged in AFTER the automated scan
# so a CSV rebuild never loses them.
MANUAL_OVERRIDES_NOLI = {
    "Tinyente Guevarra": ["Kapangyarihan at Kawalang-Katarungan", "Kalayaan at Pagmamahal sa Bayan"],
    "Padre Sibyla":      ["Relihiyon", "Kapangyarihan at Kawalang-Katarungan", "Katiwalian"],
    "Tiya Isabel":       ["Pamilya at Dangal", "Pagkamapagpatuloy"],
    "Don Filipo":        ["Pag-asa at Reporma", "Kapangyarihan at Kawalang-Katarungan", "Kalayaan at Pagmamahal sa Bayan"],
    "Lucas":             ["Paghihiganti", "Kapangyarihan at Kawalang-Katarungan"],
    "Kapitan Pablo":     ["Kapangyarihan at Kawalang-Katarungan", "Paghihiganti", "Kalayaan at Pagmamahal sa Bayan"],
    "Kapitan Basilio":   ["Kapangyarihan at Kawalang-Katarungan", "Relihiyon", "Katiwalian"],
}
MANUAL_OVERRIDES_ELFILI: dict = {}  # None needed for Fili at this time

def build_character_theme_map():
    base_dir = Path(__file__).resolve().parent.parent.parent
    backend_data_dir = base_dir / "backend" / "app" / "data"
    
    # Load characters
    aliases_path = backend_data_dir / "character_aliases.json"
    if not aliases_path.exists():
        print(f"File not found: {aliases_path}")
        return
        
    with open(aliases_path, 'r', encoding='utf-8') as f:
        character_data = json.load(f)
        
    characters = []
    for c in character_data:
        name = c.get('name', '')
        aliases = c.get('aliases', [])
        if name and name not in aliases:
            aliases.append(name)
        # compile regex for each alias for fast matching
        pat_list = []
        for a in aliases:
            if not a: continue
            # Avoid matching inside words
            pat_list.append(r'\b' + re.escape(a.lower()) + r'\b')
        characters.append({
            'canon_name': name,
            'pattern': re.compile('|'.join(pat_list), re.IGNORECASE) if pat_list else None
        })
        
    # Process both books
    manual_overrides_by_book = {'noli': MANUAL_OVERRIDES_NOLI, 'elfili': MANUAL_OVERRIDES_ELFILI}
    for book in ['noli', 'elfili']:
        csv_path = base_dir / "csvFiles" / f"{book}_themes.csv"
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        
        # Build mapping: canon_name -> set of tagalog theme labels
        char_to_themes: dict[str, set] = {}
        
        for _, row in df.iterrows():
            tagalog_title = row['Tagalog Title']
            meaning = str(row['Meaning']).lower()
            
            for char in characters:
                if char['pattern'] and char['pattern'].search(meaning):
                    if char['canon_name'] not in char_to_themes:
                        char_to_themes[char['canon_name']] = set()
                    char_to_themes[char['canon_name']].add(tagalog_title)
                    
        # Convert sets to list
        output_data = {k: list(v) for k, v in char_to_themes.items()}
        
        # Merge manual overrides — these characters are absent from CSV Meaning rows
        overrides = manual_overrides_by_book.get(book, {})
        for char_name, themes in overrides.items():
            if char_name in output_data:
                # Merge without duplicates
                existing = set(output_data[char_name])
                output_data[char_name] = list(existing | set(themes))
            else:
                output_data[char_name] = list(themes)
        
        output_path = backend_data_dir / f"character_theme_map_{book}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
            
        print(f"Generated {output_path.name}: {len(output_data)} characters mapped to themes.")
        print(f"  (includes {len(overrides)} manual override entries)")

if __name__ == "__main__":
    build_character_theme_map()
