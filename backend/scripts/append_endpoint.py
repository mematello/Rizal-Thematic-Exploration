import sys
from pathlib import Path

file_path = Path("c:/Users/Rhia/Documents/AntiGravity/Rizal-Thematic-Exploration/backend/app/api/v1/content.py")
content = file_path.read_text(encoding="utf-8")

new_code = """

class CharacterThemeResult(BaseModel):
    label: str
    description: str

class CharacterPaksaResponse(BaseModel):
    characterName: str
    themes: List[CharacterThemeResult]

@router.get("/characters/{name}/paksa", response_model=CharacterPaksaResponse)
def get_character_paksa(
    name: str, 
    book: str, 
    db: Session = Depends(get_db), 
    engine: RizalEngine = Depends(get_engine)
):
    book_key = "noli" if book.lower() in ("noli", "noli me tangere") else "elfili"
    
    backend_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    char_aliases_path = backend_data_dir / "character_aliases.json"
    char_aliases = json.loads(char_aliases_path.read_text(encoding='utf-8')) if char_aliases_path.exists() else []
    
    canonical_name = name
    name_lower = name.lower().strip()
    
    for c in char_aliases:
        c_name = c.get('name', '')
        aliases = c.get('aliases', [])
        if name_lower == c_name.lower() or name_lower in [a.lower() for a in aliases]:
            canonical_name = c_name
            break
            
    char_theme_map_path = backend_data_dir / f"character_theme_map_{book_key}.json"
    char_theme_map = json.loads(char_theme_map_path.read_text(encoding='utf-8')) if char_theme_map_path.exists() else {}
    
    associated_themes = char_theme_map.get(canonical_name, [])
    
    engine._ensure_themes_loaded(db)
    
    results = []
    for theme_label in associated_themes:
        meanings = [t['meaning'] for t in getattr(engine, 'theme_cache', []) if t['tagalog_title'] == theme_label]
        
        best_meaning = ""
        if meanings:
            explicit_meanings = [m for m in meanings if re.search(r'\\b' + re.escape(canonical_name) + r'\\b', m, re.IGNORECASE)]
            if explicit_meanings:
                best_meaning = explicit_meanings[0]
            else:
                best_meaning = meanings[0]
                
        results.append(CharacterThemeResult(
            label=theme_label,
            description=best_meaning
        ))
        
    return CharacterPaksaResponse(
        characterName=canonical_name,
        themes=results
    )
"""

if "get_character_paksa" not in content:
    file_path.write_text(content + new_code, encoding="utf-8")
    print("Appended backend endpoint.")
else:
    print("Endpoint already exists.")

