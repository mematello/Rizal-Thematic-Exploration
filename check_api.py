import requests
import json

url = "http://localhost:8000/api/v1/characters/chapters"
params = {
    "name": "Crisostomo Ibarra,Ibarra,Crisostomo",
    "sort_by": "number"
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    
    # Check if any result has book='fili' or 'elfili'
    fili_chapters = [c for c in data if c.get('book') in ['fili', 'elfili']]
    noli_chapters = [c for c in data if c.get('book') in ['noli']]
    
    print(f"Total results: {len(data)}")
    print(f"Noli chapters: {len(noli_chapters)}")
    print(f"Fili chapters: {len(fili_chapters)}")
    
    if fili_chapters:
        print("\nExample Fili chapter:")
        print(json.dumps(fili_chapters[0], indent=2))
        
except Exception as e:
    print(f"Error: {e}")
