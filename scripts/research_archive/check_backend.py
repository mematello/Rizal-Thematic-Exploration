import requests
import json

def check_character(name):
    url = "http://localhost:8000/api/v1/characters/chapters"
    params = {"name": name, "sort_by": "number"}
    print(f"\n--- Checking: {name} ---")
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        noli_count = sum(1 for c in data if c['book'] == 'noli')
        fili_count = sum(1 for c in data if c['book'] in ['fili', 'elfili'])
        
        print(f"Total: {len(data)}")
        print(f"Noli: {noli_count}")
        print(f"Fili: {fili_count}")
        
        if fili_count == 0:
            print("WARNING: No Fili chapters found!")
        else:
            print("Fili chapters found (Example):")
            print(json.dumps([c for c in data if c['book'] in ['fili', 'elfili']][:1], indent=2))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_character("Crisostomo Ibarra,Simoun")
    check_character("Basilio")
