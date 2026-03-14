import requests
import json

def check_search():
    url = "http://127.0.0.1:8000/api/v1/search"
    params = {"q": "edukasyon", "source_type": "full"}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        noli_results = data.get("results", {}).get("noli", [])
        print(f"Found {len(noli_results)} results in Noli")
        
        for i, res in enumerate(noli_results[:3]):
            print(f"\nResult {i+1}:")
            print(f"  ID: {res.get('id')}")
            print(f"  Sentence: {res.get('sentence_text')[:100]}...")
            print(f"  Themes: {res.get('themes')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_search()
