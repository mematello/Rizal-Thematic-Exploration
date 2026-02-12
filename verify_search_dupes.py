
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_search_duplicates():
    print("\nTesting Search Duplicates...")
    query = "edukasyon"
    url = f"{BASE_URL}/search?q={query}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            data_results = data.get("results", {})
            results = data_results.get("noli", []) + data_results.get("elfili", [])
            if results:
                print(f"First result structure: {results[0]}")
            
            seen_texts = set()
            duplicates = 0
            for i, res in enumerate(results):
                # Adjust key access based on printed structure
                text = res.get('text') or res.get('sentence_text') # Try possible keys
                if not text:
                    print(f"Skipping item with no text: {res}")
                    continue
                if text in seen_texts:
                    duplicates += 1
                    print(f"DUPLICATE FOUND ({i+1}): {text[:50]}...")
                seen_texts.add(text)
                print(f"{i+1}. [Ch.{res.get('chapter_number')}] {text[:50]}...")
            
            if duplicates > 0:
                print(f"FAIL: Found {duplicates} duplicates.")
            else:
                print("PASS: No duplicates found.")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_search_duplicates()
