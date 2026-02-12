
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_deduplication():
    print("\nTesting Character Appearance Deduplication...")
    name = "Sisa"
    url = f"{BASE_URL}/characters/appearances?name={name}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Found {len(data)} appearances for '{name}'.")
            
            seen_texts = set()
            duplicates = 0
            for i, app in enumerate(data):
                text = app['sentence_text']
                if text in seen_texts:
                    duplicates += 1
                    print(f"DUPLICATE FOUND: {text[:50]}...")
                seen_texts.add(text)
                print(f"{i+1}. [{app['book']} Ch.{app['chapter_number']}] {text[:50]}...")
            
            if duplicates == 0:
                print("PASS: No duplicates found.")
            else:
                print(f"FAIL: Found {duplicates} duplicates.")
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_deduplication()
