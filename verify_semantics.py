
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1/search"

queries = [
    "padre damaso",
    "gago ka sisa",
    "rebolusyon"
]

for q in queries:
    print(f"Testing query: '{q}'")
    try:
        response = requests.get(BASE_URL, params={'q': q})
        if response.status_code == 200:
            data = response.json()
            noli_count = len(data.get('noli', []))
            elfili_count = len(data.get('elfili', []))
            print(f"  Result: {noli_count + elfili_count} items found.")
            if noli_count + elfili_count == 0:
                print("  -> REJECTED (Empty results)")
            else:
                print("  -> ACCEPTED")
        else:
            print(f"  Error: {response.status_code}")
    except Exception as e:
        print(f"  Exception: {e}")
    print("-" * 20)
