
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1/search"
query = "kamatayan ni Basilio"

print(f"Testing query: '{query}'")
try:
    response = requests.get(BASE_URL, params={'q': query})
    if response.status_code == 200:
        data = response.json()
        all_results = data.get('noli', []) + data.get('elfili', [])
        
        print(f"Found {len(all_results)} results.")
        
        for item in all_results[:5]:
            text = item['sentence_text'].lower()
            print(f"Result: {text[:100]}...")
            
            # Check for coverage terms
            has_basilio = 'basilio' in text
            has_death = any(w in text for w in ['kamatayan', 'patay', 'yumao', 'libing', 'bangkay'])
            
            print(f"  - Contains Basilio: {has_basilio}")
            print(f"  - Contains Death-related: {has_death}")
            
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Exception: {e}")
