import requests
import json

def test_search(query, mode="buod"):
    url = f"http://localhost:8000/api/v1/search?q={query}&source_type={mode}"
    print(f"Testing API: {url}")
    try:
        response = requests.get(url, timeout=120)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            n_noli = len(data.get('results', {}).get('noli', []))
            n_elfili = len(data.get('results', {}).get('elfili', []))
            print(f"Results: Noli={n_noli}, Elfili={n_elfili}")
            if n_noli > 0:
                 print(f"Top Result: {data['results']['noli'][0]['sentence_text'][:100]}")
        else:
            print(f"Error Detail: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_search("Maria")
    test_search("Maria Clara")
