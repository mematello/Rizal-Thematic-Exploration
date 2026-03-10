import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_search():
    q = "edukasyon"
    mode = "full"
    res = requests.get(f"{BASE_URL}/search", params={"q": q, "source_type": mode})
    data = res.json()
    print(data)

if __name__ == "__main__":
    test_search()
