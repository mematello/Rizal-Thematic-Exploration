
import requests

def test_chapters():
    url = "http://localhost:8000/api/v1/chapters"
    print(f"Testing {url}...")
    try:
        res = requests.get(url)
        print(f"Status: {res.status_code}")
        if res.status_code == 200:
            data = res.json()
            print(f"Count: {len(data)}")
            if data:
                print(f"First chapter: {data[0]}")
        else:
            print(f"Error: {res.text}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_chapters()
