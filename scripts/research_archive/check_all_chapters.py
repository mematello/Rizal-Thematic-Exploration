
import requests

def check_all():
    url = "http://localhost:8000/api/v1/chapters"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        for i, c in enumerate(data):
            if not c.get('chapter_title'):
                print(f"Index {i} has None title: {c}")
    else:
        print(f"Error {res.status_code}: {res.text}")

if __name__ == "__main__":
    check_all()
