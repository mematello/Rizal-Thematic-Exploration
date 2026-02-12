
import requests
import json

def test_chapter_endpoint():
    # Test Noli Chapter 1
    url = "http://localhost:8000/api/v1/chapters/noli/1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"Success! Fetched {len(data)} sentences for Noli Chapter 1.")
            if data:
                print("First sentence:", data[0])
                print("Last sentence:", data[-1])
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error accessing endpoint: {e}")

    # Test Invalid Chapter
    url = "http://localhost:8000/api/v1/chapters/noli/999"
    try:
        response = requests.get(url)
        if response.status_code == 404:
            print("Success! Correctly returned 404 for invalid chapter.")
        else:
            print(f"Failed! Expected 404, got {response.status_code}")
    except Exception as e:
        print(f"Error accessing endpoint: {e}")

if __name__ == "__main__":
    test_chapter_endpoint()
