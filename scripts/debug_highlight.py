import requests

# Test character appearances to see sentence_index values
url = "http://localhost:8000/api/v1/characters/appearances?name=Sisa"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(f"Character appearances for 'Sisa': {len(data)} results\n")
    
    for i, app in enumerate(data[:3]):  # Show first 3
        print(f"{i+1}. Book: {app['book']}, Chapter: {app['chapter_number']}")
        print(f"   Sentence Index: {app.get('sentence_index', 'MISSING')}")
        print(f"   Text: {app['sentence_text'][:60]}...\n")
else:
    print(f"Error: {response.status_code}")

# Also check chapter content to see sentence_index values
print("\n" + "="*60)
print("Chapter content for Noli Chapter 1:\n")

url2 = "http://localhost:8000/api/v1/chapters/noli/1"
response2 = requests.get(url2)

if response2.status_code == 200:
    data2 = response2.json()
    print(f"Total sentences: {len(data2)}\n")
    
    for i, sent in enumerate(data2[:3]):  # Show first 3
        print(f"{i+1}. Index: {sent['sentence_index']}")
        print(f"   Text: {sent['sentence_text'][:60]}...\n")
else:
    print(f"Error: {response2.status_code}")
