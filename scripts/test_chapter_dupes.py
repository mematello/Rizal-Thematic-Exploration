import requests

# Test chapter endpoint for duplicates
url = "http://localhost:8000/api/v1/chapters/noli/1"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(f"Total sentences: {len(data)}")
    
    # Check for duplicates
    seen = set()
    duplicates = []
    
    for i, item in enumerate(data):
        text = item['sentence_text']
        if text in seen:
            duplicates.append((i, text[:50]))
        seen.add(text)
    
    if duplicates:
        print(f"\nFOUND {len(duplicates)} DUPLICATES:")
        for idx, text in duplicates:
            print(f"  Index {idx}: {text}...")
    else:
        print("\nNo duplicates found!")
        
    # Show first 3 sentences
    print("\nFirst 3 sentences:")
    for i in range(min(3, len(data))):
        print(f"{i+1}. {data[i]['sentence_text'][:80]}...")
else:
    print(f"Error: {response.status_code}")
