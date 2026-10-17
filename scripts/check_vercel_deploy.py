import urllib.request
import re
import sys
import time

def check():
    req = urllib.request.Request(
        "https://rizalengine.vercel.app",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8")
    
    chunks = set(re.findall(r'/_next/static/chunks/[^"\']+\.js', html))
    print(f"Found {len(chunks)} unique chunks")
    
    found = False
    for c in chunks:
        url = "https://rizalengine.vercel.app" + c
        try:
            req_c = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req_c) as r:
                content = r.read().decode("utf-8")
                if "/api/v1/chapters/" in content:
                    print(f"FOUND /api/v1/chapters/ in {c}!")
                    found = True
                    break
        except Exception as e:
            print(f"Error fetching {c}: {e}")
            
    return found

if __name__ == "__main__":
    for attempt in range(10):
        print(f"Attempt {attempt+1} checking Vercel...")
        if check():
            print("Vercel deployment is LIVE with the new Sanggunian chapter endpoint!")
            sys.exit(0)
        time.sleep(15)
    print("Timed out waiting for Vercel deployment.")
    sys.exit(1)
