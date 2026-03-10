import os
import subprocess
import time

queries_by_category = {
    "1. Single-word semantic fallback": [
        "edukasyon",
        "kalayaan",
        "karunungan",
        "kapangyarihan",
        "pamamahala"
    ],
    "2. Lexical matches": [
        "simbahan",
        "prayle",
        "kastila",
        "paaralan",
        "instruccion"
    ],
    "3. Phrase queries (multi-word)": [
        "edukasyon ng pilipino",
        "pag-aaral ng kabataan",
        "bakit ipinagbawal ang pag-aaral",
        "ano ang edukasyon",
        "paaralan sa pilipinas"
    ],
    "4. Religion / society phrases": [
        "kapangyarihan ng simbahan",
        "prayle sa pilipinas",
        "mga prayle at kastila",
        "pari at prayle"
    ],
    "5. Historical concepts": [
        "kolonyal na pamahalaan",
        "kalayaan ng pilipino",
        "kalagayan ng mga indio",
        "pang-aapi ng kastila"
    ],
    "6. Natural student queries": [
        "ano ang papel ng simbahan",
        "bakit mahalaga ang edukasyon",
        "ano ang sinabi ni rizal tungkol sa edukasyon",
        "paano pinigilan ang pag-aaral",
        "ano ang kalagayan ng pilipino noon"
    ],
    "7. Out-of-domain single words": [
        "tiktok",
        "bitcoin",
        "facebook",
        "netflix",
        "instagram",
        "covid",
        "wifi"
    ],
    "8. Out-of-domain phrases": [
        "bitcoin price",
        "facebook messenger",
        "covid pandemic",
        "online classes",
        "netflix movies",
        "instagram reels"
    ],
    "9. Edge-case stopword queries": [
        "ano ang edukasyon ng pilipino",
        "sino ang prayle",
        "bakit may simbahan",
        "ano ang kalayaan"
    ]
}

def run_query(query):
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print(f"{'='*60}")
    
    cmd = ["python", "scripts/debug_search.py", "--book", "noli", "--mode", "full", "--query", query, "--top_k", "5"]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["DEBUG_SEARCH"] = "1"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        lines = result.stdout.split('\n')
        
        mode = "N/A"
        anchor_score = "N/A"
        trigger_tokens = []
        top_results = []
        
        in_results = False
        current_result = ""
        
        for line in lines:
            if "Result Mode:" in line:
                mode = line.split("Result Mode:")[1].strip()
            elif "[DEBUG] Theme Anchor Score:" in line:
                anchor_score = line.split("Theme Anchor Score:")[1].split("|")[0].strip()
            elif "[DEBUG] Passed Validation:" in line and "Trigger Token:" in line:
                token = line.split("Trigger Token:")[1].split("|")[0].strip()
                if token not in trigger_tokens:
                    trigger_tokens.append(token)
            elif "Rank" in line and "ID:" in line:
                in_results = True
                if current_result:
                    top_results.append(current_result)
                current_result = line + "\n"
            elif in_results and line.strip() == "-" * 60:
                if current_result:
                    top_results.append(current_result)
                current_result = ""
                in_results = False
            elif in_results and line.strip():
                current_result += line + "\n"
                
        if current_result:
            top_results.append(current_result)
            
        print(f"Result Mode: {mode}")
        print(f"Theme Anchor Score: {anchor_score}")
        print(f"Validation Trigger Tokens: {', '.join(trigger_tokens)}")
        print("Top 5 Results:")
        if not top_results:
            print("  None")
        else:
            for r in top_results:
                print(f"  {r.strip().replace(chr(10), chr(10)+'  ')}")
                
        if "Traceback" in result.stderr:
            print("\nERROR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running query: {e}")

if __name__ == "__main__":
    for category, queries in queries_by_category.items():
        print(f"\n\n{'#'*80}\n# {category}\n{'#'*80}")
        for q in queries:
            run_query(q)
            time.sleep(0.5)
