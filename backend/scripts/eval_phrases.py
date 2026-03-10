import os
import subprocess
import time

queries = [
    "edukasyon ng pilipino",
    "pag-aaral ng kabataan",
    "paaralan sa pilipinas",
    "ano ang edukasyon",
    "bakit ipinagbawal ang pag-aaral",
    "kapangyarihan ng simbahan",
    "prayle sa pilipinas",
    "role ng simbahan sa lipunan",
    "kalayaan ng pilipino",
    "kolonyal na pamahalaan",
    "mga prayle at kastila",
    "bitcoin price",
    "facebook messenger",
    "covid pandemic",
    "online classes"
]

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
        
        # Parse output to extract specific fields requested by user
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
                
        # Optional: Print raw output if something went wrong
        if "Traceback" in result.stderr:
            print("\nERROR:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running query: {e}")

if __name__ == "__main__":
    for q in queries:
        run_query(q)
        time.sleep(1) # prevent connection flooding just in case
