import sys
import os
import re

# Add the project root to sys.path to import the module if needed, 
# but here we can just test the logic isolated from the full class if we want, 
# or import RobustAligner.

def test_tauhan_logic():
    # Mocking the extraction and the loop logic
    
    def extract_tauhan(text, tauhan_list):
        found = set()
        text_lower = text.lower()
        for tauhan in tauhan_list:
            pattern = r'\b' + re.escape(tauhan.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found.add(tauhan.lower())
        return found

    tauhan_list = ["Maria Clara", "Ibarra"]
    
    cases = [
        {
            "name": "Exact match",
            "buod": "Maria Clara and Ibarra meet.",
            "full": "Maria Clara and Ibarra meet in the garden.",
            "expected": 1.0 # 2/2
        },
        {
            "name": "Missing required tauhan (Ibarra)",
            "buod": "Maria Clara and Ibarra meet.",
            "full": "Maria Clara is in the garden.",
            "expected": 0.0
        },
        {
            "name": "Extra tauhan in full",
            "buod": "Maria Clara is here.",
            "full": "Maria Clara and Ibarra are here.",
            "expected": 0.5 # 1/2
        },
        {
            "name": "Single exact match",
            "buod": "Maria Clara is here.",
            "full": "Maria Clara is here.",
            "expected": 1.0 # 1/1
        },
        {
            "name": "No match at all",
            "buod": "Maria Clara is here.",
            "full": "Ibarra is here.",
            "expected": 0.0
        },
        {
            "name": "Empty buod",
            "buod": "The sun is shining.",
            "full": "Maria Clara is here.",
            "expected": 0.0 # Per requested edge case
        }
    ]

    print("Running Tauhan Matching Logic Tests...")
    print("-" * 40)
    
    passed_all = True
    for case in cases:
        b_tauhan = extract_tauhan(case["buod"], tauhan_list)
        w_tauhan = extract_tauhan(case["full"], tauhan_list)
        
        # Implementation of the logic from robust_aligner.py
        if not b_tauhan:
            score = 0.0
        elif b_tauhan.issubset(w_tauhan):
            score = len(b_tauhan) / len(w_tauhan)
        else:
            score = 0.0
            
        status = "PASS" if score == case["expected"] else "FAIL"
        print(f"Case: {case['name']}")
        print(f"  Buod Mentions: {b_tauhan}")
        print(f"  Full Mentions: {w_tauhan}")
        print(f"  Result Score: {score} (Expected: {case['expected']}) -> {status}")
        
        if status == "FAIL":
            passed_all = False
            
    print("-" * 40)
    if passed_all:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    test_tauhan_logic()
