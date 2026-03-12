import sys
import os

# Add backend directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.suggestions import DynamicSuggestionGenerator
from app.core.engine import get_search_results

def run_ux_eval():
    generator = DynamicSuggestionGenerator()
    
    # 7 Queries Requested
    queries = [
        "edukasyon ng pilipino",
        "pang-aapi ng kastila",
        "simbahan sa pilipinas",
        "elias",
        "maria clara",
        "pag-aaral ng kabataan",
        "korupsyon sa pamahalaan"
    ]
    
    print("=" * 60)
    print("📚 UX VALIDATION PASS: SUGGESTION HELPFULNESS")
    print("=" * 60)
    
    for q in queries:
        print(f"\n▶ QUERY: '{q}'")
        
        # 1. Run Search engine to get fake context text
        # We don't need actual DB results, we just need to see if the generator yields good UX
        results = get_search_results(q)
        
        # 2. Get Suggestions
        suggestions = generator.generate_suggestions(q, results.get("results", {}))
        
        # 3. Print Assessment Format
        if suggestions:
            print("   ↳ SUGGESTIONS SHOWN:")
            for i, sug in enumerate(suggestions):
                print(f"      {i+1}. {sug}")
        else:
            print("   ↳ [SUPPRESSED] - No suggestions generated.")

if __name__ == "__main__":
    run_ux_eval()
