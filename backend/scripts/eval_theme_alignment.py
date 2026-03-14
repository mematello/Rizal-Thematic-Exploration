import sys
import os
import traceback

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.core.engine import RizalEngine
from app.models.database import SessionLocal

def main():
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "pag-aaral ng kabataan",
        "elias",
        "maria clara",
        "korupsyon sa pamahalaan",
        "kalayaan",
        "karapatan ng mga indio"
    ]
    
    print("# Theme-Result Alignment Evaluaton")
    engine = RizalEngine()
    
    with SessionLocal() as db:
        for q in queries:
            print(f"\n## Query: `{q}`")
            try:
                res = engine.search(db=db, query=q, source_type="full")
                results = res.get("results", {})
                
                combined = results.get("noli", []) + results.get("elfili", [])
                top_results = combined[:3]
                
                if not top_results:
                    print("*No results found.*")
                    continue
                    
                for i, item in enumerate(top_results):
                    text = item.get("sentence_text", "")
                    if len(text) > 300:
                        text = text[:297] + "..."
                    
                    themes_raw = item.get("themes", [])
                    
                    themes_str = "None"
                    if themes_raw and isinstance(themes_raw[0], dict):
                        scored_themes = [f"{t.get('label')} ({abs(round(float(t.get('score', 0)), 2))})" for t in themes_raw]
                        themes_str = ", ".join(scored_themes)
                    elif themes_raw and isinstance(themes_raw[0], str):
                        themes_str = ", ".join(themes_raw)
                        
                    print(f"**Result {i+1}:**")
                    print(f"> {text}")
                    print(f"- **Assigned Themes:** {themes_str}")
                    print("---")
            except Exception as e:
                print(f"*Error during search: {e}*")
                traceback.print_exc()

if __name__ == "__main__":
    main()
