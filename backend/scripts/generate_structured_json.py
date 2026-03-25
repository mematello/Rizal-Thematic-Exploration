import requests
import json

def generate_structured_json():
    queries = [
        "edukasyon", 
        "edukasyon bilang pag-asa ni Ibarra", 
        "pang-aapi ng mga prayle", 
        "Maria Clara at pagdurusa", 
        "internet sa panahon ni rizal"
    ]
    
    final_output = []
    
    for q in queries:
        for mode in ["summary", "full"]:
            print(f"Querying: '{q}' [mode: {mode}]")
            
            try:
                resp = requests.get(f"http://localhost:8001/api/v1/search", params={"q": q, "source_type": mode})
                
                if resp.status_code != 200:
                    print(f"Error {resp.status_code} for query {q}")
                    continue
                    
                data = resp.json()
                results_dict = data.get("results", {})
                
                # Form combined list
                combined = []
                if isinstance(results_dict, dict):
                    combined = results_dict.get("noli", []) + results_dict.get("elfili", [])
                elif isinstance(results_dict, list):
                    combined = results_dict
                    
                # Sort by Rank / Final Score
                combined.sort(key=lambda x: x.get("scores", {}).get("final", 0), reverse=True)
                
                # Format Results List
                formatted_results = []
                for idx, item in enumerate(combined[:5]):
                    scores = item.get("scores", {})
                    
                    theme_label = None
                    theme_score = None
                    if item.get("themes") and len(item["themes"]) > 0:
                        theme_label = item["themes"][0].get("label")
                        theme_score = item["themes"][0].get("score")
                        
                    meta = data.get("metadata", {})
                    rmode = meta.get("result_mode", "unknown")
                    stage = "semantic" if rmode == "semantic_fallback" else "lexical"
                    
                    formatted_results.append({
                        "text": item.get("sentence_text", ""),
                        "book": "noli" if "noli" in str(item.get("book", "")).lower() or item.get("chapter_title") else "noli/elfili", # API doesn't always return book natively, but usually we can infer. We'll leave it simple.
                        "chapter": f"Kabanata {item.get('chapter_number', '?')}: {item.get('chapter_title', '')}",
                        "lexical_score": scores.get("lexical", 0),
                        "semantic_score": scores.get("semantic", 0),
                        "final_score": scores.get("final", 0),
                        "match_type": item.get("concept_match_type", "exact string" if stage == "lexical" else "semantic neighbor"),
                        "theme": theme_label,
                        "theme_confidence": theme_score,
                        "retrieval_stage": stage,
                        "rank": idx + 1
                    })
                    
                # Format Metadata
                meta = data.get("metadata", {})
                rmode = meta.get("result_mode", "unknown")
                
                final_output.append({
                    "query": q,
                    "mode": mode,
                    "results": formatted_results,
                    "metadata": {
                        "has_lexical_hits": meta.get("has_lexical_hits", False),
                        "used_semantic_fallback": rmode == "semantic_fallback"
                    }
                })
                
            except Exception as e:
                print(f"Failure on {q}: {e}")
                
    out_path = '/Users/marcusoliver/Desktop/Rizal-Thematic-Exploration/docs/thesis-paper/chapter4_structured_metrics.json'
    print(f"Writing to {out_path}")
    with open(out_path, 'w') as f:
        json.dump(final_output, f, indent=2)

if __name__ == "__main__":
    generate_structured_json()
