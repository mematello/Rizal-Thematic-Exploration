import re
from app.core.analyzer import extract_words
from app.services.suggestions import DynamicSuggestionGenerator

def run_stress_test():
    queries = [
        "si elias",
        "si damaso",
        "si ibarra",
        "damaso",
        "prayle",
        "mga prayle",
        "kastila",
        "kastila sa pilipinas",
        "ano ang ginawa ni simoun",
        "ano ang nangyari kay basilio",
        "bakit namatay si elias",
        "simbahan sa pilipinas"
    ]
    
    gen = DynamicSuggestionGenerator
    
    print("\n================ MESSY QUERY STRESS TEST ================\n")
    
    hit_count = 0
    suppress_count = 0
    
    for q in queries:
        query_cleaned = q.lower().strip()
        matched_class = None
        matched_key = None
        sugs = []
        
        # 1. Question Rule
        query_words_set = set(extract_words(query_cleaned))
        if query_words_set.intersection(gen.QUESTION_WORDS):
            matched_class = "Class 4 (Question)"
            matched_key = "N/A"
            sugs = []
        else:
            # 2. Alias Normalization
            resolved_key = query_cleaned
            if query_cleaned in gen.THEMATIC_ALIASES:
                resolved_key = gen.THEMATIC_ALIASES[query_cleaned]
                
            # 3. Exact Dictionary Match (on resolved key)
            if resolved_key in gen.THEMATIC_SUGGESTIONS:
                matched_key = resolved_key
                # Hacky mapping for reporting
                if matched_key in ["edukasyon", "simbahan", "katarungan", "pang-aapi", "kalayaan", "korupsyon", "kapangyarihan", "paghihiganti", "rebolusyon", "karalitaan", "karapatan", "pag-ibig", "pamilya", "lipunan", "bayani", "kastila"]:
                    matched_class = "Class 1 (Broad Theme)"
                elif matched_key in ["elias", "maria clara", "crisostomo ibarra", "ibarra", "padre damaso", "simoun", "basilio", "crispin", "sisa", "padre salvi", "padre florentino", "kapitan tiyago", "pilosopo tasyo", "donya victorina", "donya consolacion", "kabesang tales", "paulita gomez", "isagani", "tiya isabel", "tiburcio"]:
                    matched_class = "Class 3 (Entity Exact)"
                else:
                    matched_class = "Class 2 (Theme Phrase)"
                
                sugs = gen.THEMATIC_SUGGESTIONS[resolved_key]
            else:
                # 4. Dynamic Entity Intercept
                entity_keys = [
                    "maria clara", "crisostomo ibarra", "padre damaso", "padre salvi", 
                    "padre florentino", "kapitan tiyago", "pilosopo tasyo", "donya victorina",
                    "donya consolacion", "kabesang tales", "paulita gomez", "tiya isabel",
                    "elias", "ibarra", "simoun", "basilio", "crispin", "sisa", "isagani", "tiburcio"
                ]
                found_entity = None
                for e in entity_keys:
                    if re.search(r'\b' + re.escape(e) + r'\b', query_cleaned):
                        found_entity = e
                        break
                        
                if found_entity:
                    matched_class = "Class 3 (Entity Dynamic Canonical)"
                    matched_key = found_entity
                    sugs = gen.THEMATIC_SUGGESTIONS[found_entity]
                else:
                    # Check Alias intercept
                    for alias, canonical in gen.THEMATIC_ALIASES.items():
                        if re.search(r'\b' + re.escape(alias) + r'\b', query_cleaned):
                            if canonical in gen.THEMATIC_SUGGESTIONS:
                                found_entity = canonical
                                matched_class = "Class 3 (Entity Dynamic Alias)"
                                matched_key = f"{alias} -> {canonical}"
                                sugs = gen.THEMATIC_SUGGESTIONS[canonical]
                                break
                    
                    if not found_entity:
                        # 5. Fallback (Literal fragment / Unknown)
                        matched_class = "Class 5 (Fragment / Unknown)"
                        matched_key = "N/A"
                        sugs = []
                    
        if sugs:
            hit_count += 1
            status = "[RENDERED]"
        else:
            suppress_count += 1
            status = "[SUPPRESSED]"
            
        print(f"Query: '{q}'")
        print(f"  Class      : {matched_class}")
        print(f"  Match Key  : {matched_key}")
        print(f"  Status     : {status}")
        if sugs:
            print(f"  Suggestions:")
            for i, s in enumerate(sugs, 1):
                print(f"    {i}. {s}")
        print("")
        
    print("--------------------------------------------------")
    print(f"TOTAL QUERIES: {len(queries)}")
    print(f"RENDERED: {hit_count} ({(hit_count/len(queries))*100:.1f}%)")
    print(f"SUPPRESSED: {suppress_count} ({(suppress_count/len(queries))*100:.1f}%)")
    print("==================================================\n")

if __name__ == "__main__":
    run_stress_test()
