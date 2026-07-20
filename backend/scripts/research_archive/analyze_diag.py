import json

with open('diagnostic_results.json') as f:
    data = json.load(f)

print(f"{'Query':<30} | {'Lang':<10} | {'Found':<5} | {'ThemeScore':<10} | {'ResultMode':<15} | {'BridgeTokens'}")
print("-" * 100)

for d in data:
    if d['TotalResults'] == 0:
        found_status = "0"
    else:
        found_status = str(d['TotalResults'])
        
    theme_score = d['ThemeScore'] if d['ThemeScore'] != 'None' else 'N/A'
    
    # Determine basic language flag by looking if the native tokens are empty and crosslingual is true
    if not d['CrossLingualDetected']:
        lang = "Tagalog"
    elif d['Category'] == "mixed_entity":
        lang = "Mixed"
    else:
        lang = "Foreign"

    print(f"{d['Query']:<30} | {lang:<10} | {found_status:<5} | {theme_score:<10} | {d['ResultMode']:<15} | {d['BridgeTokens'][:40]}")
