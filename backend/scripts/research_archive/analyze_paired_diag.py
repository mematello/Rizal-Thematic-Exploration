import json

with open('paired_diagnostic_results.json') as f:
    data = json.load(f)

print(f"{'Query':<35} | {'Natives':<20} | {'Enriched':<12} | {'Found':<5} | {'ThemeScore':<10} | {'Mode':<15} | {'StageA_Cands'}")
print("-" * 125)

for d in data:
    found_status = str(d['TotalResults']) if d['TotalResults'] > 0 else "0"
    theme_score = d['ThemeScore'] if d['ThemeScore'] != 'None' else 'N/A'
    native = d['NativeTokens']
    enriched = d.get('EnrichmentAnchor', 'None')
    print(f"{d['Query']:<35} | {native:<20} | {enriched:<12} | {found_status:<5} | {theme_score:<10} | {d['ResultMode']:<15} | {d['Stage_A_Candidates']}")
