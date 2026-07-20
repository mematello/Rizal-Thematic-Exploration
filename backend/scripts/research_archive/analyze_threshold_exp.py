import json

with open("threshold_experiment.json", "r") as f:
    data = json.load(f)

print("=== BASELINE RESULTS ===")
print("Query | Rslt | Score | Mode | FailStage")
for r in data['baseline']:
    print(f"{r['Query'][:22]:<22} | {r['TotalResults']:<4} | {r['TopScore'][:10]:<10} | {r['ResultMode']:<15} | {r['FailStage']}")

print("\n=== EXPERIMENT RESULTS ===")
for exp_name, results in data['experiments'].items():
    print(f"\nExperiment: {exp_name}")
    for r in results:
        cnt = r['count']
        snip = r['snippet'][:40] if r['snippet'] else ""
        print(f"  {r['query']:<20} -> {cnt} results | {snip}")
