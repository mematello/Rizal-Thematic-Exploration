import requests
import json
import hashlib

sid = 39363
r_hf = requests.get(f"https://mematello-rizal-thematic-search-api.hf.space/api/v1/sentences/{sid}/sanggunian")
r_loc = requests.get(f"http://127.0.0.1:8000/api/v1/sentences/{sid}/sanggunian")

print("HF status:", r_hf.status_code)
print("Loc status:", r_loc.status_code)
print("HF bytes len:", len(r_hf.content))
print("Loc bytes len:", len(r_loc.content))

hf_d = r_hf.json()
loc_d = r_loc.json()

print("HF keys:", list(hf_d.keys()))
print("Loc keys:", list(loc_d.keys()))

for k in hf_d:
    val_hf = hf_d[k]
    val_loc = loc_d.get(k)
    match = val_hf == val_loc
    print(f"Field '{k}': match={match} (HF={val_hf} | Loc={val_loc})")

# If full_is_short is removed from loc_d:
loc_d_copy = dict(loc_d)
loc_d_copy["full_is_short"] = []
norm_hf = json.dumps(hf_d, sort_keys=True)
norm_loc = json.dumps(loc_d_copy, sort_keys=True)
print("Normalized match without full_is_short difference:", norm_hf == norm_loc)
print("HF SHA:", hashlib.sha256(norm_hf.encode()).hexdigest())
print("Loc SHA:", hashlib.sha256(norm_loc.encode()).hexdigest())
