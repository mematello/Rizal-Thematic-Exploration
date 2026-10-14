import time
import requests
import statistics

# Test 1: localhost without Session (what benchmark.py currently does)
times_lh = []
for i in range(20):
    t0 = time.perf_counter()
    r = requests.get('http://localhost:8002/api/v1/search', params={'q': 'bayan', 'source_type': 'summary', 'novel': 'both'})
    t1 = time.perf_counter()
    times_lh.append((t1 - t0) * 1000)
    time.sleep(0.02)

# Test 2: 127.0.0.1 without Session
times_127 = []
for i in range(20):
    t0 = time.perf_counter()
    r = requests.get('http://127.0.0.1:8002/api/v1/search', params={'q': 'bayan', 'source_type': 'summary', 'novel': 'both'})
    t1 = time.perf_counter()
    times_127.append((t1 - t0) * 1000)
    time.sleep(0.02)

# Test 3: localhost WITH Session (connection reuse)
s_lh = requests.Session()
times_lh_sess = []
for i in range(20):
    t0 = time.perf_counter()
    r = s_lh.get('http://localhost:8002/api/v1/search', params={'q': 'bayan', 'source_type': 'summary', 'novel': 'both'})
    t1 = time.perf_counter()
    times_lh_sess.append((t1 - t0) * 1000)
    time.sleep(0.02)

# Test 4: 127.0.0.1 WITH Session (connection reuse)
s_127 = requests.Session()
times_127_sess = []
for i in range(20):
    t0 = time.perf_counter()
    r = s_127.get('http://127.0.0.1:8002/api/v1/search', params={'q': 'bayan', 'source_type': 'summary', 'novel': 'both'})
    t1 = time.perf_counter()
    times_127_sess.append((t1 - t0) * 1000)
    time.sleep(0.02)

print("=== 20x RUNS: BAYAN ===")
print("1. requests.get('http://localhost:8002') [NO Session]:")
print(f"   Min:    {min(times_lh):.2f} ms")
print(f"   Max:    {max(times_lh):.2f} ms")
print(f"   Mean:   {statistics.mean(times_lh):.2f} ms")
print(f"   Median: {statistics.median(times_lh):.2f} ms")
print(f"   Raw:    {[round(x, 1) for x in times_lh]}")

print("\n2. requests.get('http://127.0.0.1:8002') [NO Session]:")
print(f"   Min:    {min(times_127):.2f} ms")
print(f"   Max:    {max(times_127):.2f} ms")
print(f"   Mean:   {statistics.mean(times_127):.2f} ms")
print(f"   Median: {statistics.median(times_127):.2f} ms")
print(f"   Raw:    {[round(x, 1) for x in times_127]}")

print("\n3. requests.Session('http://localhost:8002') [REUSED Session]:")
print(f"   Min:    {min(times_lh_sess):.2f} ms")
print(f"   Max:    {max(times_lh_sess):.2f} ms")
print(f"   Mean:   {statistics.mean(times_lh_sess):.2f} ms")
print(f"   Median: {statistics.median(times_lh_sess):.2f} ms")
print(f"   Raw:    {[round(x, 1) for x in times_lh_sess]}")

print("\n4. requests.Session('http://127.0.0.1:8002') [REUSED Session]:")
print(f"   Min:    {min(times_127_sess):.2f} ms")
print(f"   Max:    {max(times_127_sess):.2f} ms")
print(f"   Mean:   {statistics.mean(times_127_sess):.2f} ms")
print(f"   Median: {statistics.median(times_127_sess):.2f} ms")
print(f"   Raw:    {[round(x, 1) for x in times_127_sess]}")
