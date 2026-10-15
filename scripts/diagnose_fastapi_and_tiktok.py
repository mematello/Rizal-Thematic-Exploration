import requests
import json
import time

queries = ["bayan", "oppression", "tiktok"]

print("================================================================================")
print("FASTAPI LAYER BREAKDOWN & ENGINE TIMING ANALYSIS (via http://127.0.0.1:8002)")
print("================================================================================")

for q in queries:
    print(f"\n#################### QUERY: '{q}' ####################")
    for run in range(1, 6):
        t0 = time.perf_counter()
        resp = requests.get(
            "http://127.0.0.1:8002/api/v1/search",
            params={"q": q, "source_type": "summary", "novel": "both"}
        )
        t1 = time.perf_counter()
        http_total_ms = (t1 - t0) * 1000

        data = resp.json()
        headers = resp.headers
        meta = data.get("metadata", {})
        fastapi_layer = meta.get("fastapi_layer", {})
        engine_timings = meta.get("engine_timings", {})

        print(f"\n--- Run {run} ---")
        print(f"HTTP End-to-End Latency: {http_total_ms:.2f} ms")
        print("FastAPI Middleware Headers:")
        print(f"  X-FastAPI-Middleware-Total-Ms: {headers.get('X-FastAPI-Middleware-Total-Ms')} ms")
        print(f"  X-FastAPI-Routing-DI-Ms:       {headers.get('X-FastAPI-Routing-DI-Ms')} ms")
        print(f"  X-FastAPI-DB-Create-Ms:        {headers.get('X-FastAPI-DB-Create-Ms')} ms")
        print(f"  X-FastAPI-Route-Handler-Ms:    {headers.get('X-FastAPI-Route-Handler-Ms')} ms")
        print(f"  X-FastAPI-Serialization-Ms:    {headers.get('X-FastAPI-Serialization-Ms')} ms")

        print("In-Route FastAPI Layer Timings (from response metadata):")
        for k, v in fastapi_layer.items():
            print(f"  {k:<28}: {v} ms")

        print("Engine Timings (from response metadata):")
        for k in [
            "search_total_ms", "translator_ms", "translator_network_calls",
            "diagnostic_suggestions_ms", "untimed_gap_ms", "base_model_encode",
            "lexical_retrieval", "semantic_fallback", "rerank_loop",
            "theme_classification", "context_expansion", "db_query_count"
        ]:
            if k in engine_timings or k in meta:
                val = engine_timings.get(k, meta.get(k))
                print(f"  {k:<28}: {val}")

        time.sleep(0.05)
