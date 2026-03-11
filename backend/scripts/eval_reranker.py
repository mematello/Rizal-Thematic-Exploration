import json
import subprocess
import time
import os

QUERIES = [
    "edukasyon",
    "bakit mahalaga ang edukasyon",
    "ano ang edukasyon ng pilipino",
    "kalagayan ng mga indio",
    "pang-aapi ng kastila",
    "pag-aaral ng kabataan",
    "paaralan sa pilipinas",
    "tiktok",
    "bitcoin price",
    "netflix"
]

BOOKS = ["noli", "elfili"]
MODE = "full"
OUTPUT_FILE = "/tmp/eval_reranker.txt"

def run_debug_search(query, book):
    cmd = [
        "python3", "scripts/debug_search.py",
        "--query", query,
        "--book", book,
        "--mode", MODE,
        "--top_k", "5"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    try:
        backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True, cwd=backend_dir)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"ERROR running query: {e}\n{e.stderr}\n{e.stdout}"

def main():
    print(f"Starting Reranker Evaluation...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("RAG Reranker Diagnostics\n")
        f.write("=" * 50 + "\n\n")
        for q in QUERIES:
            print(f"Evaluating: '{q}'...")
            f.write(f"=== Query: {q} ===\n")
            for book in BOOKS:
                f.write(f"\n--- Novel: {book.upper()} ---\n")
                output = run_debug_search(q, book)
                
                lines = output.split('\n')
                relevant_lines = []
                capture = False
                for line in lines:
                    if "--- METADATA ---" in line:
                        capture = True
                    if capture:
                        relevant_lines.append(line)
                
                if relevant_lines:
                    f.write('\n'.join(relevant_lines) + "\n")
                else:
                    f.write(output + "\n")
            f.write("\n" + "=" * 50 + "\n\n")

    print(f"\nEvaluation complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
