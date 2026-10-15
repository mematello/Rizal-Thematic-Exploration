"""
Word Vector Cache Precomputation Script
Collects the exact word set that the validation loop can encode
from engine.py, encodes each word ONE AT A TIME with base_model.encode(w, show_progress_bar=False),
and saves the result as a versioned file with model name and vocabulary hash.
"""
import os
import sys
import time
import hashlib
import pickle
import random
import argparse
from pathlib import Path
import numpy as np
import torch

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.config import get_settings
from app.core.engine import RizalEngine

def get_exact_candidate_words(engine: RizalEngine) -> tuple[list[str], str, str]:
    """
    Collects the exact set of words the validation loop can encode in RizalEngine.
    Extracts the exact tokenization and filter logic:
      - w in engine.vocabulary
      - len(w) > 4
      - w not in engine.LOW_INFO_WORDS
    Returns sorted list of candidate words, full vocabulary hash, and candidate words hash.
    """
    vocab = engine.vocabulary
    low_info = engine.LOW_INFO_WORDS

    # Exact filtering logic matching engine.py lines 1164-1173:
    # cand_words = [w for w in sent_words if len(w) > 4 and w not in self.LOW_INFO_WORDS]
    # if w not in self.vocabulary: continue
    candidate_words = sorted([w for w in vocab if len(w) > 4 and w not in low_info])

    vocab_str = "\n".join(sorted(vocab))
    vocab_hash = hashlib.sha256(vocab_str.encode("utf-8")).hexdigest()

    cand_str = "\n".join(candidate_words)
    cand_hash = hashlib.sha256(cand_str.encode("utf-8")).hexdigest()

    return candidate_words, vocab_hash, cand_hash

def main():
    parser = argparse.ArgumentParser(description="Precompute word vector cache for RizalEngine")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing cache against fresh encode")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of random samples for verification")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Interval for saving checkpoints")
    args = parser.parse_args()

    settings = get_settings()

    print("=" * 80)
    print("WORD VECTOR CACHE PRECOMPUTATION")
    print(f"Model: {settings.BERT_MODEL_NAME}")
    print("=" * 80)

    # Initialize engine to load base_model, vocabulary, and LOW_INFO_WORDS
    t_init = time.perf_counter()
    engine = RizalEngine()
    print(f"Engine initialized in {time.perf_counter() - t_init:.2f} s")

    candidate_words, vocab_hash, cand_hash = get_exact_candidate_words(engine)
    total_words = len(candidate_words)
    print(f"Total vocabulary size: {len(engine.vocabulary)}")
    print(f"Exact candidate words to encode: {total_words}")
    print(f"Vocabulary SHA-256 hash: {vocab_hash}")
    print(f"Candidate words SHA-256 hash: {cand_hash}")

    # Output file paths
    app_data_dir = backend_dir / "app" / "data"
    app_data_dir.mkdir(parents=True, exist_ok=True)
    cache_path = app_data_dir / "word_vec_cache_v1.pkl"
    checkpoint_path = app_data_dir / "word_vec_cache_checkpoint.pkl"

    docs_perf_dir = Path(__file__).resolve().parent.parent / "docs" / "perf"
    docs_perf_dir.mkdir(parents=True, exist_ok=True)
    docs_cache_path = docs_perf_dir / "word_vec_cache_v1.pkl"

    vectors: dict[str, np.ndarray] = {}

    if args.verify_only:
        if not cache_path.exists():
            print(f"[FAIL] Cache file does not exist at {cache_path}")
            sys.exit(1)
        print(f"Loading existing cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        vectors = data["vectors"]
    else:
        # Load from checkpoint or existing file if available
        if checkpoint_path.exists():
            print(f"Found checkpoint at {checkpoint_path}. Resuming...")
            with open(checkpoint_path, "rb") as f:
                checkpoint_data = pickle.load(f)
                if (checkpoint_data.get("vocab_hash") == vocab_hash and
                    checkpoint_data.get("model_name") == settings.BERT_MODEL_NAME):
                    vectors = checkpoint_data.get("vectors", {})
                    print(f"Resumed with {len(vectors)} / {total_words} words already encoded.")
                else:
                    print("Checkpoint metadata mismatch. Starting fresh.")
                    vectors = {}
        elif cache_path.exists():
            print(f"Found existing cache at {cache_path}. Checking...")
            with open(cache_path, "rb") as f:
                existing_data = pickle.load(f)
                if (existing_data.get("vocab_hash") == vocab_hash and
                    existing_data.get("model_name") == settings.BERT_MODEL_NAME and
                    len(existing_data.get("vectors", {})) == total_words):
                    vectors = existing_data.get("vectors", {})
                    print(f"Existing cache is complete ({len(vectors)} words). Proceeding to verification.")

        # Encode missing words ONE AT A TIME
        words_to_encode = [w for w in candidate_words if w not in vectors]
        if words_to_encode:
            print(f"\nStarting encoding for {len(words_to_encode)} words (ONE AT A TIME)...")
            t_start = time.perf_counter()
            done = len(vectors)

            for i, word in enumerate(words_to_encode, 1):
                # Strictly encode ONE AT A TIME with base_model.encode(w, show_progress_bar=False)
                vec = engine.base_model.encode(word, show_progress_bar=False)
                vec = np.asarray(vec, dtype=np.float32)
                vec.flags.writeable = False
                vectors[word] = vec
                done += 1

                if i % 500 == 0 or i == len(words_to_encode):
                    elapsed = time.perf_counter() - t_start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta_s = (len(words_to_encode) - i) / rate if rate > 0 else 0
                    print(f"  [{done}/{total_words}] ({done/total_words*100:.1f}%) | "
                          f"Encoded {i}/{len(words_to_encode)} | "
                          f"{rate:.1f} words/s | ETA: {eta_s/60:.1f} min")

                    # Periodic checkpoint save
                    if i % args.checkpoint_interval == 0:
                        temp_save = {
                            "version": 1,
                            "model_name": settings.BERT_MODEL_NAME,
                            "vocab_hash": vocab_hash,
                            "candidate_hash": cand_hash,
                            "word_count": len(vectors),
                            "vectors": vectors
                        }
                        with open(checkpoint_path, "wb") as f:
                            pickle.dump(temp_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            total_elapsed = time.perf_counter() - t_start
            print(f"\n[DONE] Finished encoding all {total_words} words in {total_elapsed:.1f} s ({total_elapsed/60:.2f} min).")

        # Save final versioned cache file
        print(f"\nSaving final precomputed cache to {cache_path}...")
        final_data = {
            "version": 1,
            "model_name": settings.BERT_MODEL_NAME,
            "vocab_hash": vocab_hash,
            "candidate_hash": cand_hash,
            "word_count": len(vectors),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vectors": vectors
        }
        with open(cache_path, "wb") as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Also save copy to docs/perf for documentation/distribution
        print(f"Saving copy to {docs_cache_path}...")
        with open(docs_cache_path, "wb") as f:
            pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Clean up checkpoint
        if checkpoint_path.exists():
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass

        file_size_mb = cache_path.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB ({len(vectors)} vectors)")

    # Verification (a): np.array_equal for a random sample of 500 loaded vectors vs fresh encode
    print("\n" + "=" * 80)
    print(f"VERIFICATION (a): np.array_equal for {args.sample_size} random loaded vectors vs fresh encode")
    print("=" * 80)

    random.seed(42)
    sample_words = random.sample(candidate_words, min(args.sample_size, len(candidate_words)))

    mismatches = 0
    t0_verify = time.perf_counter()
    for idx, word in enumerate(sample_words, 1):
        cached_vec = vectors[word]
        fresh_vec = engine.base_model.encode(word, show_progress_bar=False)
        fresh_vec = np.asarray(fresh_vec, dtype=np.float32)

        if not np.array_equal(cached_vec, fresh_vec):
            max_diff = np.max(np.abs(cached_vec - fresh_vec))
            print(f"  [FAIL] Vector mismatch for '{word}'! Max abs diff: {max_diff}")
            mismatches += 1
            if mismatches >= 10:
                print("  Aborting further verification due to excessive mismatches.")
                break

    t_verify = time.perf_counter() - t0_verify
    if mismatches == 0:
        print(f"[PASS] All {len(sample_words)} sample vectors are 100% bit-identical to fresh encode! (Verified in {t_verify:.2f} s)")
    else:
        print(f"[FAIL] Found {mismatches} vector mismatches!")
        sys.exit(1)

    print("\nPrecomputation and sample verification completed successfully.")

if __name__ == "__main__":
    main()
