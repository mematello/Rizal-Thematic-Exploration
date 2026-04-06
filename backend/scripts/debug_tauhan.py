from app.core.robust_aligner import RobustAligner

def debug_tauhan():
    # User's case: Identity mapping and multi-word suffixes
    tauhan_map = {
        "ibarra": "Crisostomo Ibarra",
        "crisostomo ibarra": "Crisostomo Ibarra",
        "tiago": "Kapitan Tiago",
        "kapitan tiago": "Kapitan Tiago",
        "sinang": "Sinang"
    }
    aligner = RobustAligner(tauhan_map=tauhan_map)
    
    # Text snippet with Kahoy and Punongkahoy
    summary = "Gusto kong maghanap ng kahoy para sa panggatong."
    passage = "Sila ay nagtago sa ilalim ng napakalaking punongkahoy. Ang mga ibon ay umaawit. Hindi nila alam ang paparating na panganib."
    
    # Let's run full alignment trick using debug pass
    # (Since debug_tauhan was written just for tauhan, we can modify it to output Lexical scores)
    buod_sentences = [summary]
    full_sentences = [
        "Sila ay nagtago sa ilalim ng napakalaking punongkahoy.",
        "Ang mga ibon ay umaawit.",
        "Hindi nila alam ang paparating na panganib."
    ]
    
    # Mocking semantic embeddings so align doesn't crash on length mismatch
    import numpy as np
    b_embs = np.random.rand(1, 384)
    f_embs = np.random.rand(3, 384)
    
    blocks = aligner.align(buod_sentences, full_sentences, b_embs, f_embs)
    print("Alignment matched words:")
    for b in blocks:
        print(f"Summary: {summary}")
        print(f"Passage Match Indices: [{b.best_window_start}:{b.best_window_end}]")
        print(f"Lexical Score: {b.lexical_score}")


if __name__ == "__main__":
    debug_tauhan()
