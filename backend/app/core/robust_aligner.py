import numpy as np
import re
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass

@dataclass
class RobustAlignedBlock:
    buod_index: int
    buod_text: str
    best_window_start: int
    best_window_end: int
    best_center_sentence: int
    lexical_score: float
    semantic_score: float
    position_score: float
    tauhan_score: float
    final_score: float
    matched_characters: List[str] = None

class RobustAligner:
    """
    Implements a 9-step Buod-to-Full alignment system for Filipino literary texts.
    
    1. Window Generation (3-6 sentences)
    2. Lexical Scoring (TF-IDF + character n-grams)
    3. Semantic Scoring (Embeddings)
    4. Tauhan Scoring (Provided list)
    5. Position Scoring (Relative position)
    6. Final Scoring (Weighted combination)
    7. Best Window Selection
    8. Center Sentence Identification
    9. Global Alignment (Dynamic Programming)
    """
    
    def __init__(
        self,
        tauhan_list: List[str],
        weights: Dict[str, float] = None
    ):
        self.tauhan_list = [t.lower() for t in tauhan_list]
        self.weights = weights or {
            "lexical": 0.40,
            "semantic": 0.40,
            "position": 0.10,
            "tauhan": 0.10
        }
        
    def align(
        self,
        buod_sentences: List[str],
        full_sentences: List[str],
        buod_embeddings: np.ndarray,
        full_embeddings: np.ndarray,
        return_debug: bool = False
    ) -> Any:
        """
        Main entry point for alignment.
        """
        num_buod = len(buod_sentences)
        num_full = len(full_sentences)
        
        if num_buod == 0 or num_full == 0:
            return [] if not return_debug else ([], {})

        # STEP 1: WINDOW GENERATION
        windows = []
        for start in range(num_full):
            for size in range(3, 7): # 3 to 6
                end = start + size
                if end <= num_full:
                    windows.append({
                        "start": start,
                        "end": end - 1,
                        "center": (start + end - 1) // 2,
                        "text": " ".join(full_sentences[start:end]),
                        "embeddings": full_embeddings[start:end]
                    })
        
        num_windows = len(windows)
        
        # STEP 2: LEXICAL SCORING (TF-IDF + char n-grams)
        # Combine all buod and window texts for TF-IDF context
        all_texts = buod_sentences + [w["text"] for w in windows]
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(3, 5),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        buod_tfidf = tfidf_matrix[:num_buod]
        window_tfidf = tfidf_matrix[num_buod:]
        
        # Precompute lexical scores: cosine similarity between buod and windows
        from sklearn.metrics.pairwise import cosine_similarity
        lexical_scores = cosine_similarity(buod_tfidf, window_tfidf) # (num_buod, num_windows)
        
        # STEP 3: SEMANTIC SCORING
        # Average window embeddings for comparison
        window_avg_embs = np.array([np.mean(w["embeddings"], axis=0) for w in windows])
        # Normalize embeddings for cosine similarity
        buod_embs_norm = buod_embeddings / (np.linalg.norm(buod_embeddings, axis=1, keepdims=True) + 1e-9)
        window_embs_norm = window_avg_embs / (np.linalg.norm(window_avg_embs, axis=1, keepdims=True) + 1e-9)
        semantic_scores = np.dot(buod_embs_norm, window_embs_norm.T) # (num_buod, num_windows)
        
        # STEP 4: TAUHAN SCORING
        tauhan_scores = np.zeros((num_buod, num_windows))
        buod_tauhan_mentions = [self._extract_tauhan(s) for s in buod_sentences]
        window_tauhan_mentions = [self._extract_tauhan(w["text"]) for w in windows]
        
        for i in range(num_buod):
            b_tauhan = buod_tauhan_mentions[i]
            for j in range(num_windows):
                w_tauhan = window_tauhan_mentions[j]
                
                if not b_tauhan:
                    # If buod_tauhan empty -> tauhan_score = 0 (as per requested edge case)
                    tauhan_scores[i, j] = 0.0
                elif b_tauhan.issubset(w_tauhan):
                    # Strict subset rule: if complete, score = |buod_tauhan| / |full_tauhan|
                    tauhan_scores[i, j] = len(b_tauhan) / len(w_tauhan)
                else:
                    # Missing any required tauhan = hard fail (0)
                    tauhan_scores[i, j] = 0.0
                    
        # STEP 5: POSITION SCORING
        position_scores = np.zeros((num_buod, num_windows))
        for i in range(num_buod):
            buod_rel_pos = i / max(1, num_buod - 1)
            for j in range(num_windows):
                window_rel_pos = windows[j]["center"] / max(1, num_full - 1)
                position_scores[i, j] = 1.0 - abs(buod_rel_pos - window_rel_pos)
                
        # STEP 6: FINAL SCORING
        final_scores = (
            self.weights["lexical"] * lexical_scores +
            self.weights["semantic"] * semantic_scores +
            self.weights["position"] * position_scores +
            self.weights["tauhan"] * tauhan_scores
        )
        
        # STEP 9: GLOBAL ALIGNMENT (SEQUENTIAL CONSISTENCY)
        # dp[i, j] = max cumulative score for first i buod sentences ending at window j
        # enforce j_i >= j_{i-1}
        dp = np.full((num_buod, num_windows), -np.inf)
        parent = np.full((num_buod, num_windows), -1, dtype=int)
        
        # Initialize first buod
        dp[0, :] = final_scores[0, :]
        
        # DP transitions
        # Large forward jump penalty: we want windows to be relatively close in sequence
        # Penalty is proportional to the gap between window centers
        for i in range(1, num_buod):
            for j in range(num_windows):
                # We can come from any k where window k starts at or before window j
                # To maintain story order, we use the constraint: window_start[j] >= window_start[k]
                # More strictly, window_center[j] >= window_center[k]
                
                # Optimized search: only check k where windows[k]["start"] <= windows[j]["start"]
                # and windows[k]["start"] is reasonably close to minimize jumps
                current_score = final_scores[i, j]
                
                # Find best previous state k
                # For efficiency, we can pre-calculate the running max of dp[i-1, :k]
                # but let's keep it simple for now and optimize if needed.
                prev_best_k = -1
                prev_best_score = -np.inf
                
                # Extract indices of windows that start at or before current window start
                # Since windows are generated in order of 'start', we can check all k <= j
                for k in range(j + 1):
                    # Penalty for large forward jumps
                    # gap = windows[j]["start"] - windows[k]["end"]
                    # jump_penalty = 0.05 * max(0, gap) / num_full
                    
                    # We want to encourage sequentiality
                    # Using a simple monotonicity constraint first
                    score_with_prev = dp[i-1, k] + current_score
                    if score_with_prev > prev_best_score:
                        prev_best_score = score_with_prev
                        prev_best_k = k
                
                if prev_best_k != -1:
                    dp[i, j] = prev_best_score
                    parent[i, j] = prev_best_k

        # Traceback to find the best global path
        best_end_j = np.argmax(dp[num_buod - 1, :])
        path = []
        curr_j = best_end_j
        for i in range(num_buod - 1, -1, -1):
            path.append(curr_j)
            curr_j = parent[i, curr_j]
        path.reverse()
        
        # STEP 7 & 8: SELECT BEST WINDOW & FIND CENTER SENTENCE
        aligned_results = []
        for i, window_idx in enumerate(path):
            best_window = windows[window_idx]
            
            # Find center sentence within the best window (Step 8)
            window_sentences = full_sentences[best_window["start"] : best_window["end"] + 1]
            window_sentence_embs = best_window["embeddings"]
            
            # Normalize for comparison
            b_emb_norm = buod_embeddings[i] / (np.linalg.norm(buod_embeddings[i]) + 1e-9)
            s_embs_norm = window_sentence_embs / (np.linalg.norm(window_sentence_embs, axis=1, keepdims=True) + 1e-9)
            
            sent_sims = np.dot(s_embs_norm, b_emb_norm)
            best_local_idx = np.argmax(sent_sims)
            best_center_abs_idx = best_window["start"] + best_local_idx
            
            aligned_results.append(RobustAlignedBlock(
                buod_index=i,
                buod_text=buod_sentences[i],
                best_window_start=best_window["start"],
                best_window_end=best_window["end"],
                best_center_sentence=int(best_center_abs_idx),
                lexical_score=float(lexical_scores[i, window_idx]),
                semantic_score=float(semantic_scores[i, window_idx]),
                position_score=float(position_scores[i, window_idx]),
                tauhan_score=float(tauhan_scores[i, window_idx]),
                final_score=float(final_scores[i, window_idx]),
                matched_characters=sorted(list(buod_tauhan_mentions[i] & window_tauhan_mentions[window_idx]))
            ))
            
        if return_debug:
            return aligned_results, {
                "lexical_scores": lexical_scores,
                "semantic_scores": semantic_scores,
                "tauhan_scores": tauhan_scores,
                "position_scores": position_scores,
                "final_scores": final_scores,
                "windows": windows,
                "buod_tauhan_mentions": buod_tauhan_mentions,
                "window_tauhan_mentions": window_tauhan_mentions,
                "dp_path": path
            }
            
        return aligned_results


    def _extract_tauhan(self, text: str) -> set:
        """
        Extracts tauhan mentions from the provided list.
        Exact string match (case-insensitive).
        """
        found = set()
        text_lower = text.lower()
        for tauhan in self.tauhan_list:
            # Use regex to ensure word boundary
            pattern = r'\b' + re.escape(tauhan) + r'\b'
            if re.search(pattern, text_lower):
                found.add(tauhan)
        return found
