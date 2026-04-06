import numpy as np
import re
from typing import List, Dict, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from .tagalog_stemmer import TagalogStemmer

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
        tauhan_map: Dict[str, str], # Map of Alias -> Canonical Name
        weights: Dict[str, float] = None
    ):
        self.tauhan_map = {k.lower(): v for k, v in tauhan_map.items()}
        # Also include canonical names as their own aliases if missing
        for v in tauhan_map.values():
            if v.lower() not in self.tauhan_map:
                self.tauhan_map[v.lower()] = v
        
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
        full_is_short: List[bool] = None,
        return_debug: bool = False
    ) -> Any:
        """
        Main entry point for alignment.
        """
        num_buod = len(buod_sentences)
        num_full = len(full_sentences)
        
        if num_buod == 0 or num_full == 0:
            return [] if not return_debug else ([], {})

        # If flags not provided, assume all are valid (non-short)
        if full_is_short is None:
            full_is_short = [False] * num_full

        # STEP 1: WINDOW GENERATION (3-6 valid sentences)
        windows = []
        # Find indices of all valid (non-short) sentences
        valid_indices = [i for i, is_short in enumerate(full_is_short) if not is_short]
        num_valid = len(valid_indices)

        if num_valid < 3:
            # Fallback: if there are fewer than 3 valid sentences in the whole chapter, 
            # just use the standard 3-6 sentence windowing (or whatever is available)
            for start in range(num_full):
                for size in range(3, 7):
                    end = start + size
                    if end <= num_full:
                        windows.append({
                            "start": start,
                            "end": end - 1,
                            "center": (start + end - 1) // 2,
                            "text": " ".join(full_sentences[start:end]),
                            "embeddings": full_embeddings[start:end]
                        })
        else:
            # Revised Window Logic: For each start position, find end positions that hit 3-6 valid sentences
            for start in range(num_full):
                v_count = 0
                for end in range(start, num_full):
                    if not full_is_short[end]:
                        v_count += 1
                    
                    if 3 <= v_count <= 6:
                        # We found a window with 3-6 valid sentences.
                        # Only add if 'end' is a valid sentence frontier OR last sentence.
                        if not full_is_short[end] or end == num_full - 1:
                            windows.append({
                                "start": start,
                                "end": end,
                                "center": (start + end) // 2,
                                "text": " ".join(full_sentences[start : end + 1]),
                                "embeddings": full_embeddings[start : end + 1]
                            })
                    elif v_count > 6:
                        break
        num_windows = len(windows)
        if num_windows == 0:
            return []
        
        # STEP 2: LEXICAL SCORING (Dynamic Layered Subword Matching)
        stemmer = TagalogStemmer()
        
        stopwords = {
            'ang', 'ng', 'sa', 'mga', 'at', 'ay', 'na', 'ni', 'rin', 'din', 
            'si', 'sila', 'kami', 'tayo', 'kayo', 'kanya', 'nila', 'namin',
            'ito', 'iyan', 'iyon', 'dito', 'diyan', 'doon', 'kung', 'para',
            'upang', 'dahil', 'habang', 'bagamat', 'kaysa', 'nang', 'noon',
            'mula', 'hanggang', 'pa', 'lamang', 'lang', 'ba', 'po', 'ho',
            'naman', 'o', 'pati', 'maging', 'kaya', 'tulad', 'gaya', 'ako', 'ikaw',
            'siya', 'niya', 'ko', 'mo', 'ata', 'raw', 'daw', 'yata', 'nga',
            'hindi', 'di', 'wala', 'may', 'mayroon', 'walang', 'isang', 'niyang', 'siyang'
        }
        
        def get_tokens(text):
            import string
            clean_text = text.lower().translate(str.maketrans('', '', string.punctuation))
            return [w for w in clean_text.split() if w not in stopwords and len(w) > 2]
            
        buod_tokens_list = [get_tokens(s) for s in buod_sentences]
        window_tokens_list = [get_tokens(w["text"]) for w in windows]
        
        lexical_scores = np.zeros((num_buod, num_windows))
        
        for i, b_tokens in enumerate(buod_tokens_list):
            if not b_tokens:
                lexical_scores[i, :] = 0.0
                continue
                
            for j, w_tokens in enumerate(window_tokens_list):
                if not w_tokens:
                    continue
                
                match_count = 0
                for b_word in b_tokens:
                    b_stem = stemmer.stem(b_word)
                    matched = False
                    
                    for w_word in w_tokens:
                        w_stem = stemmer.stem(w_word)
                        
                        # 1. Canonical Context Equality
                        if b_stem == w_stem:
                            matched = True
                            break
                            
                        # 2. Partial Canonical (Subword overlap on stems)
                        if len(b_stem) >= 4 and len(w_stem) >= 4:
                            if b_stem in w_stem or w_stem in b_stem:
                                matched = True
                                break
                                
                        # 3. Subword Sliding Check (Compound resolution)
                        if len(b_word) >= 5 and len(w_word) >= 5:
                            if b_word in w_word or w_word in b_word:
                                matched = True
                                break
                                
                    if matched:
                        match_count += 1
                        
                lexical_scores[i, j] = match_count / len(b_tokens)
        
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
                    # If buod has no characters, characters aren't a matching signal here
                    tauhan_scores[i, j] = -1.0 # N/A match
                else:
                    # Overlap Ratio: (Characters in summary found in window) / (Total Unique Characters in Summary)
                    matched = b_tauhan.intersection(w_tauhan)
                    tauhan_scores[i, j] = len(matched) / len(b_tauhan)
                    
        # STEP 5: POSITION SCORING
        position_scores = np.zeros((num_buod, num_windows))
        for i in range(num_buod):
            buod_rel_pos = i / max(1, num_buod - 1)
            for j in range(num_windows):
                window_rel_pos = windows[j]["center"] / max(1, num_full - 1)
                position_scores[i, j] = 1.0 - abs(buod_rel_pos - window_rel_pos)
                
        # STEP 6: FINAL SCORING
        final_scores = np.zeros((num_buod, num_windows))
        for i in range(num_buod):
            if tauhan_scores[i, 0] == -1.0:
                # No characters in this buod sentence - ignore the tauhan weight
                # Calculate normalized weights for the remaining signals
                remaining_weight = self.weights["lexical"] + self.weights["semantic"] + self.weights["position"]
                w_lex = self.weights["lexical"] / remaining_weight
                w_sem = self.weights["semantic"] / remaining_weight
                w_pos = self.weights["position"] / remaining_weight
                
                final_scores[i, :] = (
                    w_lex * lexical_scores[i, :] +
                    w_sem * semantic_scores[i, :] +
                    w_pos * position_scores[i, :]
                )
            else:
                # Normal weighted combination
                final_scores[i, :] = (
                    self.weights["lexical"] * lexical_scores[i, :] +
                    self.weights["semantic"] * semantic_scores[i, :] +
                    self.weights["position"] * position_scores[i, :] +
                    self.weights["tauhan"] * tauhan_scores[i, :]
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
        Supports Tagalog suffixes: -ng, -y, -n.
        """
        found = set()
        text_lower = text.lower()
        
        for alias, canon in self.tauhan_map.items():
            # Support multi-word names with potential Tagalog suffixes after EACH word
            # e.g., "Kapitang Tiagong" matches alias "Kapitan Tiago"
            parts = alias.split()
            if not parts: continue
            
            # Pattern: \bPart1(?:ng|y|n)?\s+Part2(?:ng|y|n)?\b
            pattern = r'\b' + r'\s+'.join([re.escape(p) + r'(?:ng|y|n)?' for p in parts]) + r'\b'
            
            if re.search(pattern, text_lower, re.IGNORECASE):
                found.add(canon) # Aggregate by Canonical Person Name
                
        return found
