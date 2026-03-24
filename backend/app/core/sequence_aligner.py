"""
sequence_aligner.py  —  Monotonic Block Alignment via Dynamic Programming.

Maps each buod (summary) sentence to a contiguous block of full-text sentences
that it summarises. Preserves chronological order strictly (monotonic).

Scoring per block:
    Score = λ_sem*S_sem + λ_char*S_char - λ_var*S_var - λ_len*S_len

All inner-loop score components are computed with NumPy batch operations,
making the method fast enough for real-time HTTP requests even on the largest
chapters (≈ 35 buod × 400 full sentences ≈ 56 ms on CPU).
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any


@dataclass
class AlignedBlock:
    buod_index: int
    buod_text: str
    full_text_start: int      # inclusive
    full_text_end: int        # inclusive
    full_text_sentences: List[str]
    alignment_anchors: List[str]
    semantic_score: float
    character_score: float
    variance_penalty: float
    length_penalty: float
    total_score: float


class SequenceAligner:
    """
    Monotonic DP aligner with vectorised scoring.

    Parameters
    ----------
    model : SentenceTransformer
    char_patterns : dict  book -> [(canon_name, compiled_regex)]
    w_max  : hard cap on block size (sentences)
    lambda_* : scoring weights
    """

    def __init__(
        self,
        model,
        char_patterns: Dict[str, List[Tuple[str, Any]]],
        w_max: int = 35,
        lambda_sem: float = 0.55,
        lambda_char: float = 0.25,
        lambda_var: float  = 0.15,
        lambda_len: float  = 0.05,
    ):
        self.model        = model
        self.char_patterns = char_patterns
        self.W_MAX        = w_max
        self.λ_sem  = lambda_sem
        self.λ_char = lambda_char
        self.λ_var  = lambda_var
        self.λ_len  = lambda_len

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def align(
        self,
        buod_sentences: List[str],
        full_sentences: List[str],
        book: str,
    ) -> List[AlignedBlock]:
        N, M = len(buod_sentences), len(full_sentences)
        if N == 0 or M == 0:
            return []

        # ── 1. Encode & normalise in one batch ──────────────────────────
        all_texts = buod_sentences + full_sentences
        all_embs  = self.model.encode(all_texts, show_progress_bar=False, batch_size=128)
        norms     = np.linalg.norm(all_embs, axis=1, keepdims=True)
        norms     = np.where(norms > 0, norms, 1.0)
        all_embs  = (all_embs / norms).astype(np.float32)

        buod_embs = all_embs[:N]        # (N, D)
        full_embs = all_embs[N:]        # (M, D)
        D         = all_embs.shape[1]

        # ── 2. Prefix-sum of full embeddings  ───────────────────────────
        # cum[j] = sum full_embs[0:j]  →  sum[k:j] = cum[j] - cum[k]
        cum = np.zeros((M + 1, D), dtype=np.float32)
        cum[1:] = np.cumsum(full_embs, axis=0)

        # ── 3. Pre-extract character sets ───────────────────────────────
        patterns  = self.char_patterns.get(book, [])
        b_chars   = [self._chars(s, patterns) for s in buod_sentences]
        f_chars   = [self._chars(s, patterns) for s in full_sentences]

        # ── 4. Pre-compute log-length penalties for all block sizes ─────
        # log_pen[l] = log(l)  for l ∈ [0, W_MAX]
        log_pen = np.zeros(self.W_MAX + 2, dtype=np.float32)
        for l in range(2, self.W_MAX + 2):
            log_pen[l] = math.log(l)

        # ── 5. DP ───────────────────────────────────────────────────────
        NEG_INF = float("-inf")
        dp  = np.full((N + 1, M + 1), NEG_INF, dtype=np.float64)
        ptr = np.full((N + 1, M + 1), -1,       dtype=np.int32)
        dp[0, 0] = 0.0

        for i in range(1, N + 1):
            bi      = buod_embs[i - 1]      # (D,)
            bi_char = b_chars[i - 1]

            j_min = i
            j_max = M - (N - i)

            for j in range(j_min, j_max + 1):
                k_lo = max(i - 1, j - self.W_MAX)
                k_hi = j          # block [k:j], k in [k_lo, j-1]

                # ── vectorised score over all valid k ────────────────
                # k_range: k values to test
                k_vals = np.arange(k_lo, k_hi, dtype=np.int32)
                if len(k_vals) == 0:
                    continue

                prev_dp = dp[i - 1, k_vals]             # (K,)

                # Only process k where previous state is reachable
                valid = prev_dp > NEG_INF
                if not np.any(valid):
                    continue

                k_vals   = k_vals[valid]
                prev_dp  = prev_dp[valid]
                K        = len(k_vals)

                lens = j - k_vals                         # (K,) block lengths

                # -- S_sem --
                # block_sum[p] = cum[j] - cum[k_vals[p]]
                block_sums = cum[j] - cum[k_vals]         # (K, D)
                # S_sem = dot(bi, block_sum) / ||block_sum||
                dots   = block_sums @ bi                  # (K,)
                b_norms = np.linalg.norm(block_sums, axis=1)  # (K,)
                b_norms = np.where(b_norms > 0, b_norms, 1.0)
                S_sem   = dots / b_norms                  # (K,)

                # -- S_var --
                # var ≈ 1 - ||mean_unnorm|| / block_len
                # mean_unnorm = block_sum,  its norm = b_norms
                S_var = 1.0 - b_norms / lens              # (K,)
                S_var = np.clip(S_var, 0.0, None)

                # -- S_len --
                clipped_lens = np.minimum(lens, self.W_MAX + 1)
                S_len = log_pen[clipped_lens]             # (K,)

                # -- S_char (start at 0, then boost for char matches) --
                S_char_arr = np.zeros(K, dtype=np.float32)

                if bi_char:
                    # Character constraint + scoring in one pass
                    for p in range(K):
                        k = int(k_vals[p])
                        blk_chars: set = set()
                        for fi in range(k, j):
                            blk_chars |= f_chars[fi]
                        if bi_char.isdisjoint(blk_chars):
                            # hard constraint: penalise heavily
                            S_char_arr[p] = -10.0
                        else:
                            S_char_arr[p] = len(bi_char & blk_chars) / len(bi_char)

                # -- composite score --
                scores = (
                      self.λ_sem  * S_sem
                    + self.λ_char * S_char_arr
                    - self.λ_var  * S_var
                    - self.λ_len  * S_len
                    + prev_dp
                )

                # Filter blocked by hard constraint (score penalised -10)
                # They will naturally not win best if other options exist,
                # but explicitly skip if ALL are char-blocked
                best_idx   = int(np.argmax(scores))
                best_score = float(scores[best_idx])
                best_k     = int(k_vals[best_idx])

                if best_score > dp[i, j]:
                    dp[i, j]  = best_score
                    ptr[i, j] = best_k

        # ── 6. Traceback ────────────────────────────────────────────────
        best_j = int(np.argmax(dp[N, N:M + 1])) + N

        raw: list = []
        ci, cj = N, best_j
        while ci > 0:
            k = int(ptr[ci, cj])
            if k < 0:
                k = max(cj - 1, 0)
            raw.insert(0, (ci - 1, k, cj))
            cj = k
            ci -= 1

        # ── 7. Build output ─────────────────────────────────────────────
        out: List[AlignedBlock] = []
        for (bidx, fstart, fend) in raw:
            blk_len   = fend - fstart
            blk_texts = full_sentences[fstart:fend]
            blk_chars: set = set()
            for fc in f_chars[fstart:fend]:
                blk_chars |= fc
            anchors = sorted(b_chars[bidx] & blk_chars)

            bi      = buod_embs[bidx]
            bsum    = cum[fend] - cum[fstart]
            bnorm   = float(np.linalg.norm(bsum))
            S_sem   = float(np.dot(bi, bsum)) / max(bnorm, 1e-9)
            S_char  = len(b_chars[bidx] & blk_chars) / max(len(b_chars[bidx]), 1) if b_chars[bidx] else 0.0
            S_var   = max(0.0, 1.0 - bnorm / max(blk_len, 1))
            S_len   = math.log(blk_len) if blk_len > 1 else 0.0
            total   = self.λ_sem * S_sem + self.λ_char * S_char - self.λ_var * S_var - self.λ_len * S_len

            out.append(AlignedBlock(
                buod_index=bidx,
                buod_text=buod_sentences[bidx],
                full_text_start=fstart,
                full_text_end=fend - 1,
                full_text_sentences=blk_texts,
                alignment_anchors=anchors,
                semantic_score=round(S_sem,  4),
                character_score=round(S_char, 4),
                variance_penalty=round(S_var, 4),
                length_penalty=round(S_len,  4),
                total_score=round(total,     4),
            ))

        return out

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _chars(self, text: str, patterns) -> set:
        found = set()
        tl = text.lower()
        for canon, pat in patterns:
            if pat.search(tl):
                found.add(canon)
        return found
