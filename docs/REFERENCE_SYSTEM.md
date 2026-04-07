# Buod-to-Full Alignment System (Sanggunian)

This document provides a detailed technical explanation of the **Sanggunian (Reference) Alignment System** implemented in the Rizal Thematic Exploration project. This system is responsible for mapping summary (buod) sentences to their corresponding passages in the original full-text novels (*Noli Me Tangere* and *El Filibusterismo*).

## System Objective
The primary goal is to provide a precise, high-confidence mapping that allows users to instantly verify "buod" claims against the original source text. This requires bridging the gap between simplified modern Tagalog summaries and 19th-century literary prose.

## Core Component: RobustAligner
The alignment is powered by the `RobustAligner` class, which implements a 9-step scoring and optimization process.

---

### Step 1: Multi-Scale Window Generation
Instead of a point-to-point comparison, the system scans the full text using sliding windows.
- **Dynamic Sizing:** Generates overlapping windows of 3, 4, 5, and 6 sentences.
- **Context Preservation:** Since a single summary sentence often describes a scene that spans multiple original sentences, windowing ensures the "event context" is captured.

### Step 2: Lexical Scoring (TF-IDF + Character N-grams)
Identifies literal string similarity and morphological overlaps.
- **Morphological Handling:** Uses 3-5 character n-grams to handle Tagalog's complex affixation system (e.g., matching *kumain*, *kinakain*, and *mangangain* to the root *kain*).
- **TF-IDF Weighting:** Gives higher priority to unique chapter keywords (like specific locations or rare objects) and lower priority to common particles (*ng*, *mg*, *ay*).

### Step 3: Semantic Vector Comparison (XLM-R)
Uses a deep learning model to match the "essence" or "meaning" of sentences.
- **XLM-RoBERTa:** Utilizes a Cross-Lingual Language Model fine-tuned on the Rizal corpus (Domain Adaptive Pre-training).
- **Cosine Similarity:** Measures the angular distance between the 768-dimensional summary vector and the average window vector. This handles cases where the summary uses synonyms or different wording than the original.

### Step 4: Strict Tauhan (Character) Scoring
Enforces thematic consistency through a specialized "Character Gate."
- **Subset Rule:** A full-text window only qualifies if it contains **all** characters explicitly named in the summary sentence: $buod\_tauhan \subseteq full\_tauhan$.
- **Formula:** 
  $$\text{Tauhan Score} = \frac{|buod\_tauhan|}{|full\_tauhan|}$$
- **Result:** If a character is missing, the score is **0**, acting as a hard fail for that window. Extra characters in the full text result in a soft penalty, rewarding exact character matches.

### Step 5: Relative Positional Scoring
Encourages chronological consistency across the chapter.
- **Relative Position:** Calculated as `index / total_sentences`.
- **Constraint:** $1.0 - |Position_{buod} - Position_{full}|$.
- **Rationale:** Most summaries follow the book's sequence linearly; this score penalizes "out-of-order" matches.

### Step 6: Weighted Component Consolidation
The final score for each window is calculated as a weighted sum of the components:
- **Lexical Score:** 40%
- **Semantic Score:** 40%
- **Tauhan Score:** 10%
- **Position Score:** 10%

### Step 7: Global Path Optimization (Dynamic Programming)
To ensure the entire chapter is aligned correctly as a single story, the system uses a Global Alignment algorithm.
- **Sequential Flow:** It finds the path through the chapter that maximizes the total cumulative score while enforcing **monotonicity** (matches cannot jump backwards in time).
- **Technique:** Uses a DP (Dynamic Programming) matrix to track the best possible sequence of windows for the entire summary.

### Step 8: Local Center Sentence Identification
Once the best window (scene) is identified, the system pinpoints the single most relevant sentence within it.
- **Micro-Matching:** Compares the summary sentence against each sentence in the window using semantic similarity.
- **Result:** This identifies the "Anchor Sentence" that is highlighted when the user clicks the Sanggunian button.

### Step 9: Result Serialization
The backend (FastAPI) packages the alignment results with diagnostic metadata:
- **Status Tags:** *Precise* (Score > 0.8), *High* (Score > 0.6), or *Medium*.
- **Evidence:** List of matched characters and the confidence breakdown.

---

## Technical Stack
- **Language:** Python
- **Models:** XLM-RoBERTa (Sentence-Transformers)
- **Data Handling:** NumPy, Scikit-learn, SQLAlchemy (PostgreSQL/pgvector)
- **API:** FastAPI
