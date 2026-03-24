# PHASE 1: EVIDENCE AUDIT REPORT

## 1. Methodology Evidence (Recoverable Logistics & Metrics)
Through a deep dive into the repository backend, profound logic is explicitly recoverable. The methodology is highly structured rather than purely abstract NLP. 

### Files Examined:
- `backend/app/core/engine.py` (Core Pipeline Logic)
- `backend/app/core/analyzer.py` (Query Decomposition & Weighing)
- `backend/app/services/suggestions.py` (Suggestion Archetypes)

### Evidence Present & Strong Enough for Thesis:
**A. Lexical Scoring & Penalty Logic (`analyzer.py` & `engine.py`)**
Lexical scoring behaves as a mathematical equation heavily weighting term frequency and density:
*   `semantic_weight` generation: Words in TL vocabulary with frequency `> 0.001` get weight **0.3**, `> 0.0001` get **0.7**, else **1.0**. Stopwords strictly get **0.0**.
*   Lexical Score Equation: `score = (coverage * 0.8) + (density * 0.2)`
*   High Stopword Penalty: Triggers if ratio `> 0.6`. Multiplier: `score *= (1.0 - ((stopword_ratio - 0.6) * 0.5))`
*   Exact-Phrase Bonus: `min(1.0, 0.90 + (len(query_lower) / max(1, len(sentence_lower))))`

**B. Semantic Thresholds & Fallback Logic (`engine.py`)**
*   **In-Domain Gate**: Theme matrix projection uses a cutoff threshold of **0.50** for Out-Of-Vocabulary queries, and **0.40** for standard queries.
*   **Dynamic Validation**: Strict single-word synonym threshold requires geometric proximity `> 0.55` to pass the Stage B verification.

**C. Hybrid Reranking & Concept Coverage**
*   **Dynamic Weights Allocation**: Determined by text length. 
    *   Length `<= 8`: Lexical 0.90, Semantic 0.10
    *   Length `<= 15`: Lexical 0.70, Semantic 0.30
    *   Length `<= 25`: Lexical 0.50, Semantic 0.50
    *   Length `> 25`: Lexical 0.30, Semantic 0.70
    *(Weights are boosted to `.80` or `.60` lexical max if the query is aggressively short).*
*   **Coverage Penalty**: If a multi-word query has a coverage ratio `< 1.0` (missing semantic/lexical components), the semantic score drops by `(1.0 - penalty * 0.95)`. If there are *zero* lexical matches for a core component, penalties wipe out scores (`sem_score *= 0.1, lex_score *= 0.05`).
*   **Short Sentence Penalty**: `penalty = 0.25 * (10 - length) / 10`.

**D. Query-Class Suggestion Logic (`suggestions.py`)**
*   Explicit 60-key mapping matching input directly to categorized tags (Entities, Broad Themes). Rejects standard abstract questions (e.g., "bakit", "paano") outright instead of using LLM generative prediction.

---

## 2. Evaluation Evidence
### Files & Logs Examined:
- `verification.txt` / `coverage_results.txt` / `profile_results.txt`
- `check_backend.py` / `profile_search.py`

### Evidence Present & Strong Enough for Thesis:
*   **Performance Profiling**: `profile_output.prof` shows search execution taking heavily computationally intensive times (e.g., 127 seconds on heavy loading loops), proving a distinct architectural need for caching (Redis).
*   **Verification Scripts**: Results from `verification.txt` map behavior explicitly rejecting terms (`Testing query: 'gago ka sisa' -> REJECTED`). This serves as brilliant qualitative case-study evidence of the "In-domain Gate" in action.
*   **API Tests**: `check_backend.py` proves cross-novel routing constraints.

### Areas Lacking Evidence:
*   **Direct Before/After JSONs**: The repository lacks explicitly labeled "before-fix.json" vs "after-fix.json" files. The "Before/After" analysis will have to be reconstructed theoretically based on the current safeguards (e.g., describing why `coverage_ratio` was implemented rather than citing empirical logs of its failure).

---

## 3. Quantitative Metrics Evidence
**DO WE HAVE PRECISION, RECALL, F1, OR ACCURACY MATRICS?**
**No.** A repository-wide `grep` search for these terms confirms they *only* exist in the literature review texts (`chapter1.md`, `chapter2.md`) and the old draft (`chapter3_old.md`). 

**Actual Metrics Discovered:**
*   **Latency/Time Execution**: Found via cProfile logs (`profile_results.txt`).
*   **Threshold Constants**: Highly explicit logic floats determining acceptance/rejection (Coverage Ratio `1.0`, Synonym Threshold `0.55`).

We CANNOT claim F1-Score, Precision, or Recall evaluation in Chapters 3 or 4. The evaluation methodology must be explicitly termed **Functional Verification & Threshold Calibration** backed by edge-case query trapping.

---

## 4. UI / Implementation Evidence
### Files Examined:
- `frontend/components/ResultCard.tsx`
- `frontend/components/ScoreVisualizer.tsx`

### Evidence Present & Strong Enough for Thesis:
*   **Score Visualization**: Visual bars explicitly map out the backend metrics (e.g., `semantic={scores.semantic} lexical={scores.lexical}`) displaying them out of 100%. Lexical represented by Amber, Semantic by Teal.
*   **Explainability Badges**: Shows direct rendering of `Mataas` confidence badges based on boolean backend flags, fulfilling the requirement of algorithmic transparency for educational tools.
*   **Component Expanding**: The "Ipakita ang Konteksto" cleanly triggers CSS Grid/Motion animations to reveal the `contextHtml` fetched directly from `engine.py`'s ±8 neighbor expansion logic.
*   **Grouping**: Distinguishes `border-noli-gold` from `border-fili-magenta` reinforcing the comparative architecture.
