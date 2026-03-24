# TECHNICAL SYSTEMS ANALYSIS REPORT
**Target:** Retrieval Engine & Hybrid Information Architecture
**Purpose:** Provide strict, implementation-level evidence for writing Chapter 3 (Methodology)

---

## A. Executive Summary
The Rizal Thematic Exploration System is an educational search platform designed to execute highly complex, Tagalog-specific thematic queries against the corpuses of *Noli Me Tangere* and *El Filibusterismo*. Its primary purpose is to retrieve narratively accurate, contextually expanded, and semantically verified literature passages that match student queries.

The system abandons traditional keyword-only search in favor of a **Staged Hybrid Retrieval Architecture**, combining exact-phrase Boolean retrieval with XLM-RoBERTa dense vector similarity. Its defining technical feature is its rigorous defense against "semantic hallucination"—enforcing strict synonym thresholds, modern-term blocklists, and dynamic stopword density penalties to ensure the AI retrieves literary context rather than out-of-domain noise. 

## B. System Architecture
**1. Client Layer (Frontend)**
- **Framework:** Next.js 16.1.6 (React 19), styled with TailwindCSS 4 and Framer Motion.
- **State & Data:** Uses Zustand for localized UI states (e.g., toggling between books) and `@tanstack/react-query` for server synchronization and client caching.
- **Explainability UI:** `ResultCard.tsx` and `ScoreVisualizer.tsx` render explicit semantic (teal) vs lexical (amber) score bars to demystify backend ranking weights.

**2. API & Caching Layer (Backend)**
- **Framework:** FastAPI serving routes for `/search`, `/themes`, and `/suggestions`.
- **Caching:** Redis `5.0.1` manages highly recurrent query caching and rate limits to minimize latency bottlenecks uncovered during profiling.

**3. Engine & Data Layer**
- **Core Engine:** `RizalEngine` (`app/core/engine.py`) orchestrates the Staged Retrieval, Component Decomposition, and Re-ranking logic.
- **Database:** PostgreSQL utilizing the `pgvector` extension for storing and executing cosine similarity commands against 768-dimensional XLM-RoBERTa embeddings via IVFFlat indexing.
- **NLP Utilities:** Uses `stopwords-iso-tl` for Tagalog stopwords, and `sentence-transformers` for embeddings.

## C. Retrieval Pipeline
The end-to-end data flow operates in distinct physical stages:

1. **Ingestion & Preprocessing:** Texts (*buod* summaries) are cleaned, deduplicated using TF-IDF similarity, and segmented into sentence-level chunks.
2. **Embedding Generation:** Sentences are encoded into 768-d vectors using a pre-trained XLM-RoBERTa base model. Wait, *note*: the system contains logic trying to load a Domain-Adaptive Pre-trained (DAPT) model, but falls back to the base model if missing. 
3. **Query Decomposition:** Multi-word queries are syntactically parsed (`TagalogRoleParser`) to extract isolated components (e.g., separating "edukasyon" and "Ibarra").
4. **Stage A (Lexical-First Retrieval):** The system queries PostgreSQL using `ILIKE` for exact intersections of the query's significant words. 
5. **Stage B (Semantic Fallback):** If Stage A returns less than `top_k * 2` (20) candidates, the pipeline executes a dense vector similarity search via `pgvector` against the XLM-RoBERTa embeddings, applying strict Out-Of-Vocabulary similarity thresholds.
6. **Re-ranking & Fusion:** Retrieved candidates are merged and scored combining lexical and semantic metrics. A **Lexical-Priority Reservation** mechanic guarantees passages containing actual keyword hits survive final truncation.
7. **Context Expansion:** The final top candidates pass through `_expand_context()`, fetching up to 3 neighboring sentences (prev/next) using a thresholded semantic-lexical overlap score.
8. **Output Generation:** The system returns structured JSON containing passage HTML, semantic/lexical math scores, and thematic suggestion strings mapping to 60 pre-classified entities.

## D. Computations and Formulas (MOST IMPORTANT)

### 1. Lexical Scoring Equation
*   **Purpose:** Determine exact/substring string matching while neutralizing stopwords. 
*   **Source:** `engine.py::_compute_lexical_score` & `analyzer.py::analyze_query_words`
*   **Variables:** `semantic_weight` (assigned via Tagalog word frequency: >0.001=0.3, >0.0001=0.7, else 1.0. Stopwords=0.0).
*   **Formula (Explicit):**
    $$ S_{lex} = (Coverage \times 0.8) + (Density \times 0.2) $$
    *(Where Coverage = Matched Weight / Total Query Weight; Density = Matches / Total Words in Sentence)*

### 2. High Stopword Penalty
*   **Purpose:** Punish queries consisting mostly of noise words.
*   **Source:** `engine.py::_compute_lexical_score`
*   **Variables:** `HIGH_STOPWORD_RATIO` (0.6), `STOPWORD_PENALTY_FACTOR` (0.5).
*   **Formula (Explicit):** 
    $$ S_{lex\_adjusted} = S_{lex} \times (1.0 - ((Ratio - 0.6) \times 0.5)) $$ 

### 3. Dynamic Hybrid Weighting
*   **Purpose:** Shift relevance between Lexical and Semantic scores based on text and query geometry.
*   **Source:** `engine.py::_compute_dynamic_weights`
*   **Calculations (Explicit Constants):** 
    - Text `<= 8` words: $\lambda_{lex}=0.90$, $\lambda_{sem}=0.10$
    - Text `<= 15` words: $\lambda_{lex}=0.70$, $\lambda_{sem}=0.30$
    - Text `<= 25` words: $\lambda_{lex}=0.50$, $\lambda_{sem}=0.50$
    - Text `> 25` words: $\lambda_{lex}=0.30$, $\lambda_{sem}=0.70$
    *(Note: If a query is $\le 2$ words, $\lambda_{lex}$ is clamped to a minimum of 0.80).*

### 4. Clear Score Equation (Fusion)
*   **Purpose:** Calculate the definitive matching score for multi-word queries.
*   **Source:** `engine.py::_calculate_clear_score`
*   **Formula (Explicit):**
    $$ Score_{raw} = (\lambda_{sem} \times S_{sem}) + (\lambda_{lex} \times S_{lex}) $$ 
    **Short Sentence Penalty:** If length `< 10` words:
    $$ Penalty = 0.25 \times \frac{10 - Length}{10} $$
    $$ Score_{final} = \max(0.0, \min(1.0, Score_{raw} - Penalty)) $$

### 5. Multi-Component Precision Score (Soft-AND Geometric Mean)
*   **Purpose:** Ensure complex queries (e.g., "Edukasyon ni Ibarra") penalize passages that only discuss "Ibarra" but not "Edukasyon."
*   **Source:** `engine.py::_calculate_precision_score`
*   **Formula (Explicit):** Calculates the Geometric Mean using natural logarithms.
    $$ Score_i = \max(Lex_{score} \times 0.8, Sem_{score}) $$
    $$ GMean = \exp\left( \frac{1}{N} \sum_{i} \log(Score_i + 10^{-9}) \right) $$

### 6. Coverage & Noise Penalty
*   **Purpose:** Drop purely semantic hallucinations that miss primary query components.
*   **Source:** `engine.py::search`
*   **Formula (Explicit):** 
    $$ Ratio_{cov} = \frac{\text{Concepts Matched}}{\text{Total Query Concepts}} $$
    If $Ratio_{cov} < 1.0$: Let $Penalty = 1.0 - Ratio_{cov}$.
    $$ S_{sem} = S_{sem} \times (1.0 - (Penalty \times 0.95)) $$
    $$ S_{lex} = S_{lex} \times (1.0 - (Penalty \times 0.98)) $$
    *Extreme Penalty:* If Lexical Matches = 0 for a core component: 
    $$ S_{sem} = S_{sem} \times 0.1, \quad S_{lex} = S_{lex} \times 0.05 $$

### 7. Neighbor Context Overlap Score
*   **Purpose:** Determine if neighboring sentences belong to the same narrative thought context as the central sentence.
*   **Source:** `engine.py::_compute_neighbor_score`
*   **Formula (Explicit):**
    $$ Score_{neighbor} = (0.6 \times CosineSimilarity) + (0.4 \times JaccardLexicalOverlap) $$

### 8. Vector Similarity
*   **Purpose:** Compare 768-d sentence embeddings.
*   **Formula (Explicit):** Executed as **Cosine Similarity** directly natively in PostgreSQL (`order_by(Sentence.embedding.cosine_distance(query_list))`), and implemented locally via Numpy's **Dot Product** `np.dot(vec1, vec2)` for normalized vectors.

## E. Retrieval Parameters
- **Chunk Size / Overlap:** Segmented naturally at the Sentence boundary level during text pre-processing. Context expansion fetches `MAX_CONTEXT_EXPANSION = 3` neighbors backwards and forwards dynamically.
- **Embedding Model:** XLM-RoBERTa Base (`sentence-transformers` library).
- **Vector Dimension:** 768.
- **Top-K Limit:** Primary queries fetch `(top_k * 20)` for Lexical hits, and `(top_k * 15)` for Semantic Fallback. The final UI pagination restricts to Top 5 Precise Matches.
- **Thresholds:**
    - Explicit `NEIGHBOR_RELEVANCE_THRESHOLD` = 0.40.
    - Extreme Synonym Validation Semantic Threshold = `0.55`.
    - Out-Of-Vocabulary (OOV) Domain Similarity Gate = `0.50` (OOV) / `0.40` (In-Vocabulary).
- **Hard Blocklist:** 18 Modern anachronistic terms (e.g., 'tiktok', 'internet', 'crypto') explicitly shut down the search.

## F. Data Structures and Storage
**Relational DB (PostgreSQL):**
- **`sentences` Table:** Contains `id`, `book`, `chapter_number`, `chapter_title`, `sentence_index` (integer to trace text succession), `sentence_text` (String), and `embedding` (`pgvector` column).
- **`themes` Table:** Contains foundational canonical 768-d vectors for Tagalog concepts spanning specific themes (`tagalog_title`, `meaning`, `embedding`).

## G. Algorithmic Flow

**Step 1: Parse & Validate.** Query enters the API. The engine runs a regex extraction to drop punctuation and checks the `MODERN_TERMS` blocklist. If caught, execution aborts.
**Step 2: Component Breakdown.** `TagalogRoleParser` separates characters from themes, and `QueryAnalyzer` calculates each term's frequency, dropping stopwords to 0.0 weight.
**Step 3: Base Embedding.** The query goes through the Sentence Transformer to generate a 768-d vector. 
**Step 4: Lexical First Pass (Stage A).** Exact matching strings trigger PostgreSQL `ILIKE` queries retrieving up to 200 candidates per book. 
**Step 5: Semantic Switch (Stage B).** If candidates $< 20$, the pipeline checks the In-Domain Gate minimum cosine-similarity against the global Theme Matrix. If passed, it performs an IVFFlat nearest-neighbor semantic search fetching 150 candidates.
**Step 6: Dynamic Metric Math.** Ranks all candidates. Sentence texts are tokenized, Stopword penalties injected, and Dynamic Weights computed (e.g. short text favors lexical, long text favors semantic).
**Step 7: Re-Ranking Precision.** The Geometric Mean Precision score evaluates multi-component queries, enforcing heavy penalizations (90%+ drops) if explicit concepts are wholly missing.
**Step 8: Lexical Reservation Check.** Sentences boasting actual literal hits bypass score truncations, forcefully pinned to the top.
**Step 9: Neighbor Assembly.** Top items execute `_expand_context()`, checking adjacent vector similarities (±3 idx) to pad the sentence naturally.
**Step 10: JSON Assembly.** Confidence Badges (Strong/Partial) and scores are packaged with `DynamicSuggestionGenerator` mapping the query to 60 explicit related-search tags, responding to the frontend.

## H. Evidence Map
- **Lexical Math Logic**: `backend/app/core/engine.py` -> `_compute_lexical_score()` 
- **Dynamic Weight Logic**: `backend/app/core/engine.py` -> `_compute_dynamic_weights()`
- **Precision/Component Logic**: `backend/app/core/engine.py` -> `_calculate_precision_score()`
- **Suggestion Mapping**: `backend/app/services/suggestions.py` -> `THEMATIC_SUGGESTIONS` dictionary.
- **Database Schema**: `backend/app/models/database.py` -> `class Sentence` & `class Theme`.

## I. Chapter 3 Writing Notes for Claude
- **Methodology Frame:** Describe the thesis purely as a *Functional Systems Architecture and Calibration Research*. The testing methodology relies strictly on rigorous functional test scripts (e.g., `eval_student_UX`, `eval_messy_queries`, `eval_theme_alignment`) and latency profiles rather than standard statistical TREC precision/recall matrices.
- **Retrieval Frame:** Highlight the **Staged Hybrid Pipeline**. Do not write it as simultaneous; Stage A runs sparse ILIKE hits, and Stage B triggers dense pgvector similarity only if depleted, bounded by dynamic validation.
- **Formulas:** Use the Clear Score, Geometric Mean Precision, and Coverage Penalty math extensively. They are distinct, heavily engineered, and provide absolute proof of academic system development.
- **Transparency Tools:** Highlight how `ResultCard.tsx` exposes Semantic/Lexical ratios natively to users, fulfilling an explainability requirement.
- **Stopwords:** Emphasize the *Dual Approach*: Stopwords are scored 0.0 locally, but retained universally in the XLM-RoBERTa encoder.

## J. Uncertainties / Inferred Parts
- **Evaluation Status Inferred:** The existence of dozens of scripts in `backend/scripts/` (`eval_final.py`, `eval_reranker.py`, etc.) proves extensive *calibration testing* takes place, but because no output JSONs of academic metrics like `Precision@10` were found upon search, I strictly classify the evaluation as Qualitative Calibration. Do not invent metric digits. 
- **DAPT Status Mismatch:** The code initializes a `SentenceTransformer(dapt_path)` but logs indicate it usually fails to find it and falls back to the base model. Write Chapter 3 assuming the implementation relies fundamentally on the base XLM-RoBERTa vectors, unless the user intends to prove the DAPT files were successfully injected.
