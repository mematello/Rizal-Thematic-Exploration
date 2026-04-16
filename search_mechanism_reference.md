# Rizal Project - Search Mechanism Technical Reference

**Branch Source:** `fix/semantic-search-precision`
**Purpose:** Provide technical context and merge strategy for integrating with the `old-version-backup` branch. This document details the hybrid search workflow, scoring rules, and file topography as implemented in the precision-focused branch.

---

## 1. High-Level Overview

The search engine acts as a **Hybrid Semantic + Lexical System**. It processes and resolves Tagalog textual queries against two separate books (Noli Me Tangere and El Filibusterismo) using Postgres with pgvector. 

**The Pipeline:**
1. **Query Analysis & Decomposition:** The query is broken down into significant words vs stopwords (`QueryAnalyzer`) and grammatical roles (`TagalogRoleParser`).
2. **Dynamic Embedding Generation:** Generates vector embeddings for the query using `SentenceTransformer`. It uses a base model for single concepts and dynamically blends a highly-tuned DAPT (Domain-Adaptive Pretraining) model for complex multi-word queries.
3. **Stage-A Retrieval (Lexical First):** Scans the database for direct lexical hits on the significant words.
4. **Stage-B Retrieval (Semantic Fallback):** If lexical results are too few, it checks against a modern/out-of-domain blocklist, anchors the query against known novel concepts/themes, and retrieves items using cosine distance ranking.
5. **Hybrid Reranking:** Computes granular lexical and semantic similarity scores, penalizing partial coverage, stopword mismatches, or very short sentences. Reranks using dynamically weighted coefficients.
6. **CWPR Precision Scoring:** For multi-word queries, enforces a "Soft-AND" threshold across distinct concepts via Geometric Mean.
7. **Post-Processing Theme Tagging & Suggestion:** Attaches expanded contextual paragraphs, tags the sentences with associated core themes, and constructs related "Kaugnay na Paghahanap" suggestions.

---

## 2. Key Files Involved

| File Path | Role in the Pipeline |
|-----------|----------------------|
| **`backend/app/api/v1/search.py`** | The FastAPI endpoint (`/search`). Acts as the controller, mapping frontend requests like `source_type="buod"` to backend `source_type="summary"` and invokes the `RizalEngine`. |
| **`backend/app/core/engine.py`** | The heart of the system (`RizalEngine`). Contains the full workflow—embedding, lexical/semantic searching, penalizing, thresholding, and reranking candidates. |
| **`backend/app/core/analyzer.py`** | (`QueryAnalyzer`) Extracts semantic weight and calculates stopword significance. Used during the scoring phase to apply penalties if a query is mostly fluff. |
| **`backend/app/core/tagalog_parser.py`**| (`TagalogRoleParser`) Decomposes multi-word phrases into Tagalog semantic roles (Agent, Patient, Event, Oblique) to ensure strict component checking. |
| **`backend/app/services/suggestions.py`**| Generates context-aware string suggestions based on top search results and anchored theme clusters. |
| **`frontend/hooks/useRizalSearch.ts`**| The client-side TanStack React Query hook. Invokes the backend API and transforms the complex backend JSON return into typed `ResultCardProps` suitable for the UI. |

---

## 3. How Scoring Works (`engine.py`)

Scoring is handled during the **Hybrid Re-ranking Phase** inside `RizalEngine.search`:

1. **Initial Vector Search:** Uses `Sentence.embedding.cosine_distance(query_list)`.
2. **Lexical Score:** Computes a `lex_score` using `_compute_lexical_score()`. It gives bonuses for exact phrase matches and subtracts points if the matched words are just non-vital stopwords.
3. **Semantic Score:** Extracts base dot-product similarity (`sem_score_base`). If it's a complex query, it takes `max(sem_score_base, sem_score_dapt)`.
4. **Validations & Penalties:** 
   - Sentences extremely short (< 5 words) or just exclamations are halved in score. 
   - A `semantic_fallback` gate aggressively drops candidates showing no specific synonym or dynamic-cache similarity matches.
   - Partial match representations (when querying multiple concepts but the result only contains one) are heavily penalized using coverage ratios.
5. **Final Score Aggregation:** Evaluates `lam_lex` and `lam_sem` (weights) dynamically using `_compute_dynamic_weights()`.
   - Very short passage matches place extremely high weight (90%) on Lexical hits.
   - Longer, explanatory passages raise the Semantic weight up to 70%.
6. **Precision Score:** Multi-word concepts calculate a specific component geometric mean (CWPR Precision). If a semantic result has a High Score but extremely Low Precision (meaning it's completely missing one side of a 2-part query), it is completely discarded.

---

## 4. How Theme Tagging is Applied

Sentences are classified to specific themes during post-processing (`_classify_themes` in `engine.py`).
1. `theme_matrix` is formed when `engine.py` caches all `Theme` items from the database at startup.
2. The engine uses a hybrid score computing `(0.5 * sentence_similarity) + (0.5 * context_passage_similarity)`.
3. If the score is higher than `0.55`, it assigns the theme to the search hit.
4. **Override Rule:** If a user directly typed the exact keyword of a theme (e.g. "Edukasyon"), it bypasses suppression and guarantees the theme tag.

---

## 5. Frontend-Backend Connection

1. The frontend invokes: `GET /api/v1/search?q={query}&source_type={mode}`
2. **Note on Modes:** The frontend generally passes `buod` (Summary) or `full`. The backend `search.py` maps `buod` to `summary` behind the scenes as a database parameter limit on SQLAlchemy queries.
3. Return shape is a strict dict structure containing `results: { noli: [], elfili: [] }`, `metadata: {...}`, and `precise_matches: []`.
4. `useRizalSearch.ts` uses `.map()` to morph the Python backend objects. Most notably, it constructs the Boolean `confidenceBadge: item.scores?.final > 85`.

---

## 6. Overlap with `old-version-backup` and Merge Risks

The `old-version-backup` branch focuses on two features not currently native to the `fix/semantic-search-precision` pipeline:
1. **Tauhan-Themes Logic:** Might interact with how specific Characters (Tauhan) map onto themes. This could interfere with `engine._generate_hybrid_theme_pool` or `_classify_themes`.
2. **Sanggunian/Reference Mapping:** Translates `buod` (Summary) passages into their canonical `full` textual references.

### Conflict-Prone Files
* **`backend/app/core/engine.py`**: Extremely likely to have merge conflicts as both branches likely modified `search()` output loops. The precision branch has extensively rewritten the `.search()` iteration block.
* **`backend/app/api/v1/search.py`**: If old-version modified the database parameter intake or API payload schema, conflicts will occur.
* **`frontend/hooks/useRizalSearch.ts`**: The UI mapping schema might not account for new fields introduced by `old-version-backup`, such as a `sanggunianReference` property.
* **`frontend/components/SearchResultCard` (or similar)**: Any component displaying the "Sanggunian" text will need careful state alignment.

---

## 7. Suggested Merge Strategy

To safely merge `fix/semantic-search-precision` with `old-version-backup`, proceed incrementally:

1. **Keep `RizalEngine.search` intact.** The Precision Branch is the absolute source of truth for querying, retrieving, and ranking vectors. Do **not** replace the retrieval phase (`Stage A` and `Stage B`) with old code.
2. **Inject Sanggunian in Post-Processing.** Look at `Step 4: Post Processing` around line `676` in `engine.py`. Integrate the reference/sanggunian logic exactly where `item['context_text']` and `item['themes']` are appended. 
    * Add logic like `if source_type == 'summary': item['sanggunian'] = get_full_version_reference()`.
3. **Adopt Frontend Typed Mapping Safely:** When updating `useRizalSearch.ts`, strictly union the types: add the sanggunian arrays/strings into `ResultCardProps` without breaking the `confidenceBadge` logic.
4. **Tauhan-Themes integration:** Find where `old-version-backup` injects `tauhan-themes`. If it operates independently, ensure it does not drastically shift the `theme_matrix` logic (which calculates vocabulary and idf specifics on load). If necessary, create a standalone `_classify_tauhan` method instead of combining them into `_classify_themes`.
