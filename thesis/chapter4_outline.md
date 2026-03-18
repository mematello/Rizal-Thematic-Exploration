# PROPOSED OUTLINE FOR CHAPTER 4: IMPLEMENTATION & RESULTS

## 4.1 Implementation Overview
- Short recap of the development environment and final technology stack (Next.js, FastAPI, PostgreSQL + pgvector).
- Description of the final corpus size and database indices.

## 4.2 Retrieval Performance Analysis
### 4.2.1 Stage A vs Stage B Efficacy
- Quantitative breakdown of how often the system relies on Lexical First (Stage A) versus Semantic Fallback (Stage B).
- Impact of the Lexical-Priority Reservation feature in preserving exact matches.

### 4.2.2 Query Concept Coverage
- Evaluation of the system's ability to handle multi-word queries.
- Comparison of Strong vs. Partial matches for complex thematic queries.
- Analysis of the Coverage & Noise Penalty's effect on pruning irrelevant results.

### 4.2.3 Semantic Validation and Hallucination Prevention
- Case studies on the "In-Domain Gate" blocking out-of-domain queries (e.g., Modern terms testing).
- Evaluating the effectiveness of the Dynamic Semantic Validation (strict synonym thresholds) at reducing hallucinations by OOV terms.

## 4.3 UI and Interaction Experiences
- Implementation of the frontend features (Split-view, Context Expansion).
- Review of the Dynamic Suggestion generation and how accurately it extracts follow-up themes.
- Client-side performance metrics (Caching behavior with React Query & Redis).

## 4.4 Observations and Limitations
- Edge cases where the Semantic Fallback either excels or falls short.
- Discussion on the limitations of the Tagalog dataset variations and short-sentence penalizations.
- Summary of the impact of the hybrid architecture on thematic literary exploration.
