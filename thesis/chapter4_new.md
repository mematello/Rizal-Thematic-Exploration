# CHAPTER 4 IMPLEMENTATION AND RESULTS

## 4.1 Implementation Overview
This chapter presents the operational results of the hybrid information retrieval system designed for the thematic exploration of *Noli Me Tangere* and *El Filibusterismo*. Following the methodology established in Chapter 3, the final system consists of a Next.js frontend, a FastAPI backend, and a PostgreSQL database utilizing the `pgvector` extension for semantic storage. Performance evaluation in this context relies on functional verification of the retrieval pipeline and qualitative analysis of system behavior across varying query complexities, rather than formal statistical measures. This chapter analyzes how effectively the system translates user queries into thematic insights and how architectural refinements have resolved early implementation challenges.

## 4.2 Retrieval Behavior Analysis

The core of the system's evaluation centers on its ability to handle different levels of query complexity. By analyzing the system's behavior across five progressive levels, we can directly compare the efficacy of the lexical-first strategy (Stage A) against the semantic fallback (Stage B), while demonstrating the necessity of the hybrid architecture.

### 4.2.1 Level 1: Simple Lexical Queries
**Query Example:** "kamatayan" (death)

**System Behavior Comparison:**
Simple, single-word thematic queries rely heavily on exact term intersections.
- **Lexical-Only:** Excels at finding direct references, rapidly fetching sentences where the word "kamatayan" explicitly appears. However, it fails to capture synonymous phrases such as "pagkawala ng buhay" or "pagpanaw."
- **Semantic-Only:** Fetches the exact matches, but often dilutes the results by pulling in conceptually adjacent but narratively distant sentences about grief or illness that do not ultimately result in death.
- **Hybrid System (Final):** The system successfully leverages Stage A (Lexical-First) to secure foundational text matches. 

**Architectural Improvements (Before vs. After):**
- *Before (Semantic Fallback Flooding)*: In early iterations, the system employed a simultaneous scoring formula where dense neural embeddings would overpower sparse lexical matches. A highly semantic sentence without the word "kamatayan" would score higher than a sentence containing the exact word, burying the most direct textual evidence.
- *After (Lexical-Priority Reservation)*: The final implementation guarantees that passages with true lexical hits bypass the truncation logic, ensuring that undeniable textual evidence remains at the top of the search results, while semantic expansions follow below.

### 4.2.2 Level 2: Low-Frequency Lexical Queries
**Query Example:** "kasalanan" (sin/fault)

**System Behavior Comparison:**
Terms that occur infrequently in the *buod* summaries challenge both ends of the retrieval spectrum.
- **Lexical-Only:** Frequently returns too few results (e.g., yielding only 2 to 3 sentences across the entire corpus), resulting in an incomplete thematic picture.
- **Semantic-Only:** Struggles because the vector for "kasalanan" overlaps heavily with general negativity, retrieving overly broad passages about anger or sadness.
- **Hybrid System (Final):** The engine triggers Stage B (Semantic Fallback) when the lexical pool is depleted.

**Architectural Improvements (Before vs. After):**
- *Before (Exact-Match Consistency Bug)*: Previous iterations struggled to properly score variations like "pagkakasala" or "nagkasala," penalizing sentences that did not exactly match the root word.
- *After*: The system now integrates robust substring matching and Tagalog-specific preprocessing alongside the semantic fallback, dynamically expanding the pool of candidates when exact matches are exhausted.

### 4.2.3 Level 3: Multi-Concept Queries
**Query Example:** "edukasyon ni ibarra" (education of Ibarra)

**System Behavior Comparison:**
Queries combining a thematic concept with a specific narrative entity require high precision.
- **Lexical-Only:** Requires both "edukasyon" and "Ibarra" to appear in the same sentence. Because natural language often separates these concepts across adjacent sentences, lexical-only retrieval frequently yields zero results.
- **Semantic-Only:** Returns sentences generally about Ibarra or generally about education. It struggles to enforce that *both* concepts must be present, resulting in partial topical relevance.
- **Hybrid System (Final):** Applies Query Decomposition. The engine creates component vectors for both "edukasyon" and "Ibarra," scoring passages based on their ability to satisfy both anchors simultaneously.

**Architectural Improvements (Before vs. After):**
- *Before (Partial Coverage Noise)*: The engine would return sentences where Ibarra was merely speaking about the weather, because the strong "Ibarra" entity vector masked the absence of the "edukasyon" concept.
- *After (Coverage Penalty)*: A programmatic penalty is now applied to passages lacking comprehensive coverage. The engine explicitly labels retrievals as *Strong* (satisfying 100% of concept anchors) or *Partial*, actively suppressing noise.

### 4.2.4 Level 4: Abstract/Thematic Queries
**Query Example:** "kalayaan ng bayan" (freedom of the nation)

**System Behavior Comparison:**
Abstract queries represent the hardest challenge for traditional systems, as the text rarely states these concepts verbatim, instead conveying them through character actions or dialogues.
- **Lexical-Only:** Completely fails. Sentences discussing revolution, breaking chains, or fighting the colonizers do not necessarily contain the word "kalayaan."
- **Semantic-Only:** Required to do the heavy lifting. The XLM-RoBERTa embeddings successfully map "kalayaan" to the latent semantic space of rebellion, sacrifice, and reform.
- **Hybrid System (Final):** Recognizes the absence of Stage A hits and fully leans into Stage B, utilizing context expansion to provide narrative proof of the abstract concept.

**Architectural Improvements (Before vs. After):**
- *Before (sem_score Threshold Issue)*: Early implementations accepted any semantic match, even if the cosine similarity was incredibly low (e.g., 0.20), leading to passages that were tangentially related at best.
- *After (Dynamic Validation):* A strict similarity threshold (> 0.55) was instituted. If no passage meets this threshold, the system prefers returning zero results rather than hallucinating irrelevant narrative connections.

### 4.2.5 Level 5: Out-of-Domain / Guard Case
**Query Example:** "internet sa panahon ni rizal" (internet during Rizal's time)

**System Behavior Comparison:**
This level tests the system's ability to recognize queries that fall outside the historical and literary confines of the corpus.
- **Hybrid System (Final):** Employs explicit guardrails to prevent semantic hallucination.

**Architectural Improvements (Before vs. After):**
- *Before*: The semantic engine would attempt to find the "closest" concept to the internet, occasionally returning passages about the telegraph, letters, or communication. While technologically interesting, this is a literal failure for thematic literary analysis.
- *After (Vocabulary Guards & Blocklists)*: The `MODERN_TERMS` blocklist immediately catches anachronistic inputs. Furthermore, Out-Of-Vocabulary (OOV) terms face punishing similarity thresholds, effectively shutting down the query before Stage B can hallucinate an answer.

## 4.3 UI Behavior and User Interaction

The front-end interface serves not just as a display layer, but as an explainability tool that makes the complex backend logic transparent to the student or researcher.

### 4.3.1 Visualizing the Dual-Novel Architecture
The interface implements a localized Zustand state store to seamlessly toggle or compare results between *Noli Me Tangere* and *El Filibusterismo*. This grouping is vital for user interaction, as it allows students to map how a theme evolves from the first novel into the second. Mobile users experience an animated tab-switcher, while desktop users analyze the results in a split-view layout.

### 4.3.2 Transparent Query Explainability
A critical design requirement was breaking the "black box" nature of AI retrieval. 
- **Confidence Badges**: The UI actively reflects the backend’s scoring mechanics by assigning confidence badges to each result card. Badges explicitly denote whether a result is an *Exact Lexical Match*, a *Semantic Match*, or a *Partial Concept Match*. This directly addresses the backend's Coverage Penalty logic, visually informing the user when a sentence only partially answers a Level 3 query.
- **Visual Score Bars**: Beneath the passage, users are provided with visual meter bars displaying the explicit Semantic and Lexical scores retrieved from the PostgreSQL database. This allows users to understand *why* a particular sentence was ranked highly.

### 4.3.3 Context Expansion
Because sentences often lose meaning when isolated, user interaction relies heavily on the "Show Context" toggle. Driven by the backend's context expansion module and rendered smoothly via CSS transitions in the frontend, this feature fetches adjacent sentences on demand. This ensures that users do not just find keyword hits, but rather full, continuous narrative passages necessary for proper literary citation.

### 4.3.4 Dynamic Suggestions and Continued Exploration
Instead of leaving the user at a dead end after a search, the UI renders "Kaugnay na Paghahanap" (Related Searches). By utilizing the query-class mapping architecture established in the backend, the system presents 3 to 4 related thematic queries. If a user queries "prayle," the UI presents targeted follow-ups like "impluwensya ng prayle" or "kapangyarihan ng simbahan," encouraging deeper, structured exploration of the novels' intersecting themes.
