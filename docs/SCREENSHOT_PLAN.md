# Screenshot & Asset Plan

This document outlines the priority and specifications for capturing screenshots and assets for the Rizal Thematic Exploration System's portfolio and `README.md`.

## 1. Character Theme Explorer
*   **Filename:** `01-character-theme-explorer.png`
*   **Route/Page:** `/explore`
*   **Feature Demonstrated:** Thematic Exploration & Character Analysis
*   **Reason for Inclusion:** Translates abstract NLP features into a tangible, educational user experience. Showcasing the `ThemeGrid` and `CharacterList` immediately demonstrates the app's educational value.
*   **README Placement:** Under a "Thematic Exploration" subsection in the Features area.

## 2. Staged Retrieval & Dual Scoring
*   **Filename:** `02-staged-retrieval-scoring.png`
*   **Route/Page:** `/search` (specifically focusing on a `ResultCard`)
*   **Feature Demonstrated:** Staged lexical-first semantic-fallback retrieval
*   **Reason for Inclusion:** Clearly visualizes the hybrid AI analysis methodology to academic reviewers and recruiters by displaying both the Semantic Score (teal bar) and Lexical Score (amber bar) side-by-side.
*   **README Placement:** High up in the "Key Engineering Features" section, or directly under the main Hero/Demo.

## 3. Anti-Hallucination Gate
*   **Filename:** `03-anti-hallucination-gate.png`
*   **Route/Page:** `/search` (with an out-of-domain query like "Space travel in Noli")
*   **Feature Demonstrated:** Anti-hallucination domain validation gate
*   **Reason for Inclusion:** Highlights AI safety and strict academic domain enforcement, proving the system is rigorous and prevents LLM/search hallucinations.
*   **README Placement:** In the "Key Engineering Features" or "Architecture" section to highlight the system's robustness.

## 4. Reading Map View / RobustAligner
*   **Filename:** `04-reading-map-view.png` (for UI) & `robust-aligner-workflow.gif` (for README)
*   **Route/Page:** `/search` (Expanded `ContextModal` / `ExpandablePanel`)
*   **Feature Demonstrated:** Reading Map View & RobustAligner (dynamic-programming sequence alignment)
*   **Reason for Inclusion:** Proves that the search is context-aware and maps out the narrative structurally using neighborhood similarity weights.
*   **README Placement:** See recommendation below.

### 💡 Recommendation for RobustAligner / Sanggunian
While the `ContextModal` in the UI is the best *screen* to show the end-result of the RobustAligner (by visualizing neighbor similarity weights and context), **the RobustAligner feature itself is better communicated through a workflow animation (GIF) or simplified architecture diagram for the README.** 

**Why?** Dynamic-programming sequence alignment is an under-the-hood algorithmic process. A static screenshot of the UI only shows highlighted text, which doesn't convey the complexity of the alignment matrix or the sequence matching process. 
*   **Workflow Animation (GIF):** Highly recommended. An animation showing a query being broken down and aligned with the Sanggunian text visually communicates the "dynamic" nature of the algorithm.
*   **Simplified Architecture Diagram:** A great alternative if an animation is too time-consuming to produce, showing the flow from `Query -> Tokenization -> Alignment Matrix -> Contextual Output`.

## 5. Cross-Novel Split View
*   **Filename:** `05-cross-novel-split-view.png`
*   **Route/Page:** `/search` (Desktop view)
*   **Feature Demonstrated:** Dual Novel Comparison
*   **Reason for Inclusion:** It is the core MVP feature that solves the user's primary problem of cross-referencing *Noli Me Tangere* and *El Filibusterismo* simultaneously.
*   **README Placement:** Can serve as the primary `Application Showcase` image at the very top of the `README.md`, replacing `placeholder.gif`.
