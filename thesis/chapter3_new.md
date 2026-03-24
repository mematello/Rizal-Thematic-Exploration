# CHAPTER 3 METHODOLOGY

## 3.1 Research Design

This study employs an architectural design and system development approach to construct an enhanced hybrid information retrieval system for the thematic analysis of José Rizal's *Noli Me Tangere* and *El Filibusterismo*. The research focuses on engineering a functional pipeline that integrates XLM-RoBERTa semantic embeddings with weighted lexical matching and Tagalog-specific preprocessing. The design methodology is divided into three primary phases: (1) corpus preparation and text processing, (2) system development integrating a staged hybrid retrieval architecture, and (3) construction of the client-server ecosystem for data presentation. 

The research is guided by principles of Hybrid Retrieval and Semantic Search, translating these theories into practical, implemented mechanics such as semantic similarity thresholds, multi-anchor queries, and lexical-priority matching. Evaluation is based on functional verification and system profiling to calibrate the pipeline, rather than formal statistical metrics.

## 3.2 Textual Corpus and Preprocessing Pipeline

### 3.2.1 Corpus Selection
The study utilizes Filipino language summary versions (*buod*) of both novels. The *Noli Me Tangere* summaries are sourced from PinoyCollection.com, and *El Filibusterismo* summaries are sourced from Noypi.com.ph. These summaries maintain linguistic consistency suitable for NLP processing while preserving the core thematic content and narrative structure. The corpus comprises sentences segmented into distinct passages, providing naturally coherent boundaries for retrieval context.

### 3.2.2 Text Cleaning and Deduplication
The preprocessing pipeline prepares the raw text for database ingestion and semantic encoding:
- **String Cleaning**: Removal of special characters, whitespace normalization, and case normalization.
- **Punctuation Handling**: Preserving sentence-terminating markers for boundary detection.
- **Deduplication**: A global deduplication mechanism using term-frequency similarity identifies and removes redundant sentences, ensuring that the retrieval engine does not return identical content pieces.

### 3.2.3 Stopword Management
The system utilizes the `stopwords-iso-tl` package to manage common Tagalog stopwords. The retrieval engine handles these systematically during different retrieval stages:
- **Lexical Overlap**: Stopwords are assigned a zero-weight in lexical scoring to prevent generic terms from dominating the exact-match calculations.
- **Neural Processing**: The stopwords are retained during the XLM-RoBERTa tokenization phase to preserve the grammatical context and intended semantic meaning of the sentences.

## 3.3 Enhanced Conceptual Architecture

The core of the retrieval system is driven by a staged hybrid pipeline designed to process Tagalog thematic queries effectively.

### 3.3.1 Query Validation and Input Processing
When a query enters the engine, it is subjected to programmatic checks before retrieval begins:
- **Modern Terms Blocklist**: The system contains a predefined set of anachronistic terms (e.g., "tiktok," "internet", "cellphone"). Queries containing these terms are explicitly blocked to prevent out-of-domain semantic retrievals.
- **Query Decomposition**: Multi-word queries are parsed and split into sub-components. This allows the system to perform multi-anchor searches, verifying that retrieved sentences address the specific parts of a complex query rather than just matching a generalized holistic vector.
- **Vocabulary Guard**: The system checks query words against the known corpus vocabulary. If Out-of-Vocabulary (OOV) terms are dominant, stricter semantic similarity thresholds are applied before results are returned.

### 3.3.2 Staged Hybrid Retrieval Protocol
The engine implements a **Staged Retrieval Approach** to balance exact phrase matching with conceptual similarity:

**Stage A: Lexical-First Retrieval**
The system first scans the database utilizing PostgreSQL's `ILIKE` functionality to find sentences containing exact intersections of the query's significant words. For multi-component queries, local neighbors surrounding exact matches are also fetched to ensure context is captured.

**Stage B: Semantic Fallback & Dynamic Validation**
If Stage A returns too few candidates, the system triggers the Semantic Fallback:
- Contextual embeddings (768-dimensional vectors from the base XLM-RoBERTa pre-trained model) are used to perform cosine similarity searches via the `pgvector` extension in PostgreSQL.
- **Semantic Similarity Thresholds**: To ensure matches are conceptually relevant, the system imposes a stringent semantic similarity threshold. Passages must contain terms that exhibit high cosine similarity (> 0.55) to the query's core concepts to survive the fallback stage.
- **Coverage Penalty**: Passages are analyzed to verify if all conceptual anchors of a multi-word query are represented. The engine penalizes passages that offer only partial coverage, prioritizing results where all query concepts are either lexically present or semantically closely related.

### 3.3.3 Re-ranking and Lexical-Priority Reservation
Candidates from both stages are pooled and re-ranked using a combined formula of lexical density and semantic proximity. To prevent dense semantic scores from drowning out verbatim matches, the engine implements **Lexical-Priority Reservation**. Sentences containing actual textual matches for the query's core words bypass the final truncation logic, ensuring they remain at the top of the search results.

### 3.3.4 Context Expansion and Theme Classification
- **Context Expansion**: For retrieved results, the engine fetches the immediately preceding and succeeding sentences within the chapter to construct a continuous passage, enhancing readability.
- **Theme Tagging**: Retrieved sentences are matched against a matrix of predefined thematic embeddings. Sentences exceeding the similarity threshold are tagged with foundational themes (e.g., colonialism, justice).

### 3.3.5 Query-Class Suggestion Architecture
The system includes a Follow-Up Suggestion module using an explicit **Query-Class Mapping** approach. The system maps the user's input against 60 predefined thematic categories (Broad Themes, Theme Phrases, and Entities) and their common aliases. Based on deterministic dictionary lookups, it extracts relevant related searches (e.g., querying "prayle" maps strictly to "impluwensya ng prayle"). Interrogative questions and literal unmapped fragments are explicitly suppressed to maintain high-quality guidance without relying on unpredictable generative processes.

## 3.4 System Architecture Implementation

The final application ecosystem is built utilizing a decoupled client-server model.

### 3.4.1 Client Layer (Frontend)
The user interface is constructed using **Next.js 16** (React 19) styled with TailwindCSS 4 and Framer Motion.
- **Client State**: The `zustand` library manages localized UI states such as active tabs (mapping between *Noli* and *Fili* views) and visible result counts.
- **Server Data Synchronization**: The `@tanstack/react-query` package handles asynchronous fetching, loading states, and client-side caching of the API responses.

### 3.4.2 API and Caching Layer
- **FastAPI Gateway**: The backend is written in Python using FastAPI, servicing routes for searching, suggestions, and chapter metadata.
- **Caching**: A Redis layer caches highly recurrent query results and manages rate-limiting traffic, minimizing redundant calculations by the core engine.

### 3.4.3 Data Layer
- **PostgreSQL**: Stores the processed passage text and metadata, and handles standard queries.
- **pgvector**: The PostgreSQL extension handles the storage and retrieval (via IVFFlat indexing) of the 768-dimensional XLM-RoBERTa sentence embeddings, facilitating vector-based cosine similarity operations.

## 3.5 Functional Verification and System Calibration

The study explicitly focuses on system engineering rather than formal statistical evaluation. Evaluation is based on functional verification of the retrieval pipeline and continuous system calibration, rather than precision, recall, or F1-score metrics:
- **Retrieval Pipeline Verification**: Internal scripts (`check_backend.py`) continuously validate that the dual-novel fetching logic returns the expected structural metadata, correctly routing queries to the appropriate database sections for *Noli* and *Fili*.
- **System Calibration**: The latency of semantic lookups and query parsing logic is monitored to calibrate threshold values (e.g., the 0.55 similarity threshold for semantic validation and stopword penalty factors), ensuring reasonable response times.
- **Fallback Assessment**: Ad-hoc query runs are logged to ensure the engine's behavior during Stage A failures appropriately triggers Stage B, filtering noise via the implemented blocklists and vocabulary guards.
  