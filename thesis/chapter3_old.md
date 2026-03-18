## CHAPTER 3 METHODOLOGY

## 3.2 Preprocessing Pipeline

A systematic preprocessing pipeline transforms the raw summary texts into formats suitable for  both  lexical  and  semantic  analysis.  The  pipeline  implements  language-specific  preprocessing informed by the literature reviewed in chapter 2.

Start

Source Text Acquisition

(Noli Me Tangere &amp; El Filibusterismo)

Text Cleaning:

• Remove special characters

Fix encoding

• Normalize whitespace

Text Segmentation:

&lt;Data Validation

Yes

Storage in

Database

End

-No-

Figure 2: Data Collection and Preprocessing Pipeline

<!-- image -->

## 3.3 Conceptual Framework

The  conceptual  framework  illustrates  how  the  dual-formula  hybrid  information  retrieval system  integrates  multiple  theoretical  foundations  and  technical  components  to  enable  thematic literary analysis of Rizal's novels. The framework operates through five interconnected layers with two distinct scoring formulas optimized for different retrieval contexts, enhanced by domain-adaptive semantic grounding validation.

LEXICAL RETRIEVAL

Weighted overlap based on word frequency weights

PHASE 4: SENTENCE MATCH → THEMATIC

CLASSIFICATION

Semantic: XLM-RoBERTa cosine similarity

Lexical: Jaccard coefficient (sentence vs. theme meaning)

Dynamic weights: Based on relative meaning length

INPUT LAYER

User Query

Text Corpus

(Noli Me Tangere &amp; El Filibusterismo summaries)

Figure 3: Conceptual Framework for the Hybrid Information Retrieval System Using XLMRoBERTa for Thematic Literary Analysis of Rizal's Novels

<!-- image -->

3.3.1 Input Layer: Query and Corpus

The system receives two primary inputs:

User Query:

Thematic queries in Filipino/Tagalog representing specific literary themes (e.g.,

"kolonyalismo," "edukasyon bilang pag-asa ni Ibarra," "katarungan," "kamatayan ni Basilio"). Queries undergo both linguistic validation to ensure they contain valid Filipino words with sufficient semantic

content and domain-adaptive semantic grounding validation to verify meaningful connection to the corpus domain.

Text  Corpus:

Filibusterismo

Filipino  language  summary  versions  (

buod

)  of

Noli  Me  Tangere and

El

,  providing  comprehensive  coverage  of  major  themes  while  maintaining  linguistic consistency and manageable computational scope.

3.3.2 Processing Layer: Domain-Adaptive Semantic Validation

Drawing  from  Domain-Adaptive  Pre-training  Theory  (Krieger  et  al.,  2022),  the  system implements a novel semantic grounding validation layer that detects queries with no meaningful

semantic relation to the corpus. This layer prevents the retrieval of weakly similar but thematically irrelevant passages, addressing a critical limitation of pure semantic similarity approaches.

62

Eq is

DomainAlignment (g, C) = max CosineSim (eq, Cp)

IS C

h: DomainAlignment (g, C) &lt; Odomain w re Odomain = 0.30

Cp a pEC

## Semantic Grounding Assessment:

The domain-adaptive validation computes the semantic alignment between the query and the entire corpus domain:

where:

- is the XLM-RoBERTa embedding of the query
- are embeddings of all passages in corpus
- Maximum similarity indicates strongest domain connection

## Rejection Criteria:

Queries are rejected if:

1. Semantic Mismatch: where (empirically determined threshold)
2. Lexical Absence: All content words absent from corpus vocabulary
3. Thematic Disconnect: Query semantically distant from all predefined themes

## Rejection Response:

Rather than returning weakly similar passages (which would constitute semantic hallucination), the system explicitly returns:

"No thematically or semantically relevant passage found. The query does not align with the literary content of the novels."

s DomainAlignment a

Phase 1: Domain Adaptation

(Offline Initialization)

Encode Entire Corpus

XLM-ROBERTa → 768 dim. Embeddings

Encode Entire Corpus

#\_corpus = mean(all passage embeddings)

## Example Application:

Phase 2: Query Validation

Query:

"Kamatayan ni Basilio" (Death of Basilio)

Reality:

Basilio does not die in the novel summaries.

Filipino word frequency check (wordfreq)

Lexical grounding (corpus vocabulary)

2. DAPT Semantic Validation

• max (cos\_sim(q. P)) = 0.30

• Theme Proximity

• max (cos\_ sim(q. T)) = 0.35

## System Behavior:

Query REJECTED

System Response:

No Thematically or Semantically relevant passage found. Query

does not align with the literary content of the novels."

Diagnostic Infomartion

Theme proximity score

Absent content words

Proceed to Retrieval

Formula 1: Main Retrieval

1. Computes across all passages

Thresholds?

Formula 2: Context Expansion

2. Finds maximum semantic similarity = 0.24 (below 0.30 threshold)
3. Despite lexical matches for "Basilio," semantic embeddings indicate thematic mismatch
4. Returns: "No thematically or semantically relevant passage found."
5. Does NOT hallucinate: Results about "Basilio" or "sacrifice" that are semantically weak

Figure 4: Domain-Adaptive Pre-training (DAPT) Validation Architecture for Semantic Grounding

<!-- image -->

This domain-adaptive validation implements the principle established by Krieger et al. (2022) that domain-specific adaptation improves task performance when the pre-training domain aligns with the target task. By explicitly modeling the semantic space of Rizal's novels, the system learns to distinguish  between  queries  that  belong  to  this  domain  versus  those  that  do  not,  even  when superficial lexical overlap exists.

## 3.3.3 Processing Layer: Dual-Formula Hybrid Retrieval Architecture

After passing semantic grounding validation, the processing layer implements a dual-formula hybrid retrieval architecture that employs two distinct scoring formulas optimized for different retrieval contexts:

Formula  1  (Main  Retrieval): Query→Corpus  matching  with  dynamic  weights  favoring  lexical precision for short queries

Formula 2 (Neighbor Retrieval): Sentence→Sentence matching with dynamically adjusted weights that emphasize semantic continuity and rare-word importance for more accurate context expansion.

This dual-formula approach addresses the fundamental difference between:

- Primary retrieval: Finding passages directly relevant to user queries (requires precision and keyword alignment)
- Context retrieval: Finding thematically coherent neighboring sentences (requires semantic continuity)

The processing layer combines four parallel processing streams:

Preprocessing  Pipeline: Applies text cleaning, Tagalog stopword removal (using stopwords-iso-tl),  tokenization  (using  XLM-RoBERTa  tokenizer),  passage  segmentation,  and deduplication. This stream ensures text is properly formatted for both lexical and semantic analysis while removing noise and redundancy.

Semantic Component: Utilizes XLM-RoBERTa multilingual transformer model (sentencetransformers/paraphrase-xlm-r-multilingual-v1) to generate 768-dimensional contextual embeddings for both queries and passages. This component implements Cross-lingual Transfer Learning Theory, enabling  semantic  understanding  of  Filipino  text  through  knowledge  transfer  from  high-resource languages. Cosine  similarity  measures  semantic  relatedness  between  query  and  passage embeddings.  The  same  model  is  used  for  both  main  and  neighbor  retrieval,  but  with  different weighting schemes.

Lexical  Component: Employs  weighted  term  matching  to  quantify  keyword  overlap between queries and passages. Unlike traditional TF-IDF, the system implements stopword-aware semantic weighting where each word receives a weight based on its corpus frequency and stopword status:

Frequency here means relative frequency, or how often a word appears compared to the total number of words in the corpus.

- Stopwords: weight = 0.0
- Very common words (frequency &gt; 0.001): weight = 0.3
- Medium frequency words (0.0001 &lt; frequency ≤ 0.001): weight = 0.7

- Rare/content words (frequency ≤ 0.0001): weight = 1.0

This weighting preserves the precision advantages of exact-match retrieval while minimizing stopword influence.

Hybrid  Integration: Combines  semantic  and  lexical  signals  through  two  specialized  multicomponent scoring functions:

- Main  Formula: Dynamically  adjusts  component  weights  based  on  query  characteristics (length, stopword  ratio), favoring lexical matching  for short queries  and  semantic understanding for longer queries
- Neighbor Formula: Uses dynamically adjusted weights based on corpus word frequency and  contextual  relevance  to  assess  sentence-to-sentence  coherence.  This  adaptive weighting  emphasizes  semantically  informative  and  rare  words,  enabling  finer-grained thematic alignment between neighboring sentences.

This dual-formula integration operationalizes Hybrid Retrieval Theory by explicitly modeling complementary retrieval signals while adapting the balance to retrieval context.

## 3.3.4 Analysis Layer: Thematic Classification and Context Expansion

The  analysis  layer  enriches  retrieved  passages  with  thematic  information  and  applies neighbor-formula-based context expansion:

Thematic  Classification: Matches  retrieved  passages  against  predefined  themes  from established literary sources (GradeSaver, LitCharts, Francisco) using semantic similarity. Passages

68

scoring  above  threshold  confidence  (≥0.45)  are  tagged  with  relevant  themes,  enabling  thematic analysis beyond simple passage retrieval.

Context  Expansion  with  Neighbor  Formula:

For  top-ranked  passages,  the  system identifies adjacent sentences using Formula 2 (Neighbor Retrieval) to assess semantic coherence

between the main sentence and its neighbors. Only sentences exceeding the neighbor relevance threshold  (combined  score  ≥0.4

0)  are  included  as  context.  This  ensures  that  context  expansion maintains thematic continuity rather than including arbitrary adjacent sentences.

Multi-Component  Scoring:

Applies  final  ranking  based  on  weighted  combination  of semantic  similarity,  lexical  overlap,  and  passage  length  normalization  using  Formula  1  (Main

Retrieval).  This  implements  Thematic  Coherence  Theory  by  assessing  relevance  at  lexical, semantic, and structural levels.

## 3.3.5 Output Layer: Retrieval Results and Visualization

The output layer presents results with multiple analytical dimensions, including transparent display of both formulas' contributions:

Ranked Passages:

Top-k most relevant passages with associated metadata including:

- Chapter location (number and title)
- Match type (exact, partial lexical, semantic)
- Confidence scores (semantic, lexical, final) computed using Formula 1
- Thematic tags with confidence levels
- Context sentences with Formula 2 neighbor scores

id Asem)

al Alexa

(Alex, Asem) =

(Alex, Asem) =

( (0.75, 0.25) if sentence length ≤ 5

(0.65, 0.35)

if 5 &lt; sentence\_length ≤ 10

(0.30, 0.70) if length ratio ≥ 1.5 (much longer)

if 10 &lt; sentence length ≤ 15

## Dual-Formula Metrics Display: The system explicitly shows:

(0.50, 0.50) if 0.8 ≤ length ratio &lt; 1.2 (similar)

- Formula 1 weights for main retrieval ( and )

<!-- formula-not-decoded -->

- Formula 2 dynamic weights for neighbor retrieval

- Individual semantic and lexical scores for each neighbor sentence
- Combined neighbor scores showing how Formula 2 evaluates context relevance

Performance  Metrics: Quantitative  measures  including  precision,  recall,  F1-score,  and component contribution analysis (semantic vs. lexical vs. hybrid performance for both formulas).

Thematic  Distribution: Visualization  of  how  themes  distribute  across  both  novels, identifying thematic concentration patterns and chapter-level prevalence.

Co-occurrence  Analysis: Identification  and  visualization  of  passages  where  multiple themes intersect, revealing thematic relationships and complexity in Rizal's social commentary.

Rejection  Notifications: For  queries  failing  domain-adaptive  semantic  validation,  clear explanation of why no results were returned, preventing user confusion from weak or hallucinated matches.

## 3.3.6 Theoretical Integration

The conceptual framework integrates the four theoretical foundations as follows:

Domain-Adaptive  Pre-training  Theory provides  the  foundation  for  semantic  grounding validation, ensuring that the system has been adapted to understand the specific domain of Rizal's novels. This prevents retrieval of passages that are superficially similar but thematically irrelevant, implementing Krieger et al.'s (2022) principle that domain adaptation improves performance when pre-training and target domains align.

Hybrid  Retrieval  Theory guides  the  architectural  principle  of  integrating  lexical  and semantic retrieval through two specialized, context-aware formulas rather than relying on a single uniform  weighting  scheme.  This  dual-formula  design  acknowledges  that  the  optimal  balance between lexical precision and semantic understanding varies with retrieval context-direct query matching (Formula 1) demands adaptive term weighting for relevance, while contextual coherence assessment (Formula 2) benefits from dynamic, frequency-based weighting to preserve thematic flow across sentences.

Cross-lingual Transfer Learning Theory enables both semantic components to function effectively on Filipino text despite XLM-RoBERTa's limited Filipino-specific training data. The model's

multilingual pre-training creates shared semantic spaces where concepts cluster together regardless of language, supporting both main and neighbor retrieval.

Thematic Coherence Theory informs the hierarchical relevance assessment across lexical (keyword overlap), semantic (conceptual similarity), and thematic (topic classification) levels. The framework recognizes that literary relevance operates at multiple dimensions beyond surface-level matching, with Formula 1 optimizing for query-document relevance and Formula 2 optimizing for discourse-level coherence.

## 3.2.7 Framework Validation

The conceptual framework's effectiveness is validated through:

Domain-Adaptive  Validation  Testing: Evaluation  of  semantic  grounding  rejection  for known-negative queries (queries about events/characters not in the novels) to measure false positive prevention.

Comparative  Evaluation: Hybrid  system  performance  compared  against  keyword-only baseline to demonstrate value of semantic integration.

Ablation Studies: Individual component testing (semantic-only, lexical-only, main formula vs. neighbor formula, hybrid without penalties, with/without domain  validation) to validate architectural decisions.

Formula  Effectiveness  Analysis: Evaluation  of  whether  Formula  2  produces  more coherent context than using Formula 1 for neighbor retrieval.

Expert  Review: Qualitative  assessment  by  literature  educators  to  confirm  thematic relevance and practical utility.

Distribution  Analysis: Empirical examination of thematic patterns to  demonstrate capabilities beyond traditional close-reading.

This conceptual framework provides the organizing structure for the detailed methodology that follows, showing how theoretical foundations translate into practical dual-formula system design with domain-adaptive semantic validation and evaluation procedures.

## 3.4. Textual Corpus

The study utilizes Filipino language summary versions ( buod ) of both novels rather than the complete  original  texts.  This  decision  was  made  based  on  several  practical  and  linguistic considerations:

Noli  Me  Tangere  Summary: Sourced  from PinoyCollection.com (2017), providing comprehensive coverage of the novel's 64 chapters with major plot points and themes preserved in sentence-level segmentation.

El Filibusterismo Summary: Sourced from Noypi.com.ph (2019), covering all 39 chapters with thematic elements preserved in sentence-level segmentation.

## 3.4.1 Rationale for Using Summary Versions

The use of summaries instead of full-length texts addresses several challenges:

Linguistic  Consistency: The  original  novels  contain  mixed  Spanish-Tagalog  content reflecting  the  linguistic  reality  of  late  19th-century  Philippines.  These  summaries,  written  in contemporary Filipino, provide linguistic consistency suitable for Tagalog-focused NLP processing using XLM-RoBERTa without requiring complex multilingual historical language handling.

Manageable  Computational  Scope: Summary  versions  offer  manageable  text  lengths (approximately 15,000-20,000 words combined) suitable for iterative development and testing within thesis timeframe constraints, while full texts would exceed 200,000 words.

Thematic Representation: Educational summaries retain core thematic content, narrative structure, and key passages that exemplify major themes, making them appropriate for thematic analysis while reducing computational overhead.

Educational Relevance: These summaries represent materials actually used by Filipino high school students studying Rizal's works, increasing the practical relevance of the retrieval system for its intended educational audience.

Domain-Adaptive Pre-training Basis: The focused corpus scope enables effective domain adaptation, where the system learns the semantic space of Rizal's novels specifically, facilitating semantic grounding validation for query rejection.

## 3.4.2 Thematic Data Sources

Themes for evaluation were derived from established literary analysis sources rather than automated discovery:

Noli  Me  Tangere  themes: Sourced  from  GradeSaver  (2024)  and  Francisco  (2025), including  themes  such  as  colonialism  ( kolonyalismo ),  education  ( edukasyon ),  religion  ( relihiyon ), social inequality ( hindi pagkakapantay-pantay sa lipunan ), and corruption ( katiwalian ).

El  Filibusterismo  themes: Sourced  from  LitCharts  (2025)  and  GradeSaver  (2023), including  themes  such  as  justice  ( katarungan ),  revolution  ( rebolusyon ),  violence  versus  reform ( karahasan laban sa reporma ), greed ( kasakiman ), and disillusionment ( pagkabigo ).

These predefined themes serve as queries for the hybrid retrieval system and provide the basis for evaluation, acknowledging the study's delimitation that it does not perform automated theme discovery. The themes also serve as domain markers for semantic grounding validation, establishing the expected semantic space of valid queries.

## 3.4.3 Domain-Adaptive Corpus Representation

Following  Krieger  et  al.  (2022),  the  corpus  is  represented  in  a  domain-adapted  semantic  space through:

Corpus Embedding Generation: All passages from both novels are encoded using XLMRoBERTa to create a 768-dimensional semantic representation of the domain.

Domain  Centroid  Computation: The  mean  embedding  vector  across  all  passages establishes the semantic center of the Rizal novels domain:

where is the set of all corpus passages and are passage embeddings.

Semantic  Boundary  Calibration: The  minimum  and  maximum  semantic  similarities between  the  domain  centroid  and  actual  passages  define  the  expected  range  of  valid  queries. Queries falling significantly below this range (by more than one standard deviation) are flagged for potential rejection.

This domain representation enables the system to distinguish between queries that belong to the novels' thematic space versus those that reference events, characters, or concepts not present in the corpus.

START

Load Raw Text Flles

Extract Chapters

Segment into Passages

Manual Thematic Annotation

Quality Check

Train-Test Split

Generate Embeddings

Store in Vector Database

End

## 3.4.4 Dataset Preparation and Annotation Process

Figure 5: Dataset Preparation and Annotation Process

<!-- image -->

## 3.5 Text Cleaning and Normalization

Initial preprocessing steps standardize the text:

HTML and Special Character Removal: Stripping any HTML tags, URLs, or non-textual elements from the web-scraped summaries.

Case Normalization: Converting all text to lowercase to standardize word forms and reduce vocabulary size, following standard information retrieval practices (AlShammari, 2023).

Punctuation Handling: Removing punctuation marks except sentence-terminating periods, which are retained for sentence segmentation purposes.

Whitespace Normalization: Collapsing multiple spaces into single spaces and removing leading/trailing whitespace.

## 3.5.1 Tagalog Stopword Management

Common  Tagalog  words  with  minimal  semantic  value  are  identified  using  the  stopwords-iso-tl package, which provides comprehensive coverage of approximately 200 Filipino/Tagalog stopwords including "ang," "mga," "si," "ng," "sa," "ay," "bilang," "ni," and "dahil."

The system implements a dual approach to stopword handling:

- For lexical scoring : Stopwords are assigned semantic weight of 0.0, effectively removing their influence on lexical matching
- Content words : Receive weights of 0.3-1.0 based on their frequency characteristics

This approach effectively minimizes stopword influence on lexical matching while maintaining  semantic  understanding  through  neural  embeddings.  Stopword  removal  has  been demonstrated to significantly improve Tagalog text processing performance, with Bation et al. (2017) achieving 92% accuracy in automatic document categorization when these stopwords were removed prior to feature extraction. Additionally, stopword removal can decrease corpus size by 35-45% while improving efficiency and accuracy (Ladani &amp; Desai, 2020). The system balances these benefits by removing stopwords for lexical matching while preserving them in neural embeddings to maintain grammatical and semantic context.

## Stopword Weighting Examples

Example 1: Query - "ang edukasyon ni Ibarra"

## Word-by-Word Weight Assignment:

Table 3: Word-by-Word Weight Assignment (Example 1)

| Word               | Type     | Weight   |   Contribution to Total |
|--------------------|----------|----------|-------------------------|
| ang                | Stopword | 0        |                     0   |
| edukasyon          | Content  | 0.7      |                     0.7 |
| ni                 | Stopword | 0        |                     0   |
| Ibarra             | Content  | 1        |                     1   |
| TOTAL QUERY WEIGHT | -        | -        |                     1.7 |

## Stopword Effect:

- Original query: 4 words
- Stopwords: "ang", "ni" (50% of query)
- Effective content: "edukasyon" + "Ibarra"

- Stopwords contribute 0.0 to lexical scoring
- Only content words (weight 1.7) determine lexical overlap

## Example 2: Query - "Edukasyon bilang pag-asa ni Ibarra"

## Word-by-Word Weight Assignment:

Table 4: Word-by-Word Weight Assignment (Example 2)

| Word               | Type     | Weight   |   Contribution to Total |
|--------------------|----------|----------|-------------------------|
| Edukasyon          | Content  | 0.7      |                     0.7 |
| bilang             | Stopword | 0.0      |                     0   |
| pag-asa            | Content  | 0.7      |                     0.7 |
| ni                 | Stopword | 0.0      |                     0   |
| Ibarra             | Content  | 1.0      |                     1   |
| TOTAL QUERY WEIGHT | -        | -        |                     2.4 |

## Stopword Effect:

- Original query: 5 words
- Stopwords: "bilang", "ni" (40% of query)
- Effective content: "Edukasyon" + "pag-asa" + "Ibarra"
- Stopwords contribute 0.0 to lexical scoring
- Only content words (weight 2.4) determine lexical overlap

These examples demonstrate how stopwords like "ang," "ni," and "bilang" receive zero weight in lexical matching, ensuring that only semantically meaningful terms contribute to relevance scoring, while the complete query with stopwords is preserved for XLM-RoBERTa semantic embeddings to maintain grammatical context.

## 3.5.2 Tokenization for XLM-RoBERTa

Text is tokenized using the pre-trained tokenizer associated with the xlm-roberta-base model via the SentenceTransformer wrapper. This tokenizer employs SentencePiece subword tokenization, which  handles  out-of-vocabulary  words  by  breaking  them  into  meaningful  subword  units.  This approach is essential for processing Filipino text, which may contain:

- Informal spellings
- Morphological variations not fully represented in the model's training vocabulary
- Hyphenated compounds (e.g., "pag-asa")

The tokenizer configuration matches that used during XLM-RoBERTa pre-training to ensure compatibility with the embedding model. The system uses a custom word pattern regex [0-9a-zAZÀ-ÿñÑ]+(?:-[0-9a-zA-ZÀ-ÿñÑ]+)* that preserves hyphenated tokens as single units, which is crucial for Filipino compound words.

## 3.5.3 Passage Segmentation Strategy

To enable passage-level retrieval as specified in the scope, the corpus is segmented into discrete passages. A passage is defined as a single sentence from the chapter summaries. This sentence-level granularity was chosen because:

- Summary sentences are typically substantive and thematically coherent
- Sentence boundaries provide natural semantic units
- Fine-grained retrieval enables precise theme localization
- Context  expansion  (Section  3.4.6)  can  aggregate  multiple  sentences  when  needed  for broader understanding

Each passage is indexed with metadata including:

- chapter\_number: Integer chapter identifier
- sentence\_number: Integer sentence identifier within chapter
- chapter\_title: String chapter title
- sentence\_text: String passage content
- sentence\_word\_count: Integer word count for length penalty computation

## 3.5.4 Deduplication

Following  preprocessing  and  segmentation,  near-duplicate  passages  are  identified  and removed to prevent redundant information from skewing retrieval results. The system implements global deduplication that tracks used passages across all retrieval operations:

Initial  Corpus  Deduplication: Near-duplicate  sentences  identified  using  TF-IDF  cosine similarity with threshold of 0.90. Only the first occurrence is retained.

Runtime  Deduplication: Once  a  passage  is  retrieved  for  a  query,  it  and  its  context sentences are marked as "used" (stored in self.used\_passages[book\_key] set) and excluded from subsequent results for that query. This ensures result diversity and prevents the same content from appearing multiple times in different forms.

Passage ID System: Each passage receives a unique identifier tuple (chapter\_number, sentence\_number) for efficient tracking.

This  strict  deduplication  addresses  redundancy  challenges  identified  in  the  literature (Hendrickx et al., 2009) and ensures clean, non-repetitive result sets.

## 3.5.5 Context Expansion with Neighbor Formula

For  retrieved  passages,  the  system  identifies  adjacent  sentences  that  provide  thematic context using Formula 2 (Neighbor Retrieval). Context expansion operates as follows:

Boundary  Detection: Identify  sentences  immediately  before  and  after  the  retrieved passage within the same chapter.

Neighbor Scoring: For each adjacent sentence, compute:

- Semantic similarity between main sentence and neighbor using XLM-RoBERTa embeddings
- Lexical similarity using Jaccard coefficient (intersection over union of words)
- Combined neighbor score using Formula 2:

Relevance  Filtering: Only  sentences  exceeding  relevance  threshold  (0.40  combined neighbor score) are included as context. This higher threshold (compared to 0.30 for main retrieval) ensures context sentences are genuinely relevant.

Distance Limitation: Expand in both directions without sentence limits, stopping early if:

- Irrelevant sentence encountered (breaks thematic continuity)
- Chapter boundary reached
- Previously used passage encountered (deduplication)

Metadata  Preservation: Track  which  context  sentences  are  thematically  relevant  vs. included for narrative continuity, along with their neighbor scores.

This context expansion enables users to understand how themes manifest in surrounding narrative, supporting close reading and interpretive analysis. The use of Formula 2 ensures context expansion maintains semantic coherence rather than including arbitrary adjacent text.

Layer 1: Data Input &amp;

Processing

Corpus

Preparation

Text

Preprocessing

Linguistic

Analysis

Query

Validation

Thematic

Classification

Layer 4: Thematic Analysis

&amp; Result Ranking

Context

Expansion

Layer 3: Hybrid

## 3.6 System Architecture

Semantic

Weighted

Result

Ranking

Score

The dual-formula hybrid information retrieval system integrates lexical and semantic retrieval components through a multi-stage architecture with two specialized scoring formulas and domainadaptive semantic validation. Module Layer 2: Embedding &amp; Representation Query Embedding Ranked Passages

Figure 6: System Architecture of the Hybrid Information Retrieval System Using XLM-RoBERTa for Rizal's Novels

<!-- image -->

## 3.6.1 Overall Architecture

The system comprises six interconnected modules:

Data Ingestion Module: Loads preprocessed passages from CSV files (noli\_chapters.csv, elfili\_chapters.csv,  noli\_themes.csv,  elfili\_themes.csv)  with  metadata  including  chapter  number, sentence number, chapter title, and sentence text.

Embedding Generation Module: Uses SentenceTransformer wrapper for XLM-RoBERTa (sentence-transformers/paraphrase-xlm-r-multilingual-v1) to generate contextual embeddings for all passages and themes during initialization. Embeddings are cached in memory for efficient retrieval. Combined text (chapter title + sentence text) is embedded for main retrieval. Domain centroid and semantic boundaries are computed for validation purposes.

Layer 5: User Interface

&amp; Results Display

Rich Console

Interface

Metrics

Display

## Query Processing Module:

- Validates queries for Filipino word content using wordfreq library
- Analyzes word frequencies and stopword ratios
- Assigns semantic weights to query words
- Generates query embeddings using XLM-RoBERTa

## Domain-Adaptive Semantic Validation Module:

- Computes semantic alignment between query and corpus domain
- Evaluates lexical presence of content words in corpus vocabulary
- Assesses thematic distance from predefined themes
- Applies rejection criteria for queries with insufficient domain grounding
- Returns explicit rejection message when query fails validation

## Dual-Formula Hybrid Retrieval Module:

- Executes parallel semantic and lexical retrieval using Formula 1 for main passage retrieval
- Computes multi-component scores with dynamic weights
- Applies deduplication filters
- Expands context using Formula 2 for neighbor relevance assessment

## Ranking and Presentation Module:

- Orders passages by Formula 1 hybrid relevance scores
- Expands context for top results using Formula 2
- Performs thematic classification

Input: User Query

- Formats results with visualization showing both formulas' contributions
- Displays rejection notifications for failed queries

## Hybrid Retrieval System Architecture

Vector Similarity (top-k passages)

Keyword Results (top-k passages)

Figure 7: Hybrid Retrieval System Architecture (Detailed Component View)

<!-- image -->

## 3.6.2 Query Validation Component

Before retrieval begins, queries undergo linguistic and domain-adaptive semantic validation to ensure they contain valid Filipino words and meaningful connection to the corpus domain.

## Phase 1: Linguistic Validation

Word Frequency Validation: Uses the wordfreq library to check if query words exist in Filipino language with minimum frequency threshold of . This threshold filters out random character sequences and non-Filipino words while accepting rare but valid terms.

Stopword Analysis: Computes the ratio of stopwords to total words. Since the stopword ratio is fixed at 1.0 (all words are stopwords), this metric becomes non-informative and is excluded from affecting Formula 1 retrieval weights.

Valid Word Ratio: Requires 100% of query words to be valid Filipino words. Queries failing this threshold are rejected with diagnostic feedback showing which words were not recognized.

Lexical  Grounding  Check: Validates  that  content  words  (non-stopwords)  exist  in  the corpus  vocabulary  built  from  both  novels  and  theme  files.  Queries  with  zero  lexical  grounding proceed to Phase 2 for semantic validation rather than immediate rejection.

## Phase 2: Domain-Adaptive Semantic Validation

Drawing  from  Krieger  et  al.  (2022),  this  validation  layer  implements  domain-specific semantic grounding assessment:

Domain Alignment Computation: Calculate maximum semantic similarity between query embedding and all corpus passage embeddings:

e T is

DomainAlignment (g, C) &lt; 0.30 1 ThemeProximity (g, T) &lt; 0.35

if DomainAlignment (g, C) &lt; 0.30, t

ThemeProximity (g, T) = max CosineSim(eg, et)

teT

Semantic Threshold Application: If , the query is considered semantically disconnected from the corpus domain.

Thematic Distance Assessment: Compute semantic similarity to all predefined themes:

where is the set of theme embeddings.

Combined Rejection Criteria: A query is rejected if:

This  ensures  that  queries  must  be  either  semantically  aligned  with  actual  passages  OR thematically related to known themes to proceed to retrieval.

Rejection Response: When a query fails validation, the system returns:

 "No thematically or semantically relevant passage found. The query does not align with the Krieger literary content of the novels."

 Along with diagnostic information:

- Maximum domain alignment score achieved
- Maximum theme proximity score achieved
- Content words that were lexically absent
- Suggestion to reformulate query or verify its relevance to the novels

## Example Validation Flow:

Query:

"Kamatayan ni Basilio" (Death of Basilio)

1. Linguistic Validation:

○

✓ PASS

All words valid Filipino

○

Content words: {kamatayan, Basilio}

○

Lexical grounding: "Basilio" present in corpus

Semantic Validation:

- ○

Domain alignment: 0.24 (&lt; 0.30 threshold)

○

Theme proximity: 0.28 (&lt; 0.35 threshold)

- Result:

✗ REJECT

System Response:

2.

3.

 No thematically or semantically relevant passage found.

Domain alignment: 24.0% (below 30.0% threshold)

Theme proximity: 28.0% (below 35.0% threshold)

The query may reference events or concepts not present in the novel summaries.



This  domain-adaptive  validation  prevents  the  system  from  hallucinating  results  about

Basilio's  death  (which  does  not  occur  in  the  summaries)  by  recognizing  the  semantic  mismatch despite lexical presence of "Basilio."

3.6.3 Multi-Sentence Query Validation Protocol

For  queries  containing  multiple  sentences  (detected  by  period-separated  segments),  the  system applies sequential validation:

89

Validation Sequence

:

1.

Parse input into sentence units using period (.) delimiter

2.

3.

Validate first sentence through complete validation pipeline:

○

Linguistic validation (Phase 1)

○

Domain-adaptive semantic validation (Phase 2)

Fail-Fast Mechanism

: If first sentence fails validation:

○

Halt processing immediately

○

Return error message:



"Sentence 1 is invalid due to [specific reason: semantic relation mismatch / domain misalignment /

lexical absence]"

○



Do NOT process subsequent sentences

If first sentence passes, proceed to validate second sentence using same criteria

4.

5.

Continue until all sentences valranidated or first failure encountered

Rationale

: This sequential fail-fast approach prevents merging valid and invalid concepts in multi- sentence  queries,  maintaining  result  integrity.

It  ensures  that  semantically  contradictory  or contextually inappropriate queries are rejected before retrieval, even when embedded within longer

multi-sentence input.

Example

:

Input:

"Kabaitan ni Padre Damaso. Edukasyon ni Ibarra."

90

## Processing :

●

Sentence 1: "Kabaitan ni Padre Damaso"

- ○

Semantic validation: Domain alignment = 0.19 (&lt; 0.30)

- Result:

○

✗ REJECT

Output: "Sentence 1 is invalid due to semantic relation mismatch. Padre Damaso is portrayed as antagonistic in the novel, contradicting the attribution of virtue

(kabaitan)."

Sentence 2 is NOT processed

●

This  protocol  implements  rigorous  semantic  grounding  consistent  with  the  domain-adaptive validation framework (Section 3.2.2).

3.6.4 Semantic Retrieval Component

The semantic component leverages XLM-RoBERTa to capture thematic similarity beyond exact keyword matching, implementing Cross-lingual Transfer Learning Theory. This component is

used in both Formula 1 and Formula 2, but with different weighting schemes.

## Model Selection:

The  system  uses  sentence-transformers/paraphrase-xlm-r-multilingual-v1,  a  fine-tuned variant of xlm-roberta-base (110M parameters, trained on 100 languages) optimized for semantic

similarity tasks. This model was chosen because:

●

Pre-trained on 100+ languages including Filipino

- Fine-tuned specifically for semantic similarity using sentence-level contrastive learning

91

- Computationally efficient for CPU-based inference
- No fine-tuning required, following validated approaches for Filipino low-resource contexts (Imperial, 2021)

Start

Load Pre-trained XLM - RoBERTa

Base Model

Prepare Training Dataset

Configure Training Parameters:

: song rio

Fine Tuning Process:

Model Validation

Meets threshold?

Yes

Save Fine-tuned Model

End

Figure 8: Model Training and Fine-Tuning Process

<!-- image -->

Embedding  Generation: Each  passage  and  query  is  encoded  into  a  768-dimensional dense vector representation. The encoding process:

where and are embedding vectors in .

For main retrieval, passages are encoded with combined text (chapter title + sentence text) to provide additional context. For neighbor retrieval, only sentence text is encoded.

CLS

Input Text: "Sample sentence from Rizal's novel"

## Sentence-Level Embedding Generation

"-.

<!-- image -->

Sample sentence

Tokenization from Rizal's novel

Pooling Strategy:

CLS Token

: MeLS Poling

Output: Dense Vector Embedding (768-dimensional representation)

Figure 9: Sentence-Level Embedding Generation

if word is a stopword eg•Ep

CosineSim (9, p) =

leglepl

Wword =

if frequency &gt; 0.001 (very common)

Similarity Measurement: Cosine similarity quantifies semantic relatedness:

Cosine similarity ranges from -1 to 1, with values closer to 1 indicating greater semantic alignment. A minimum semantic threshold of 0.20 filters out clearly irrelevant passages before further processing in Formula 1.

According to Jain et al. (2017), Cosine similarity is fundamental in Vector Space Models (VSMs), where documents are represented as term-weighted vectors in an n-dimensional space. Cosine effectively captures directional similarity between document-query vectors, quantifying their proximity in semantic space regardless of vector magnitude, thereby enhancing retrieval precision in hybrid IR systems.

## 3.6.5 Lexical Retrieval Component

The lexical component captures explicit keyword overlap using stopword-aware weighted term matching, preserving the precision advantages of exact-match retrieval while accommodating stopword influence. This component is used in both formulas but with different implementations.jaccard

Semantic Weight Assignment: Each word in the query receives a semantic weight based on its linguistic properties determined by the wordfreq library and stopword list:

0.3

0.7

mt

LexicalSimneighbor(m, n) =

Wt is the

Wm. is q npre

Wn is

Wm N Wnl

(Wm UWnl

LexicalOverlap(9, p) = Stegn Wr i

Eteq Wt

This  weighting  scheme  ensures  that  content  words  ( edukasyon , katiwalian , katarungan ) contribute more to lexical scores than grammatical stopwords ( ang , mga , sa , bilang , ni ).

Weighted  Lexical  Overlap  Score  (Formula  1): For  each  query-passage  pair  in  main retrieval:

where:

- is the semantic weight of term
- represents terms present in both query and passage
- The denominator normalizes by total query weight

Jaccard  Lexical  Similarity  (Formula  2): In  the  neighbor-to-neighbor  comparison  for context expansion, similarity is computed using the Jaccard Coefficient rather than weighted overlap. According  to  Jain  et  al.  (2017),  the  Jaccard  coefficient  measures  set-based  overlap  without considering term weights, making it particularly suitable for assessing lexical coherence between adjacent sentences where the focus is on shared vocabulary rather than weighted importance:

where:

- is the set of words in the main sentence
- is the set of words in the neighbor sentence
- Jaccard coefficient measures set overlap without weighted terms

(sentence\_length):

( (0.75, 0.25) if sentence length ≤ 5

(0.65, 0.35)

(Alex, Asem) =

if 5 &lt; sentence\_length ≤ 10

if 10 &lt; sentence length ≤ 15

The  choice  of  Jaccard  over  weighted  overlap  for  neighbor  comparison  is  theoretically grounded: context expansion prioritizes lexical continuity and vocabulary cohesion rather than term importance hierarchies. Both Cosine and Jaccard similarity measures are fundamental in Vector Space  Models  (VSMs),  where  documents  are  represented  as  term-weighted  vectors  in  an  ndimensional  space  (Jain  et  al.,  2017).  These  measures  effectively  quantify  document-query proximity,  with  Cosine  capturing  directional  similarity  in  weighted  semantic  space  and  Jaccard emphasizing set-based overlap in unweighted lexical space-together enhancing retrieval precision and recall in hybrid IR systems.

## 3.6.6 Dual-Formula Hybrid Integration

The two retrieval components operate in parallel, with their outputs combined through two specialized  formulas  optimized  for  different  retrieval  contexts,  directly  addressing  the  research objectives.

## Formula 1: Main Retrieval (Query→Corpus)

Purpose: Retrieve  passages directly relevant to user queries with dynamic balancing of lexical precision and semantic understanding.

Dynamic Weight Computation: Weights are computed as a function of sentence (query) length ( ):

## Final Score Computation (Formula 1):

## Score Normalization:

Rationale: Longer queries provide more semantic context, justifying higher semantic weight. Short queries (1-2 words) rely more on lexical precision. Since stopwords are completely removed, the system focuses on semantically meaningful terms to enhance discriminative power and retrieval accuracy.

## Formula 2: Neighbor Retrieval (Sentence→Sentence)

Purpose: Evaluate the contextual coherence between the main retrieved sentence and its neighboring sentences (previous or next). This process ensures that context expansion maintains both semantic continuity and lexical consistency

Word Weight Assignment: Unlike Formula 1's query-level dynamic balancing, Formula 2 now applies token-level dynamic weighting based on word frequency to better capture thematic relevance between sentences. :

empeddings.

if neighbor score ≥ 0.40

neighbor\_score = (Asem × neighbor semantic) + (Nex × neighbor lexical)

neighbor\_score - neighbor\_lexical\_

Asem — is the weight ass

Alex — is the weight a

(0.30, 0.70) if length ratio ≥ 1.5 (much longer)

neighbor \_semantic -

(Alex, Asem) =

<!-- formula-not-decoded -->

## Final Score Computation (Formula 2):

where:

- - represents the overall similarity score between the main retrieved sentence and its neighboring sentence.
- - is the weight assigned to the semantic similarity component.
- - is the semantic similarity score between the main and neighboring sentence, computed using cosine similarity between their sentence embeddings.
- - is the weight assigned to the lexical similarity component.
- - is the lexical similarity score between the main and neighboring sentence, calculated using the Jaccard coefficient as defined in Section 3.5.4.

Relevance Threshold: Neighbors are included as context only if:

Rationale: Context expansion relies on maintaining strong semantic coherence to preserve the thematic flow between adjacent sentences. Instead of fixed proportional weighting, Formula 2 uses  dynamic  frequency-based  weighting  to  automatically  emphasize  rare  and  contextually significant  words.  The  Jaccard coefficient remains integral for measuring lexical continuity, while

adaptive word weights ensure that meaningful terms contribute more to the overall score than highfrequency or stopword terms. This approach ensures that neighboring sentences are selected based on conceptual relevance rather than mere word overlap, with the higher relevance threshold (0.40 vs. 0.30) filtering out loosely related or tangential content.

## 3.6.7 Formula Application Example

To illustrate how the dual formulas operate in practice, consider the query "Edukasyon bilang pag-asa ni Ibarra" (Education as Ibarra's hope):

## Step 1: Query Validation

## Linguistic Validation:

- Total words: 5 (Edukasyon, bilang, pag-asa, ni, Ibarra)
- Content words: 3 (Edukasyon, pag-asa, Ibarra)
- Stopwords: 2 (bilang, ni)
- Stopword ratio: 40.0%
- ●
- Valid word ratio: 100% ✓ PASS

## Semantic Validation:

- Domain alignment: (&gt; 0.30 threshold) ✓ PASS
- Theme proximity: 0.794 (&gt; 0.35 threshold) ✓ PASS
- Query proceeds to retrieval

:e L = 5(1

id Tstop = 0.40 &lt; 1.0

(Asem, Alex) = (0.75, 0.25)

y: SemanticSim (g, P) = 0.499(

Semantic weights: Wedukasyon = 0.7, Wpag-asa = 0.7

, Wibarra = 1.0

## Step 2: Formula 1 Weight Calculation

1.7

Since (counting content words) and : = 0.708 2.4

No stopword adjustment needed (stopword ratio below 1.0 threshold).

## Step 3: Main Retrieval Scoring

For the top result (Chapter 19, Sentence 5):

Passage text: "Nagtanong si Ibarra kung ano ang kinahinatnan ng pagtulong ng ama sa mga mahihirap. Sinabi ng guro na nakatulong si Don Rafael sa pagpapaunlad ng edukasyon sa kanilang bayan."

<!-- formula-not-decoded -->

- Computed via cosine similarity between XLM-RoBERTa embeddings
- High similarity captures conceptual alignment (Ibarra + education + help)

## Lexical overlap:

- Matched content words: {edukasyon, ibarra}
- Semantic weights: , ,
- Total query weight:
- Matched weight: (edukasyon and ibarra found in passage)
- (70.8%)

## Formula 1 final score:

Final score:

55.1% (shown as "Final" in output)

Step 4: Context Expansion with Formula 2

For the previous sentence (Sentence 4):

Main sentence (S5):

"Sinabi ng guro na nakatulong si Don Rafael sa pagpapaunlad ng edukasyon sa kanilang bayan."

Neighbor  sentence  (S4):

"Ayon  sa  guro  ay  wala  dapat  itong  ipagpasalamat  sapagkat malaki ang utang na loob nito kay Don Rafael dahil isa ito sa mga nabigyan ng tulong nung ito'y

nabubuhay pa."

Semantic similarity (sentence-to-sentence): (80.0%)

## Lexical similarity (Jaccard):

- Main  words:  {sinabi,  guro,  nakatulong,  don,  rafael,  pagpapaunlad,  edukasyon,  kanilang, bayan}
- Neighbor words: {ayon, guro, wala, dapat, itong, ipagpasalamat, sapagkat, malaki, utang, loob, nito, kay, don, rafael, dahil, isa, ito, mga, nabigyan, tulong, nung, nabubuhay, pa}
- Intersection: {guro, don, rafael} = 3 words (after stopword filtering)
- Union: 26 unique words

<!-- formula-not-decoded -->

## Formula 2 neighbor score:

Neighbor score:

58.4% (exceeds 0.40 threshold → included as context)

The system continues this process for all adjacent sentences, stopping when:

- Neighbor score falls below 0.40 threshold
- Maximum distance of 5 sentences reached
- Chapter boundary encountered

Step 5: Result Display

## The output shows:

 MAIN RETRIEVAL (Query→Corpus): λ\_lex=0.25 | λ\_sem=0.75

NEIGHBOR RETRIEVAL (Sentence→Sentence): λ\_lex=0.35 | λ\_sem=0.65

Result 1:

Semantic: 49.9% | Lexical: 70.8% | Final: 55.2%

Neighbor Similarity Metrics:

Previous S4: Semantic: 80.0% | Lexical: 18.2% | Combined: 58.4%

 This example demonstrates how:

- Query passed both linguistic and semantic validation
- Formula 1 dynamically balanced semantic (75%) and lexical (25%) for the 5-word query
- Formula 2 used dynamic weights (65% semantic, 35% lexical) to assess context coherence
- The neighbor achieved high relevance (58.4%) despite low lexical overlap (18.2%) because of strong semantic coherence (80.0%)
- Jaccard coefficient appropriately measured  vocabulary continuity between  adjacent sentences

## 3.6.8 Query Suggestion Component

To  assist  users  in  formulating  effective  queries,  particularly  students  unfamiliar  with  thematic terminology, the system implements intelligent query suggestions for incomplete inputs.

## 3.6.8.1 Single-Word Theme Input

When a user types a single thematic term (e.g., "edukasyon"), the system:

## Theme Detection and Character Association

- Theme Recognition : Identifies the input as a thematic keyword using:
- Comparison against predefined theme vocabulary (Section 3.3.3)
- Semantic similarity to theme embeddings (threshold ≥ 0.60)
- Character Co-occurrence Mining : Extracts characters most frequently associated with the theme across the corpus using:
- Named Entity Recognition (NER) via spaCy to identify PROPN entities

- Co-occurrence frequency within same passages
- Semantic proximity in embedding space
- Relational Marker Selection : Determines appropriate possessive/relational marker based on grammatical context:
- "ni" (personal possessive): For individual characters
- "ng" (general possessive): For groups, institutions, or abstract entities
- "sa" (locational/indirect): For place-based or indirect relationships
- Template Construction : Generates suggestions using pattern:

##  [THEME] + [MARKER] + [CHARACTER/ENTITY]



## Example: Suggestions for "edukasyon"

Input:

"edukasyon"

## System Process:

- Detects "edukasyon" as theme (matches theme: "Edukasyon")
- Extracts co-occurring entities from corpus:
- Characters: {Ibarra, Don Rafael, mga estudyante, guro}
- Institutions: {paaralan, mga prayle, San Diego}
- Computes co-occurrence frequencies and semantic similarities
- Selects appropriate grammatical markers

## Generated Suggestions (ranked by relevance):

1. "edukasyon ni Ibarra" (ni - personal possessive for main protagonist)
2. "edukasyon ni Don Rafael" (ni - personal possessive for Ibarra's father)
3. "edukasyon ng mga estudyante" (ng - general possessive for group)
4. "edukasyon sa San Diego" (sa - locational marker for place)
5. "edukasyon ng mga prayle" (ng - possessive for institution/group)

## 3.6.8.2 Single-Word Character Input

When a user types a single character name (e.g., "Basilio"), the system:

## Character Detection and Theme Association

- Entity Recognition : Identifies the input as a character name using:
- Part-of-speech tagging (PROPN) via spaCy
- Matching against character registry extracted from corpus
- Thematic Action Extraction : Extracts themes and actions associated with the character using:
- spaCy dependency parsing to find verbs and nouns co-occurring with the character
- Semantic clustering of extracted terms into thematic categories
- Grammatical Marker Application : Always uses "ni" for personal character possession
- Template Construction : Generates suggestions using pattern:

 [THEME/ACTION] + "ni" + [CHARACTER]



Example: Suggestions for "Basilio"

Input:

"Basilio"

## System Process:

●

●

●

●

Detects "Basilio" as character entity (POS: PROPN)

Extracts co-occurring thematic terms: {kabayanihan, pag-aaral, buhay, paghihirap, pamilya}

Ranks by embedding similarity to character's narrative context

Applies "ni" marker consistently

Generated Suggestions (ordered by similarity score):

1.

2.

3.

4.

5.

"kabayanihan ni Basilio"

"pag-aaral ni Basilio"

"buhay ni Basilio sa San Diego"

"paghihirap ng pamilya ni Basilio" (note: "ng pamilya" for group possession)

"pagtulong ni Basilio"

3.6.8.3 Entity-Based Chapter Location with Domain-Adaptive Semantic Grounding

When  a  user  enters  a  standalone  entity  without  thematic  context  (e.g.,  "Sisa",  "San  Diego",

"paaralan"), the system maps its location across both Noli Me Tangere and El Filibusterismo using domain-adaptive semantic validation.

108

## Entity Detection and Validation

The system uses spaCy to identify PROPN (proper nouns) and NOUN (common nouns) as entity queries. Before mapping, entities undergo semantic validation by computing maximum cosine similarity between entity embedding and corpus passages. Validation requires semantic alignment threshold of 0.35, at least 2 corpus occurrences, and 5 surrounding content words. Invalid entities receive rejection messages with diagnostic information.

## Chapter Mapping Process

Prominence Calculation : For each chapter containing the entity, prominence is calculated by  combining  mention  count  (40%  weight)  with  average  semantic  relevance  (60%  weight). Prominence classifies as Primary when score is 0.70 or above, Secondary for scores between 0.450.69, or mentioned for scores between 0.20-0.44.

Cross-Novel  Mapping :  The  system  identifies  which  novel(s)  contain  the  entity,  lists  all chapters where it appears, determines the narrative span (first to last chapter), and identifies the peak prominence location.

## Contextual Information

For each chapter location, the system establishes thematic connections for themes scoring threshold  of  0.45  or  above.  Themes  are  derived  from  predefined  literary  sources  including colonialism  (kolonyalismo),  education  (edukasyon),  religion  (relihiyon),  social  inequality  (hindi pagkakapantay-pantay sa lipunan), justice (katarungan), revolution (rebolusyon), violence versus reform (karahasan laban sa reporma), and greed (kasakiman).

Display Format

Output shows entity name with semantic alignment score, separate sections for Noli Me

Tangere and El Filibusterismo  indicating  presence  or  absence,  chapter  listings  with  prominence levels and thematic connections, and related search suggestions.

Application Examples

Example 1: Character Search - "Sisa"

ENTITY: Sisa (Character)

Semantic Alignment: 78.0%

📖📖

✓ Validated

NOLI ME TANGERE

Narrative Span: Chapters 6-20

Total Mentions: 47

Chapter 6: Si Kapitan Tiyago [Secondary - 0.64]

• Themes: Pamilya (Family), Kahirapan (Poverty)

Chapter 15: Ang mga Sakristan [Primary - 0.72]

110

- Themes: Katiwalian (Corruption), Karahasan (Violence)

Chapter 17: Ang Mamatay na Ina [Primary - 0.89]

• Themes: Pagdurusa (Suffering), Pang-aapi (Oppression)

Chapter 20: Si Sisa [Primary - 0.85]

- Themes: Karahasan sa Pamilya (Family Violence)

- [ ] 📖📖 EL FILIBUSTERISMO

- Does not appear

Related Searches: "Pagdurusa ni Sisa" | "Karahasan sa pamilya"

Example 2: Concept Search - "edukasyon"

ENTITY: edukasyon (Concept)

Semantic Alignment: 85.0% ✓ Validated

- [ ] 📖📖 NOLI ME TANGERE

```
Narrative Span: Chapters 7-55 Total Mentions: 89 Chapter 7: Ang Hapunan [Secondary - 0.68] · Themes: Kolonyalismo (Colonialism), Edukasyon (Education) Chapter 18: Ang mga Batang Tao [Primary - 0.92] · Themes: Edukasyon (Education), Pag-asa (Hope) Chapter 23: Ang Pagtitipon [Primary - 0.87] · Themes: Relihiyon (Religion), Kapangyarihan ng Simbahan 📖📖 EL FILIBUSTERISMO Narrative Span: Chapters 3-14 Total Mentions: 34
```

Chapter 3: Ang mga Makapangyarihan [Secondary - 0.63]

• Themes: Katiwalian (Corruption), Sistema ng Pamahalaan

Chapter 10: Yaman at Karalitaan [Mentioned - 0.42]

• Themes: Hindi Pagkakapantay-pantay (Social Inequality)

Related Searches: "Paaralan ni Ibarra" | "Kolonyalismo sa edukasyon"

System Integration

This mapping component extends the query suggestion architecture by applying domain-adaptive validation with semantic alignment threshold of 0.35. It uses the main retrieval formula for entity-

passage matching while leveraging thematic classification with similarity threshold of 0.45 to enrich contextual  information.  The  implementation  provides  systematic  access  to  narrative  element

locations, addressing student difficulties in tracking entities across both novels by showing where characters,  places,  objects,  and  concepts  appear  with  their  associated  themes  from  established

literary sources.

3.7 Thematic Classification

Retrieved passages undergo thematic classification to identify which predefined themes they exemplify, addressing the research objectives.

113

Start

## Thematic Classification and Analysis Process

Retrieve Relevant Passages

Extract Thematic Features

Apply Thematic Categories

Calculate Theme Relevance

Scores

Generate Theme Distribution

Visualize Results

End

Figure 10: Thematic Classification and Analysis Process

<!-- image -->

etheme = XLM-RoBERTa(theme\_text)

theme text = Tagalog Title + + Meaning

Stheme (p, t) = CosineSim(ep, et)

## Theme Matching Process:

Theme Encoding: Each predefined theme is represented as concatenated text:

Example: "Edukasyon Sa mga mata ni Ibarra, ang kaalaman ay sagisag ng pag-asa, habang sa mga prayle, ito'y panganib sa kanilang kapangyarihan"

Theme Embedding: Generate embeddings for all themes:

Passage-Theme Similarity: For each retrieved passage, compute similarity to all themes:

Theme  Assignment: Themes  with are  assigned  to  the  passage.  The highest-scoring theme becomes the "primary theme."

## Thematic Coverage Metrics:

Thematic Coverage: Proportion of retrieved passages with at least one assigned theme

Average Theme Confidence: Mean similarity score across assigned themes

Classification  Type: Results  classified  as  "thematic  analysis"  if  coverage and average confidence , otherwise "semantic search"

Example from "Edukasyon bilang pag-asa ni Ibarra" query:

e ≥ 0.45, 0

e ≥ 30% a h Stheme ≥ 0.45 a

For Result 1 (Chapter 19, Sentence 5):

- Passage embedding computed from retrieved sentence
- Compared against all theme embeddings
- Primary theme: "Edukasyon" with confidence 79.4%
- This indicates strong thematic alignment between retrieved passage and education theme

This  thematic  layer  enables  analysis  beyond  passage  retrieval,  identifying  which  literary themes are present and how they distribute across the novels.

## 3.8 Evaluation Methodology

The evaluation strategy combines quantitative performance metrics with qualitative expert assessment to comprehensively evaluate the dual-formula system's effectiveness, directly addressing the research objectives.

Scorebaseline (9, p) = LexicalOverlap(g,p)

Evaluation Framework

## Evaluation Metrics and Framework

Retrieval Performance

Precision

Mean Average

Prediction (MAP)

Figure 11: Evaluation Metrics and Framework

<!-- image -->

## 3.8.1 Baseline Comparison

System performance is evaluated against a keyword-only baseline that uses only the lexical overlap component without semantic understanding:

This  baseline  represents  traditional  Ctrl+F  keyword  search  approaches  criticized  in  the problem statement. The comparison demonstrates the incremental value provided by:

- Semantic understanding via XLM-RoBERTa embeddings
- Hybrid integration via Formula 1

1k kis

Retrieved n Relevant

Precision • Recall

Mais y 9,

MAP =

Precision =

Recall - Retrieved n Relevant

F1 = 2•

1

mg where Ql is

Precision + Recall

Id rel (k) is 1 if the

Retrieved|

Relevant q=1

IQ|

\_Precision@k • rel(k)

- Context expansion via Formula 2
- Domain-adaptive semantic validation

## 3.8.2 Quantitative Evaluation Metrics

Precision  and  Recall: For  each  theme  query,  a  relevance  judgment  set  is  created  by manually annotating passages as relevant or non-relevant to that theme.

F1-Score: The harmonic mean of precision and recall:

Mean Average Precision (MAP): To account for ranking quality:

where is the number of queries, is the number of relevant documents for query , and is 1 if the document at rank is relevant, 0 otherwise.

## Domain Validation Metrics:

True Negative Rate (Specificity): For known-negative queries (queries about events/characters not in the novels):

TNR =

Correct Rejections

Incorrect Retrievals

Correct Retrievals nly: Asem = 1, Alex = 0 (Formula 1)

Total Negative Queries

FPR =

= 1 - TNR

Total Negative Queries

Precisionvalid = Total Valid Queries vith static weights: Asem = 0.5, Alex = 0.5(

vithout length penalty: Y = 0

False Positive Rate: Proportion of invalid queries that incorrectly return results:

Precision at Valid Queries: For queries that should return results:

## Component Contribution Analysis (Ablation Studies):

## Compare performance across:

1. Semantic-only: (Formula 1)
2. Lexical-only: (Formula 1)
3. Formula 1 with static weights: (no dynamic adjustment)
4. Formula 1 without length penalty:
5. Context expansion using Formula 1 instead of Formula 2
6. Without  domain  validation: All  queries  proceed  to  retrieval  regardless  of  semantic grounding
7. Full dual-formula model: Dynamic Formula 1 + Dynamic Formula 2 + Domain validation

This  quantifies  each  component's  individual  contribution  to  overall  performance  and validates the effectiveness of using two specialized formulas with domain-adaptive validation.

## 3.8.3 Qualitative Expert Review

Two to three literature educators or scholars with expertise in Rizal's works evaluate retrieval quality.

## Evaluation Protocol:

Query Selection:

Experts presented with:

●

5 valid thematic queries (3 from Noli Me Tangere, 2 from El Filibusterismo)

●

3 known-negative queries (about events/characters not in the novels)

Result Presentation:

For each valid theme, top 10 retrieved passages from both hybrid and baseline systems shown in random order without system identification.

Relevance Rating:

Experts rate each passage on 5-point Likert scale:

●

1 = Not Relevant

- 2 = Slightly Relevant
- 3 = Moderately Relevant
- 4 = Highly Relevant
- 5 = Perfectly Relevant

## Rejection Assessment: For known-negative queries, experts evaluate:

- Appropriateness of rejection (should the system have been rejected?)
- Quality of rejection explanation
- Whether any returned results (if system failed to reject) are actually relevant

120

Qualitative Feedback:

Experts provide written comments on:

●

●

●

●

Thematic alignment quality

Context coherence (relevance of neighbor sentences)

Effectiveness of domain validation in preventing false results

Overall system utility for literary analysis

Analysis:

Expert  ratings  aggregated  to  compute  mean  relevance  scores.  Inter-rater reliability  assessed  using  Krippendorff's  alpha  (

indicates  acceptable  agreement).

Qualitative  comments  analyzed  thematically  to  identify  system  strengths  and  weaknesses, particularly regarding:

●

Formula 2's effectiveness in selecting coherent context

●

●

Domain validation's success in preventing semantic hallucination

Balance between rejection strictness and retrieval coverage

## 3.8.4 Thematic Distribution and Co-occurrence Analysis

To address the research objectives, the study analyzes how themes distribute across and co-occur within both novels.

## Distribution Analysis:

For  each  theme,  identify  all  passages  scoring  above  relevance  threshold  (0.45  theme similarity).

Count theme frequency per chapter.

121

## Visualize using:

- Bar charts showing theme frequency per novel
- Line graphs showing theme concentration across narrative progression
- Heatmaps showing chapter-level theme prevalence

## Co-occurrence Analysis:

Identify passages where multiple themes simultaneously score above threshold.

Construct co-occurrence matrix: = count of passages containing both theme   and theme

## Visualize using:

- Heatmaps of pairwise theme co-occurrence frequencies
- Network graphs (nodes = themes, edges = co-occurrence strength)
- Chord diagrams for multi-theme relationships

## 3.8.5 Query Complexity Testing

To  comprehensively  evaluate  the  system's  capability  to  handle  diverse  user  input  patterns,  the evaluation includes queries of varying complexity levels representative of actual student usage:

Start

Define Test Cases

Prepare Test Queries

(Filipino &amp; English)

Execute System Tests

Collect Performance Metrics

Optimize System

Compare with Baseline Systems

Statistical Analysis

Performance

Acceptable?

Yes

User Acceptance Testing (UAT)

Document Results

End

## System Testing and Validation Workflow

Figure 12: System Testing and Validation Workflow

<!-- image -->

Identify Issues

## Simple Character-Action Queries (Basic Relational Patterns)

These represent typical high-school level searches combining a character with thematic verbs or nouns:

- "edukasyon ni Ibarra"
- "pagmamahal ni Maria Clara"
- "kamatayan ni Basilio"
- "kabaitan

ni

Padre

Salvi"

These  inputs  test  whether  the  system  understands  basic  relational  patterns  commonly  used  by students in literary analysis.

## Multi-Concept &amp; Multi-Relation Queries (Complex Semantic Patterns)

These queries involve multiple themes, characters, and emotional or moral relationships requiring deeper semantic validation:

- "kabaitan ni Don Rafael sa mga dukha"
- "galit at kasalanan ni Padre Damaso kay Ibarra"
- "pagdurusa ni Sisa dahil sa kawalan ng hustisya"
- ●
- "pang-aapi ng mga prayle sa pamilya ni Ibarra"

## These complex samples test the system's ability to detect:

- Semantic contradictions (e.g., "kabaitan ni Padre Damaso")

- Multi-theme coherence across character relationships
- Cross-character relational dynamics
- Correctness based on canonical novel context

The domain-adaptive semantic validation (Section 3.2.2) is specifically designed to reject queries containing semantic contradictions (e.g., ascribing virtue to antagonistic characters) while accepting valid multi-concept queries that reflect actual narrative relationships.

## Statistical Testing:

Test whether theme co-occurrence exceeds random chance:

where is  observed co-occurrence  and is expected co-occurrence under independence assumption.

This  analysis  reveals  thematic  patterns  potentially  obscured  in  traditional  close-reading, providing empirical evidence for claims about thematic interconnections in Rizal's novels.

## 3.9 Implementation Details

## 3.9.1 Development Environment and Tools

Programming Language:

Python 3.9+

## Key Libraries:

- sentence-transformers (4.0+): XLM-RoBERTa model loading and inference

- scikit-learn (1.3+): Cosine similarity computation, evaluation metrics

- pandas (2.0+): Data manipulation and passage metadata management

- numpy (1.24+): Numerical computations for scoring functions

- stopwordsiso (0.6+): Official Tagalog stopwords from ISO 639

- wordfreq (3.0+): Word frequency validation for Filipino

- rich (13.0+): Terminal-based result visualization with dual-formula metrics display

Hardware: Development  conducted  on  standard  CPU-based  hardware  (8GB  RAM minimum).  XLM-RoBERTa  base  model  inference  is  computationally feasible  without  GPU acceleration for the corpus size used.

## 3.9.2 Data Storage Format

## Chapters CSV Structure:

- chapter\_number: Integer chapter identifier
- chapter\_title: String chapter title
- sentence\_number: Integer sentence identifier within chapter
- sentence\_text: String passage content

## Themes CSV Structure:

- Tagalog Title: String theme name in Filipino
- Meaning: String theme description/explanation
- Novel: String identifier (Noli/Fili)

## 3.9.3 System Workflow

## Initialization Phase (Offline):

1. Load noli\_chapters.csv, elfili\_chapters.csv, noli\_themes.csv, elfili\_themes.csv
2. Initialize XLM-RoBERTa model via SentenceTransformer wrapper
3. Load official Tagalog stopwords from stopwords-iso
4. Generate and cache embeddings for all passages (combined chapter title + sentence text) and themes

5. Compute domain centroid and semantic boundaries from corpus embeddings
6. Compute sentence word counts for length penalty
7. Build corpus vocabulary from passages and themes for lexical grounding validation

Start

## Query Phase (Online):

Execute Semantic Search

7

Figure 13: Query Processing Execution Workflow

<!-- image -->

## Step 1: Receive and Validate Query

- Receive thematic query from user
- Phase 1 Validation (Linguistic):
- Check for valid Filipino words using wordfreq
- Verify basic lexical presence in corpus vocabulary
- Compute stopword ratio
- If fails: Return linguistic validation error

## Step 2: Domain-Adaptive Semantic Validation

- Generate query embedding using XLM-RoBERTa
- Compute domain alignment:
- Compute theme proximity:
- Apply rejection criteria:
- If : REJECT
- Return: "No thematically or semantically relevant passage found."
- If passes: Proceed to retrieval

## Step 3: Query Analysis

- Extract words using regex pattern (preserving hyphens)
- Identify stopwords using stopwords-iso
- Compute word frequencies
- Assign semantic weights to each word
- Calculate query length and stopword ratio

## Step 4: Execute Formula 1 (Main Retrieval)

- Compute semantic similarities for all passages (filter by 0.20 threshold)
- Compute weighted lexical overlap scores
- Calculate dynamic weights based on query length and stopword ratio
- Compute multi-component final scores using Formula 1
- Sort by final score
- Apply ranking constraints:
- Maximum 3 passages per chapter (diversity)
- Mark used passages for deduplication

## Step 5: Execute Formula 2 (Context Expansion)

- For each top-k passage, identify adjacent sentences
- Compute semantic similarity (sentence-to-sentence)
- Compute lexical similarity using Jaccard coefficient
- Calculate neighbor scores using Formula 2 weights
- Include neighbors exceeding 0.40 threshold

- Stop expansion at irrelevant sentence or chapter boundary
- Mark context sentences as used

## Step 6: Perform Thematic Classification

- Compare passages to theme embeddings
- Assign themes exceeding 0.45 similarity threshold
- Identify primary themes

## Step 7: Format and Display Results

- Show dual-formula metrics (Formula 1 weights, Formula 2 scores)
- Display neighbor similarity breakdown with Jaccard lexical scores
- Present thematic analysis
- For  rejected  queries:  Show  diagnostic  information  (domain  alignment,  theme  proximity, rejection reason)

## Evaluation Phase:

1. Execute retrieval for all test themes (valid and known-negative)
2. Record domain validation decisions (accept/reject)
3. Compare results against manual relevance judgments
4. Compute precision, recall, F1, MAP for each configuration
5. Compute TNR, FPR for domain validation
6. Conduct ablation studies (semantic-only, lexical-only, single formula, without validation, etc.)
7. Facilitate expert review sessions with randomized result presentation
8. Generate distribution and co-occurrence visualizations

9. Perform statistical testing on theme patterns

## 3.9.4 System Parameters

## Domain Validation:

- Domain alignment threshold: 0.30

- Theme proximity threshold: 0.35

- Combined rejection rule: Both thresholds must fail

## Main Retrieval (Formula 1):

- Minimum semantic threshold: 0.20
- Short sentence threshold: 5 words
- Short sentence penalty: 0.08
- High stopword ratio threshold: 1.0
- Length penalty coefficient ( ): 1.0

## Neighbor Retrieval (Formula 2):

- Semantic weight ( ): Dynamic, adjusts based on contextual relevance
- Lexical weight ( ): Dynamic, adjusts based on corpus frequency distribution
- Lexical similarity metric: Jaccard coefficient
- Neighbor relevance threshold: 0.40
- No limit on context expansion in either direction

## Thematic Classification:

●

Theme assignment threshold: 0.45

●

Thematic coverage threshold: 30%

●

Average confidence threshold: 0.45

## General:

●

Valid Filipino word frequency threshold:

●

●

Valid word ratio requirement: 100%

Maximum passages per chapter: 3

- Top-k results returned: 9

## 3.10 Ethical Considerations

Data Sources:

The corpus consists of publicly available educational summaries of classic literature in the public domain. No copyrighted material is reproduced beyond fair use for academic

research purposes.

Attribution:

All theoretical frameworks, methodologies, and external sources are properly cited following academic standards.

Expert Review Ethics:

Participating literature educators/scholars provide informed consent and are informed that evaluations are for research purposes. No personally identifiable information

is  collected beyond expertise credentials. Participation is voluntary and can be withdrawn at any time.

134

Reproducibility: The methodology is documented in sufficient detail to enable replication. Code implementation will be made available upon reasonable request to support future research, subject to ethical review.

Cultural  Sensitivity: The  study  recognizes  José  Rizal's  novels  as  culturally  significant works in Philippine national identity. The research aims to enhance accessibility and understanding of  these  works  rather  than  replace  human  literary  scholarship  or  impose  external  interpretive frameworks.

Bias  Mitigation: The  dual-formula  hybrid  system  with  domain-adaptive  validation  is designed to minimize algorithmic bias by:

- Using multilingual models pre-trained on diverse corpora
- Incorporating multiple relevance signals (semantic + lexical) to reduce dependence on any single representation
- Implementing domain validation to prevent semantic hallucination and false results
- Validating results through expert review by Filipino literature specialists
- Transparently documenting system limitations and delimitations
- Making dual-formula scoring process interpretable and explainable

Query  Rejection  Transparency: The  domain-adaptive  validation  mechanism  explicitly informs users when queries are rejected and why, preventing confusion and maintaining trust in system outputs.

## 3.11 Limitations and Delimitations

## 3.11.1 Methodological Limitations

Use  of  Summary  Versions: While  summaries  preserve  core  thematic  content,  they necessarily omit narrative details, character development nuances, and linguistic subtleties present in full texts. Retrieval results reflect themes as represented in summaries rather than original novels. However,  these  summaries  represent  actual  educational  materials  used  by  Filipino  students, increasing practical relevance.

No Fine-tuning: The XLM-RoBERTa model is used without fine-tuning on Filipino literary texts  due  to  lack  of  annotated  training  data.  While  cross-lingual  transfer  enables  effective performance  (Imperial,  2021),  fine-tuning  could  potentially  improve  semantic  understanding  of Filipino literary language and archaic expressions.

Manual  Relevance  Judgments: Quantitative  evaluation  requires  manual  annotation  of passage relevance, which is inherently subjective despite use of established theme definitions. This limitation is partially mitigated through:

- Use of multiple annotators with inter-rater reliability assessment
- Grounding annotations in established literary sources (GradeSaver, LitCharts)
- Supplementing quantitative metrics with qualitative expert review

Limited Corpus Scope: Evaluation  focuses on two novels only. Generalization to other Filipino literary works (e.g., Ibong Adarna , Florante at Laura ) requires additional validation. The dualformula methodology is designed to be adaptable, but performance may vary with different writing styles, time periods, and linguistic registers.

Context Window Limitations: XLM-RoBERTa has a 512-token maximum input length. For longer  passages  (rare  in  summaries  but  possible),  this  may  affect  embedding  quality.  However, summary sentences typically remain well within limits.

Formula Parameter Selection: The weighting behavior and thresholds in Formula 2 were determined empirically through pilot testing to optimize sentence-to-sentence coherence. Instead of fixed  semantic-lexical  ratios,  the  system  employs  dynamic  frequency-based  weights  that  adjust according to corpus characteristics. Different literary datasets or retrieval objectives may still benefit from tuning frequency thresholds or relevance cutoffs to achieve optimal contextual performance.

Domain Validation Threshold Sensitivity: The domain alignment threshold of 0.30 was calibrated using pilot testing with known-positive and known-negative queries. While this threshold effectively prevents semantic hallucination in tested scenarios, edge cases near the boundary may produce  false  rejections  (blocking  valid  queries)  or  false  acceptances  (allowing  weakly  related queries). The dual-threshold system (0.30 for domain alignment, 0.35 for theme proximity) provides additional robustness but requires periodic recalibration when corpus content changes significantly.

## 3.11.2 Delimitations

Predefined  Themes  Only: The  system  retrieves  passages  for  predefined  themes  from established  literary  sources  rather  than  discovering  themes  automatically  through  unsupervised methods (e.g., LDA, NMF). This design choice:

- Focuses research on retrieval effectiveness rather than theme extraction
- Ensures evaluation against validated thematic frameworks
- Aligns with educational use cases where teachers query specific curriculum themes

- Acknowledges that automated theme discovery for Filipino literature represents a separate, substantial research question

No Large-Scale User Studies: Evaluation relies on expert review rather than extensive user testing with students or teachers, acknowledging resource constraints of a thesis project. This delimitation means:

- Technical performance is validated but pedagogical effectiveness in classroom settings is not empirically measured
- User experience and usability aspects receive limited evaluation
- Future work should validate educational effectiveness through classroom deployment and longitudinal studies

Single Model Architecture: The study uses XLM-RoBERTa exclusively without comparing against other multilingual models (e.g., mBERT, mT5, LaBSE). This choice was made because:

- Literature  review  (Chapter  2)  established  XLM-RoBERTa  as  optimal  for  Filipino  codeswitched text (Salve &amp; Tubil, 2025; Cosme &amp; De Leon, 2024)
- Resource constraints limit comprehensive model comparison
- The  research  focuses  on  dual-formula  hybrid  architecture  design  with  domain-adaptive validation rather than model selection

Sentence-Level Granularity: Passages are defined as individual sentences rather than multi-sentence paragraphs or variable-length discourse units. This was chosen for:

- Clear semantic boundaries
- Alignment with summary structure

- Computational efficiency
- Precise theme localization

However,  some  themes  may  span  multiple  sentences,  requiring  context  expansion  to capture full thematic expression-which Formula 2 addresses.

No Temporal Analysis: The study does not analyze how themes evolve chronologically within  each  novel's  narrative  progression.  While  distribution  visualizations  show  where  themes appear, detailed temporal/narrative arc analysis is beyond the current scope.

Binary Formula Design: The system uses exactly two formulas (main and neighbor) rather than exploring a continuous spectrum of weighting schemes or additional specialized formulas for other retrieval subtasks.

Summary Corpus Only: The domain-adaptive semantic validation is calibrated specifically for the summary corpus semantic space. Queries about content present in the original novels but omitted from summaries may be incorrectly rejected as semantically misaligned. This is an inherent trade-off of using summary versions rather than full texts.

## 3.11.3 Validation Boundaries

Thematic  Classification  Threshold: The  0.45  cosine  similarity  threshold  for  theme assignment  was  determined  empirically  through  pilot  testing.  This  threshold  balances  precision (avoiding false theme assignments) with recall (capturing relevant themes). Different thresholds may be appropriate for different use cases or literary corpora.

Neighbor Threshold: The 0.40 combined score threshold for Formula 2 was set higher than the main retrieval threshold to ensure context quality. This conservative threshold may exclude some marginally relevant context sentences that could provide useful narrative continuity.

Chapter  Diversity  Limit: The  maximum  of  3  passages  per  chapter  prevents  overrepresentation  of  theme-rich  chapters  but  may  exclude  some  relevant  passages  in  particularly thematic chapters. This trade-off prioritizes result diversity over exhaustive retrieval.

Expert Review Sample Size: With 2-3 expert reviewers, the qualitative evaluation provides informed feedback but may not capture a full range of interpretive perspectives. Larger expert panels would strengthen validity but exceed thesis resource constraints.

Known-Negative  Query  Set: The  domain  validation  evaluation  uses  a  curated  set  of queries known to reference events/characters not in the summaries. While this validates rejection capability, the set may not cover all possible types of invalid queries. Real-world deployment may encounter unanticipated query patterns requiring threshold adjustment.

These  limitations  and  delimitations  are  acknowledged  transparently  to  contextualize  the study's contributions appropriately within the research landscape established in Chapter 2. They also identify clear directions for future work that extends beyond this thesis scope, particularly regarding the scalability of domain-adaptive validation to larger and more diverse literary corpora.

## 3.12 Summary

This chapter detailed the comprehensive methodology for developing and evaluating a dualformula hybrid information retrieval system with domain-adaptive semantic validation for thematic analysis of Rizal's novels. The approach integrates:

Theoretical Grounding: Four  interconnected  frameworks (Domain-Adaptive Pre-training Theory, Hybrid Retrieval Theory, Cross-lingual Transfer Learning Theory, and Thematic Coherence Theory) guide dual-formula system design with semantic grounding validation.

Domain-Adaptive Semantic Validation: A novel validation layer that computes semantic alignment  between  queries  and  the  corpus  domain,  preventing  retrieval  of  weakly  similar  but thematically irrelevant passages. Queries failing domain alignment (&lt; 0.30) and theme proximity (&lt; 0.35) thresholds are explicitly rejected rather than returning hallucinated results.

Preprocessing  Pipeline: Filipino-specific  stopword  handling  via  stopwords-iso,  XLMRoBERTa  tokenization, sentence-level segmentation, stopword-aware weighting, and strict deduplication with corpus vocabulary building for lexical grounding validation.

## Dual-Formula Hybrid Architecture:

- Formula 1 (Main Retrieval): pairing favoring lexical precision for short queries, semantic understanding  for  longer  queries,  with  weighted  term  matching  using  frequency-based semantic weights
- Formula 2 (Neighbor Retrieval): Employs dynamic frequency-based weighting to assess sentence-to-sentence  coherence,  emphasizing  semantic  continuity  while  accounting  for lexical  similarity  through  the  Jaccard  coefficient.  This  adaptive  mechanism  prioritizes contextually meaningful and rare words, ensuring smoother thematic transitions between adjacent sentences.

Multi-Component Scoring: Balanced integration of semantic similarity (cosine), weighted lexical  overlap,  and length normalization in Formula 1; semantic-prioritized scoring with Jaccard-

based lexical similarity in Formula 2 to capture different aspects of document-query proximity within the Vector Space Model framework (Jain et al., 2017).

Thematic Classification: Passage-theme matching using XLM-RoBERTa embeddings for literary analysis beyond simple retrieval.

Comprehensive Evaluation: Quantitative metrics (P/R/F1/MAP) including domain validation metrics (TNR,  FPR),  ablation studies comparing  single-formula vs. dual-formula performance and with/without domain validation, expert review assessing rejection appropriateness, and distribution/co-occurrence analysis.

The methodology directly addresses all research objectives while acknowledging appropriate limitations and delimitations. Implementation details ensure reproducibility, and ethical considerations  demonstrate  responsible  research  practices.  The  dual-formula  approach  with domain-adaptive validation represents a novel contribution to hybrid retrieval systems by:

1. Recognizing that optimal signal integration depends on retrieval context (query→corpus vs. sentence→sentence)
2. Implementing semantic grounding validation to prevent false retrievals for queries outside the corpus domain
3. Using  complementary  lexical  similarity  measures  (weighted  overlap  for  query  matching, Jaccard for neighbor coherence) appropriate to each retrieval task

The next chapter will present the results obtained through application of this methodology, including  quantitative  performance  metrics,  ablation  study  findings  demonstrating  the  value  of

domain  validation  and  dual-formula  design,  expert  evaluation  of  retrieval  quality  and  rejection decisions, and thematic distribution patterns revealed by the system.

