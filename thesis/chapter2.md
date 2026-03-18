## CHAPTER 2 REVIEW OF RELATED LITERATURE AND STUDIES

The  development  of  effective  information  retrieval  systems  requires  understanding  both traditional keyword-based methods and modern semantic approaches to text processing, particularly for  multilingual  and  low-resource  language  contexts.  This  review  examines  the  progression  of research from basic information retrieval models to specialized applications in Filipino/Tagalog text processing, establishing a theoretical and empirical foundation for the present study.

## 2.1 Hybrid Retrieval Models: Lexical-Semantic Integration

The evolution of information  retrieval  systems  has  been  marked  by  a  gradual  shift  from purely  keyword-based  retrieval  to  hybrid  approaches  that  incorporate  semantic  understanding. Traditional keyword-based retrieval systems, while computationally efficient, have long suffered from vocabulary mismatch problems and inability to capture semantic relationships between queries and documents.

Gao et al. (2021) addressed this fundamental limitation through the development of CLEAR (Complement Lexical Retrieval Model with Semantic  Residual Embeddings), which trains neural embeddings to capture language structures and semantics that lexical retrieval fails to identify. Their residual-based embedding learning method demonstrated substantial improvements over classical BM25  models,  achieving  enhanced  accuracy  and  efficiency  in  reranking  pipelines.  This  work established that semantic and lexical signals are not competing but complementary dimensions of information retrieval.

Building upon this foundation, Kuzi et al. (2020) expanded understanding of hybrid retrieval systems by examining their application across the two-phase retrieval paradigm-initial document retrieval followed by reranking. Their empirical study using TREC collections revealed that combining deep  neural  network-based  semantic  models  with  keyword  matching-based  lexical  models significantly  improved  recall  rates  during  the  retrieval  stage.  Critically,  they  demonstrated  that semantic approaches excel at capturing conceptual similarities while lexical approaches maintain precision for exact-match requirements. This complementary nature justified the integration of both approaches  in  modern  retrieval  architectures,  particularly  for  complex  query  scenarios  where semantic understanding becomes essential.

The  theoretical  foundation  for  semantic-lexical  integration  was  further  strengthened  by Melamud et al. (2015), who developed a word embedding model for lexical substitution that used context embeddings from the skip-gram architecture. Their work demonstrated that explicit use of context embeddings-previously considered merely internal components of the learning processcould achieve state-of-the-art results in identifying meaning-preserving word substitutes. This finding had important implications for information retrieval, suggesting that embeddings capture detailed semantic relationships that extend beyond simple word co-occurrence patterns.

## 2.1.1 Justification for Hybrid Architecture in Filipino Literary Retrieval

The necessity of a hybrid approach combining semantic embeddings with lexical matching becomes evident when examining the fundamental limitations of single-method retrieval systems in Filipino literary contexts.

## The Inadequacy of Pure Lexical Approaches

Traditional keyword-based retrieval systems operate on exact string matching, identifying passages containing specified terms regardless of semantic context. This approach encounters three critical  problems.  First,  vocabulary  mismatch  occurs  because  themes  manifest  through  diverse expressions without exact keywords-a passage discussing "kolonyalismo" might describe forced labor  or  cultural  suppression  without  using  "kolonyal"  itself.  Second,  synonym  blindness  treats "katarungan,"  "patas  na  paghatol,"  and  "hustisya"  as  distinct  terms  despite  their  semantic equivalence. Third, context insensitivity cannot distinguish "edukasyon" as Ibarra's hope versus the friars' fear-identical words carrying opposite thematic meanings.

## The Limitations of Pure Semantic Approaches

Relying  solely  on  neural  embeddings  introduces  different  deficiencies.  Semantic  models may  retrieve  conceptually  related  passages  lacking  specific  terminology  users  expect,  reducing result trust. This computational opacity undermines pedagogical objectives when students cannot understand why passages without explicit keywords were retrieved. Additionally, semantic systems risk false associations based on spurious correlations rather than genuine thematic relationships.

## The Complementary Strengths of Hybrid Integration

The hybrid approach leverages both methodologies' strengths. XLM-RoBERTa embeddings capture  thematic  meaning  and  conceptual  relationships  beyond  keywords,  while  weighted  term matching ensures explicit terminology receives appropriate prioritization. The dual-formula architecture  validates  character-action  relationships  against  canonical  events,  preventing  absurd results like "kabaitan ni Padre Damaso" for consistently antagonistic characters. For complex queries

such as "pang-aapi ng mga prayle sa pamilya ni Ibarra," spaCy dependency parsing verifies that actions, agents, and targets semantically align in contextually correct relationships.

## Computational Trade-offs and Their Justification

This  architectural  sophistication  introduces  computational  costs:  embedding  generation, similarity computation, multi-component scoring, and domain validation. However, these costs are justified when result quality outweighs speed. Hybrid retrieval achieves significantly higher precision and recall than keyword-only baselines. Students benefit more from semantically appropriate results than immediate but incomplete matches requiring manual filtering. The system remains deployable on standard educational hardware without GPU requirements.

As Gao et al. (2021) and Kuzi et al. (2020) demonstrated, performance gains from hybrid integration justify computational overhead when retrieval quality impacts downstream tasks. This principle applies more strongly to educational applications in low-resource languages, where retrieval errors compound learning difficulties. When students struggle to locate relevant passages, they may abandon  sophisticated  analysis  for  superficial  observations  a  pedagogical  failure  outweighing computational costs.

This foundation establishes hybrid architecture as a fundamental requirement for effective thematic literary  analysis  in  Filipino  texts.  The  integration  of  semantic  understanding  with  lexical precision,  validated  through  domain-adaptive  pre-training  and  multi-component  scoring,  supports complex analytical tasks while maintaining computational feasibility in educational environments.

## Comparison of Traditional and Hybrid Information Retrieval Systems

Information retrieval in traditional literary analysis operates through lexical matching processes where documents are retrieved based on exact term correspondence between queries and indexed content. The Boolean model, which represents the classical approach to information retrieval, treats documents as sets of terms and employs logical operators (AND, OR, NOT) to specify retrieval conditions (Sciencedirect.com, n.d.). Traditional methods such as BM25 (Best Matching 25) and TF-IDF (Term Frequency-Inverse Document Frequency) calculate relevance scores based on statistical measures of term frequency, inverse document frequency, and document length normalization (Iterate.ai, n.d.; Mtham8.github.io, n.d.). While these approaches excel at exact keyword matching with query processing times under 10 milliseconds, they face fundamental limitations in semantic understanding (Google Cloud, 2024). The vocabulary mismatch problem-where semantically relevant documents use different terminology than the search query-represents the primary weakness of traditional systems (University of South Africa, 2021). For instance, a search for "cardiovascular disease" would fail to retrieve documents containing the synonymous term "heart condition," despite their semantic equivalence.

In contrast, hybrid information retrieval systems integrate lexical precision with semantic understanding through multiple parallel retrieval pathways. These systems combine sparse vector indices (using BM25 or learned sparse models like SPLADE) for keyword matching with dense vector indices (using transformer-based embeddings from BERT or Sentence Transformers) for semantic retrieval (Zilliz.com, 2024; Meilisearch, 2023). The architecture employs neural encoding pipelines that represent documents in both high-dimensional dense vectors (768-1024 dimensions) and sparse vectors with thousands of dimensions but few non-zero values (Google Cloud, 2024; Zilliz.com, 2024). A fusion layer, typically implementing Reciprocal Rank Fusion (RRF), merges

result from both retrieval methods based on ranking positions rather than raw scores, avoiding the complexities of score normalization across different algorithms (Microsoft Learn, 2024; OpenSearch, 2024). This dual approach addresses the vocabulary mismatch problem by understanding context and meaning beyond exact term matches, enabling the system to recognize synonyms, related concepts, and contextual variations automatically (Whisperit.ai, n.d.).

Empirical evidence demonstrates substantial performance advantages for hybrid systems across standardized benchmarks. Research by Siriwardhana et al. (2024) on the Natural Questions (NQ) dataset showed hybrid approaches achieved an NDCG@10 score of 0.67 compared to 0.633 for baseline methods, representing a 5.8% improvement, while the TRECCOVID dataset demonstrated even greater gains with NDCG@10 of 0.87 versus 0.804 baseline, an 8.2% improvement. On the SQuAD dataset, hybrid retrieval achieved an F1 score of 68.4 compared to 52.63 for fine-tuned baseline systems, marking a 30% improvement (Siriwardhana et al., 2024). Comparative studies report that semantic search components improve retrieval precision by 25-35% over keyword methods for ambiguous queries and reduce irrelevant results by up to 40% in enterprise knowledge bases (Whisperit.ai, n.d.). Industry implementations have shown response time reductions of 30-40% and retrieval accuracy increases of 40% in production environments (Dev.to, 2024). Rakuten's implementation of semantic search reported a 5% increase in sales, demonstrating tangible business impact beyond academic metrics (Celerdata.com, n.d.).

However, hybrid systems introduce significant computational and implementation complexities that must be considered. Dense vector indices consume substantially more memoryapproximately 50GB for 5 million documents in the HotPotQA dataset-compared to the minimal storage requirements of traditional inverted indices (Siriwardhana et al., 2024). Query processing

latency increases to 10-100 milliseconds depending on the fusion method employed, compared to sub-10 millisecond response times for traditional systems (OpenSearch, 2024; Google Cloud, 2024). The implementation requires careful tuning of fusion parameters, weight optimization between keyword and semantic components, and often necessitates GPU resources for acceptable performance with neural embedding models (Fuzzylabs.ai, n.d.; Couchbase, 2024). Performance dependencies on embedding model quality mean that general-purpose models may underperform in specialized domains without fine-tuning, and bi-encoder architectures offer minimal advantages over BM25 in zero-shot, out-of-domain scenarios (University of Waterloo, 2024; Emergentmind.com, n.d.). Storage overhead for advanced architectures like ColBERT can reach 6-10 times larger footprints than traditional indices (arXiv, 2021).

The choice between traditional and hybrid retrieval systems depends on specific application requirements and resource constraints. Traditional systems remain appropriate for scenarios requiring exact term matching, minimal computational overhead, and situations where vocabulary is standardized and controlled. Hybrid systems demonstrate clear advantages for applications involving natural language queries, diverse terminology, cross-lingual searches, and contexts where semantic understanding is critical-such as literary analysis of texts with rich figurative language, historical vocabulary variations, and complex thematic interconnections. For analyzing works like Noli Me Tangere and El Filibusterismo , where themes manifest through various narrative devices, symbolic imagery, and evolving terminology across the two novels, hybrid retrieval's capacity to identify semantically related passages regardless of exact lexical matches provides substantial analytical advantages that justify the increased computational investment

Table 2: Traditional vs. Hybrid Information Retrieval Systems

| Aspect                            | Traditional IR                                        | Hybrid IR                                                       | Sources                                      |
|-----------------------------------|-------------------------------------------------------|-----------------------------------------------------------------|----------------------------------------------|
| Core Method                       | Lexical matching with Boolean operators, BM25, TF-IDF | Lexical matching + semantic understanding via neural embeddings | Sciencedirect.com (n.d.); Meilisearch (2023) |
| Index Structure                   | Inverted index (sparse)                               | Multiple indices: inverted + dense vector                       | Google Cloud (2024); Zilliz.com (2024)       |
| Query Processing Time             | <10ms                                                 | 10-100ms                                                        | OpenSearch (2024); Google Cloud (2024)       |
| Storage Requirements              | Minimal (inverted index only)                         | Significant (50GB for 5M documents)                             | Siriwardhana et al. (2024)                   |
| Semantic Understanding            | None-exact term matching only                         | Yes-context, synonyms, related concepts                         | Whisperit.ai (n.d.)                          |
| Vocabulary Mismatch               | Cannot resolve                                        | Effectively addresses                                           | University of South Africa (2021)            |
| NDCG@10 Performance (NQ)          | 0.633                                                 | 0.67 (+5.8%)                                                    | Siriwardhana et al. (2024)                   |
| NDCG@10 Performance (TREC- COVID) | 0.804                                                 | 0.87 (+8.2%)                                                    | Siriwardhana et al. (2024)                   |
| F1 Score (SQuAD)                  | 52.63                                                 | 68.4 (+30%)                                                     | Siriwardhana et al. (2024)                   |
| Precision Improvement             | Baseline                                              | +25-35% for ambiguous queries                                   | Whisperit.ai (n.d.)                          |
| Implementation Complexity         | Low                                                   | High-requires parameter tuning, fusion optimization             | Fuzzylabs.ai (n.d.); Couchbase (2024)        |
| Hardware Requirements             | CPU sufficient                                        | Often requires GPU for embeddings                               | Couchbase (2024)                             |
| Best Use Cases                    | Exact matching, controlled vocabulary                 | Natural language, diverse terminology, semantic search          | Dev.to (2024)                                |

## 2.2 Redundancy Management in Multi-Document Retrieval

As  retrieval  systems  evolved  to  handle  larger  document  collections,  the  challenge  of redundancy detection became increasingly critical. Hendrickx et al. (2009) investigated this problem in the context of multi-document summarization for Dutch, introducing semantic overlap detection tools  that  went  beyond  simple  string  matching.  While  their  initial  results  did  not  demonstrate superiority  over  traditional  methods,  their  work  highlighted  a  fundamental  tension  in  information retrieval: the need to balance comprehensive coverage with the elimination of redundant information. This  challenge  becomes  particularly  acute  in  passage-based  retrieval  systems  where  multiple segments may convey similar information through different word choices.

## 2.3 Preprocessing and Stopword Management in Multilingual Contexts

The  effectiveness  of  any  retrieval  system  fundamentally  depends  on  appropriate  text preprocessing,  particularly  regarding  stopword  handling  and  term  extraction.  Ladani  and  Desai (2020)  provided  a  comprehensive  survey  of  stopword  identification  and  removal  techniques, documenting that stopword removal can decrease corpus size by 35-45% while improving efficiency and accuracy of text mining applications. Their analysis revealed that most stopword resources have been developed for high-resource languages, with limited attention to language-specific stopword lists  for  low-resource  languages.  This  gap  poses  significant  challenges  for  multilingual  retrieval systems.

Kalykulova and Alzhanov (2025) advanced  the field by introducing UNI-TEX, an unsupervised method for unigram term extraction that uses embedding-based filtering to address

data noise caused by common words. Testing on the ACTER corpus across four domains and three languages  demonstrated  an  average  F1-score  of  46%  for  English,  establishing  the  viability  of semantic filtering for term extraction. However, their work also highlighted the persistent challenge of adapting such methods to languages with limited NLP resources and tools.

For  Tagalog  specifically,  Bation  et  al.  (2017)  demonstrated  the  practical  importance  of stopword  removal  in  document  classification  tasks.  Their  automatic  categorization  system  for Tagalog documents using Support Vector Machines achieved 92% accuracy when stopwords such as "ang," "mga," "si," and "dahil" were removed prior to feature extraction. The study established that these  high-frequency  Tagalog  words,  while  grammatically  essential,  offer  minimal  discriminative power  for  document  categorization-a  finding  with  direct  implications  for  information  retrieval systems operating on Filipino text.

The role of word frequency in retrieval quality was further examined by AlShammari (2023), who implemented text similarity measurement using word frequency and cosine similarity in Python. Their  work  demonstrated  that  term  weighting  based  on  frequency  distributions  provides  a computationally efficient mechanism for relevance scoring. However, the English-centric nature of most frequency-based approaches necessitates validation on other language corpora, particularly for languages like Tagalog where frequency distributions may exhibit different characteristics.

## 2.4 Multilingual Semantic Models: XLM-RoBERTa and Cross-Lingual Understanding

The advent of multilingual transformer models represented a paradigm shift in cross-lingual information retrieval, with XLM-RoBERTa emerging as a particularly powerful architecture for lowresource languages. Conneau et al. (2019) developed XLM-RoBERTa as a cross-lingual language

model trained on 100 languages, enabling transfer learning from high-resource languages to lowresource languages through shared semantic representations.

Wiciaputra et al. (2021) explored XLM-RoBERTa's transfer learning capabilities for bilingual text classification in English and Indonesian, achieving 93.3% accuracy on English datasets and 90.2%  on  Indonesian  datasets  through  a  mixed-language  training  approach.  Their  results demonstrated  that  XLM-RoBERTa's  cross-lingual  representations  effectively  captured  semantic patterns across related languages, suggesting potential applicability to Filipino/Tagalog contexts.

Imperial (2021) provided crucial evidence for XLM-RoBERTa's effectiveness in Filipino NLP tasks through research on automatic readability assessment. By combining BERT embeddings with handcrafted linguistic  features,  Imperial  achieved  a  12.4%  improvement  in  F1  performance  over classical  approaches  on  Filipino  datasets.  Significantly,  the  study  demonstrated  that  BERT embeddings could serve as substitute feature sets for low-resource languages with limited semantic and syntactic NLP tools-a finding directly relevant to the challenges of Filipino information retrieval where traditional linguistic resources are scarce.

The applicability of XLM-RoBERTa to Filipino and Tagalog was further validated by Asai et al.  (2022)  in  their  cross-lingual  open-retrieval  question  answering  shared  tasks  spanning  16 typologically diverse languages, including Tagalog. While the best-performing system achieved 32.2 F1 overall, performance on Tagalog remained challenging, with some systems yielding near-zero scores.  This  highlighted  the  persistent  difficulties  of  information  retrieval  in  truly  low-resource contexts  and  underscored  the  need  for  specialized  approaches  tailored  to  Filipino  linguistic characteristics.

Cosme and De Leon (2024) provided definitive evidence for XLM-RoBERTa's superiority in handling  Filipino-English  code-switching  through  their  sentiment  analysis  study  of  product  and service reviews. Their fine-tuned XLM-RoBERTa model achieved 0.84 accuracy and weighted F1score,  with  individual  class  F1-scores  of  0.89,  0.86,  and  0.78  for  positive,  negative,  and  neutral sentiments respectively. The dramatic performance gap between XLM-RoBERTa and lexicon-based tools  designed  for  monolingual  text  showed  the  limitations  of  single-language  approaches  when applied  to  the  bilingual  reality  of  Filipino  communication.  This  finding  has  direct  implications  for retrieval systems targeting Filipino literature, which frequently exhibits code-switching patterns.

The performance advantages of XLM-RoBERTa over alternative transformer architectures were systematically documented by Salve and Tubil (2025) in their study of bilingual code-switching. Testing  XLM-RoBERTa Large against both its Base variant and multilingual BERT (mBERT) on Taglish text classification tasks, they achieved 97.62% precision, 95.77% F1-score, and 97.50% ROC-AUC. The study's advanced preprocessing techniques-including language tagging, ambiguity tagging, and customized tokenization-enabled the model to handle informal and code-switched constructs effectively. These results established XLM-RoBERTa Large as particularly well-suited for Filipino text processing applications.

Comparative analyses across multiple transformer architectures by Timoneda and Vallejo Vera (2023) provided broader context for model selection in political science text analysis. Their cross-lingual evaluation demonstrated that XLM-RoBERTa  significantly outperformed both multilingual BERT and multilingual DeBERTa in cross-lingual applications, though RoBERTa and DeBERTa showed advantages over BERT in certain monolingual circumstances. This reinforced XLM-RoBERTa's position as the optimal choice for multilingual retrieval tasks involving low-resource languages.

## 2.5 Domain-Adaptive Pre-training for Specialized Tasks

The enhancement of pre-trained language models through domain-specific adaptation has emerged as a critical strategy for improving performance in specialized applications. Krieger et al. (2022)  explored  domain-adaptive  pre-training  (DAPT)  for  language  bias  detection  in  news, demonstrating  how  equipping  pre-trained  models  with  domain-specific  knowledge  significantly improves  task  performance.  Their  approach  built  upon  foundational  work  by  Sun  et  al.  and Gururangan et al., which established that adapting BERT and RoBERTa to domains related to the target task substantially boosts classification accuracy, while irrelevant domain pre-training leads to performance decline.

Krieger et al. (2022) performed domain-adaptive pre-training on the WNC corpus, consisting of  180,000  manually  annotated  sentence  pairs  contrasting  biased  and  neutral  statements  from Wikipedia. The corpus included epistemological, framing, and demographic bias types. Given that both WNC (used for pre-training) and BABE (used for fine-tuning) shared similar bias structures, their  hypothesis  posited  that  domain-specific  pre-training  would  substantially  enhance  model performance in sentence-level media bias detection. Their focused and direct setup leveraged largescale  bias-related  data  to  achieve  improved  accuracy  and  robustness  in  bias  classification, demonstrating that domain adaptation strategies previously successful in biomedical (BioBERT) and scientific text processing (SciBERT) could be effectively applied to media analysis tasks.

This work has important implications for Filipino information retrieval, suggesting that models pre-trained on general multilingual corpora could benefit from additional adaptation to Filipino literary text. The principle that domain-specific pre-training enhances performance when the pre-training and

target  domains  share  structural  similarities  indicates  potential  pathways  for  optimizing  XLMRoBERTa specifically for Filipino literature retrieval tasks.

## 2.6 Scoring Functions and Ranking Mechanisms

The effectiveness  of  hybrid  retrieval  systems  depends  critically  on  sophisticated  scoring functions that appropriately weight multiple signals. Yoo et al. (2024) addressed this challenge in the context  of  abstractive  summarization,  proposing  a  hierarchical  supervision  method  that  jointly performs summary-level and sentence-level supervision. Their approach included intra- and intersentence ranking losses, enabling more detailed evaluation of content quality. While developed for summarization,  their  hierarchical  scoring  framework  offered  insights  applicable  to  information retrieval, particularly regarding the importance of sentence-level detail in relevance assessment.

The concept of penalty factors in scoring functions was implicitly demonstrated by Bation et al. (2017), whose document classification system showed improved performance when stopwordheavy sentences received reduced weighting. This suggested that effective retrieval systems should incorporate penalty mechanisms for low-information content, a principle particularly relevant when processing documents containing substantial boilerplate or formulaic text.

## 2.7 Thematic Retrieval and Topic Modeling

While  lexical  matching  and  semantic  similarity  form  the  foundation  of  retrieval  systems, thematic understanding represents a higher-level capability increasingly recognized as essential for literature-based applications. Ignaco and Ballera (2025) demonstrated this through their LDA-based Bible verse search system, which retrieved verses based on topical relevance rather than exact word matches.  Their  system  achieved  33-67%  accuracy  in  finding  thematically  relevant  verses  that

traditional keyword search completely missed (0% accuracy). Though retrieval time increased from 0.04 to 17-23 seconds, the study established that statistical topic modeling could uncover meaningful thematic connections invisible to keyword-based approaches. This work highlighted the fundamental trade-off between retrieval speed and thematic depth, a consideration essential for literature-focused retrieval systems.

The  integration  of  transformer-based  embeddings  with  topic  modeling  was  explored  by Aamir et al. (2025), who developed a framework combining BERTopic, XLM-R, and GPT for Urdu text analysis. Their approach achieved a coherence improvement of 0.05 and diversity score of 0.87 over  traditional  methods  like  LDA  and  NMF,  demonstrating  that  transformer  embeddings  could capture contextual details and grammatical intricacies that statistical methods missed. Significantly, their work on Urdu-another low-resource language with complex morphology suggested pathways for applying similar techniques to Filipino text analysis.

Gaikwad et al. (2024) synthesized recent advances in document clustering, emphasizing that BERT embeddings, when applied as feature vectors, effectively captured intricate syntactical and semantic relationships for clustering tasks. Their analysis of BERTopic, which combines BERT embeddings with topic modeling, demonstrated comprehensive syntactical and semantic understanding  of  document  content.  The  study  established  that  transformer-based  embeddings enabled more detailed thematic organization than traditional bag-of-words approaches, though the effectiveness depended on appropriate preprocessing and feature extraction strategies.

For Filipino literature specifically, thematic analysis has traditionally focused on canonical works  such  as  Noli  Me  Tangere  and  El  Filibusterismo.  Rizal  (1887,  1891)  embedded  recurring themes in these novels including religion, power, radicalism versus incrementalism, education, family

and honor, sacrifice, privilege, isolation, and revenge. These established thematic categories provide a framework for evaluating whether retrieval systems can capture conceptual relationships beyond surface-level lexical matching a capability essential for supporting literary analysis and education.

## 2.8 The Research Gap

The reviewed literature  establishes  a  clear  progression  from  basic  keyword  matching  to sophisticated  hybrid  retrieval  systems  that  integrate  lexical,  semantic,  and  thematic  dimensions. However, several critical gaps remain, particularly regarding Filipino/Tagalog applications:

First, while XLM-RoBERTa  has  demonstrated  strong  performance  on  Filipino  text classification and sentiment analysis tasks, its application to passage-level retrieval from literary texts remains unexplored. The code-switching capabilities documented by Cosme and De Leon (2024) and Salve and Tubil (2025) suggest potential for handling Filipino literature, but require validation in retrieval contexts.

Second, existing studies of hybrid retrieval systems (Gao et al., 2021; Kuzi et al., 2020) have focused  primarily  on  high-resource  languages  and  general  document  collections.  The  specific challenges  of  retrieving  from  Filipino  literary  works  which  exhibit  distinctive  linguistic  features including formal/archaic language and culturally-specific references have not been systematically addressed.

Third, while thematic retrieval has been explored for religious texts (Ignaco &amp; Ballera, 2025) and  other  domains,  its  application  to  Filipino  literature  analysis  remains  underdeveloped.  The established thematic frameworks for works like Noli Me Tangere provide evaluation criteria, but no

existing system has demonstrated the ability to retrieve passages based on thematic similarity using Filipino-capable embeddings.

Fourth,  the  preprocessing  requirements  for  Filipino  text  particularly  regarding  stopword handling, frequency-based validation, and deduplication have been examined in isolation (Bation et al., 2017; Kalykulova &amp; Alzhanov, 2025) but not integrated into a comprehensive retrieval pipeline optimized for literary texts.

Fifth, while domain-adaptive pre-training has proven effective for specialized tasks in highresource languages (Krieger et al., 2022), its application to low-resource language retrieval systems remains unexplored. The potential benefits of adapting multilingual models specifically to Filipino literary text characteristics have not been empirically validated.

Finally, scoring functions that appropriately balance lexical overlap, semantic similarity, and passage quality for Filipino text remain undeveloped. While general principles have been established (Yoo et al., 2024), their adaptation to Filipino linguistic characteristics requires empirical validation.

Lexical

Index

(inverted index)

Semantic

Index

(document embeddings)

Lexical

Lexical

Retrieval

Result List

## 2.9 Theoretical Framework

Initial

The  present  study  is  grounded  in  three  interconnected  theoretical  frameworks  that collectively inform the design and implementation of the hybrid information retrieval system:

Semantic

Retrieval

Result List

Figure 1: Theoretical Framework of a Hybrid Document Retrieval System Leveraging Semantic and Lexical Matching Based on Kuzi et al. (2020)

<!-- image -->

## 2.9.1 Hybrid Retrieval Theory

Based on the work of Gao et al. (2021) and Kuzi et al. (2020), Hybrid Retrieval Theory proposes that  optimal  information  retrieval  emerges  from  the  integration  of  lexical  and  semantic approaches rather than their competition. Lexical methods excel at precise matching and handle outof-vocabulary  terms  effectively,  while  semantic  methods  capture  conceptual  relationships  and synonymy. The theory suggests that residual-based learning where semantic models are explicitly trained  to  capture  information  that  lexical  models  miss  produces  superior  retrieval  performance compared to either approach alone. This framework guides the study's hybrid architecture, which

combines  TF-IDF-based  lexical  matching  with  XLM-RoBERTa  semantic  embeddings  through  a weighted scoring function.

## 2.9.2. Cross-lingual Transfer Learning Theory

Drawing  from  the  multilingual  NLP  research  of  Conneau  et  al.  (2019),  Imperial  (2021), Cosme and De Leon (2024), and Salve and Tubil (2025), Cross-lingual Transfer Learning Theory explains  how  pre-trained  multilingual  models  develop  language-agnostic  representations  that transfer across typologically diverse languages. The theory proposes that transformer models trained on large multilingual corpora learn to map similar concepts to proximate regions in embedding space regardless of surface-level linguistic differences. For low-resource languages like Filipino/Tagalog, this  enables  using  semantic  understanding  developed  from  high-resource  languages.  This framework  justifies  the  selection  of  XLM-RoBERTa  as  the  semantic  encoder  and  informs  the preprocessing strategies that maximize cross-lingual transfer effectiveness.

## 2.9.3 Thematic Coherence Theory

Synthesized from the work of Yoo et al. (2024), Ignaco and Ballera (2025), and Aamir et al. (2025), Thematic Coherence Theory proposes that document relevance operates at multiple levels lexical,  semantic,  and  thematic  each  contributing  distinct  information  to  retrieval  quality.  Lexical relevance captures explicit term overlap; semantic relevance captures conceptual similarity through embeddings; thematic relevance captures higher-order topical relationships. The theory suggests that optimal retrieval requires explicit modeling of each level and appropriate weighting mechanisms that account for passage characteristics such as length, information density, and stopword presence. This  framework  guides  the  study's  multi-component  scoring  function  and  thematic  retrieval capabilities.

## 2.9.4 Integration of Theoretical Frameworks

These three theoretical frameworks interact in the present study. Hybrid Retrieval Theory provides  the  overarching  architectural  principle  of  integration.  Cross-lingual  Transfer  Learning Theory enables the application of this architecture to Filipino text by using multilingual pre-training. Thematic  Coherence  Theory  operationalizes the hybrid approach  through specific scoring mechanisms  that  balance  multiple  relevance  signals.  Together,  these  frameworks  establish  a comprehensive  theoretical  foundation  for  developing  and  evaluating  a  Filipino  literature  retrieval system that advances beyond existing keyword-based approaches.

## 2.10 Synthesis of the Study

The reviewed literature reveals a clear trajectory of advancement in information retrieval systems,  progressing  from  simple  keyword  matching  to  sophisticated  hybrid  approaches  that integrate multiple forms of linguistic understanding. However, this progression has occurred primarily within  high-resource  language  contexts,  leaving  significant  gaps  in  the  application  of  advanced retrieval techniques to Filipino/Tagalog literary texts.

## 2.10.1 Key Findings and Convergence

Several critical findings emerge from the literature synthesis:

Hybrid Superiority: The superiority of hybrid retrieval approaches over purely lexical or purely semantic methods has been conclusively established (Gao et al., 2021; Kuzi et al., 2020), providing strong justification for the architectural decisions in the present study.

XLM-RoBERTa  Effectiveness: Multiple  independent  studies  (Conneau  et  al.,  2019; Imperial, 2021; Cosme &amp; De Leon, 2024; Salve &amp; Tubil, 2025; Timoneda &amp; Vallejo Vera, 2023) have converged on XLM-RoBERTa as the optimal multilingual transformer architecture for Filipino text processing, supporting its selection for semantic encoding in this study.

Domain Adaptation Potential: Research on domain-adaptive pre-training (Krieger et al., 2022) has demonstrated that specialized adaptation of pre-trained models significantly enhances performance  on  domain-specific  tasks,  suggesting  potential  pathways  for  optimizing  multilingual models for Filipino literary text retrieval.

Preprocessing Criticality: The importance of language-specific preprocessingparticularly stopword removal and term validation has been demonstrated across multiple contexts (Bation  et  al.,  2017;  Ladani  &amp;  Desai,  2020;  Kalykulova  &amp;  Alzhanov,  2025),  informing  the preprocessing pipeline design.

Thematic Potential: Emerging research on topic modeling and thematic retrieval (Aamir et al.,  2025;  Gaikwad  et  al.,  2024)  suggests  that  embeddings  can  capture  higher-order  thematic relationships, though application to Filipino literature remains unexplored.

## 2.10.2 Research Gaps Addressed by This Study

The present study directly addresses several critical gaps identified in the literature:

Gap 1: Filipino Literary Retrieval While semantic retrieval has been extensively studied for  general documents, news articles, and social media text, no existing research has examined passage-level retrieval from Filipino literary works. This study fills this gap by developing a system specifically optimized for canonical Filipino literature.

Gap 2: Hybrid Retrieval for Low-Resource Languages Existing hybrid retrieval systems have focused on high-resource languages with abundant training data and linguistic tools. This study extends hybrid retrieval methodology to Filipino, demonstrating how multilingual pre-trained models can enable sophisticated retrieval in low-resource contexts.

- Gap  3:  Thematic  Retrieval  in  Filipino  Literature While  thematic  frameworks  exist  for Filipino literature (Rizal, 1887, 1891), no computational system has attempted to retrieve passages based  on  thematic  similarity  using  Filipino-capable  embeddings.  This  study  pioneers  thematic retrieval for Filipino literary analysis.
- Gap  4:  Integrated  Preprocessing  for  Filipino  IR Previous  studies  have  examined individual  preprocessing  techniques  (stopword  removal,  frequency  validation,  deduplication)  in isolation. This study integrates these techniques into a comprehensive pipeline specifically designed for Filipino literary text characteristics.
- Gap 5: Empirical Validation of Scoring Functions While  general  principles  for  hybrid scoring exist, their application to Filipino text with its distinctive code-switching, formal register, and cultural references has not been empirically validated. This study provides systematic evaluation of scoring function components in the Filipino context.
- Gap 6: Domain Adaptation for Filipino Retrieval While domain-adaptive pre-training has proven  effective  in  specialized  high-resource  language  tasks  (Krieger  et  al.,  2022),  its  potential application  to  low-resource  language  retrieval  systems  remains  unexplored.  This  study  lays groundwork  for  future  research  on  adapting  multilingual  models  specifically  to  Filipino  literary domains.

## 2.10.3 Contribution to the Field

This study represents the logical next link in the chain of information retrieval research by:

Extending Proven Architectures: Applying the validated hybrid retrieval architecture of Gao et al. (2021) and Kuzi et al. (2020) to a new linguistic context, demonstrating its applicability beyond high-resource languages.

Leveraging Multilingual Capabilities: Building upon the Filipino NLP advances of Imperial (2021), Cosme and De Leon (2024), and Salve and Tubil (2025) by applying XLM-RoBERTa to a novel task domain (literary passage retrieval).

Advancing Thematic Understanding: Extending the topic modeling approaches of Ignaco and  Ballera  (2025)  and  Aamir  et  al.  (2025)  to  Filipino  literature,  demonstrating  that  transformer embeddings can support thematic analysis in low-resource educational contexts.

Addressing Practical Needs: Responding to the documented challenges in Filipino literary education by providing a tool that supports close reading, thematic analysis, and passage discovery capabilities currently unavailable to educators and students.

## 2.10.4 Positioning Within the Research Landscape

The  present  study  occupies  a  unique  position  at  the  intersection  of  several  research trajectories: hybrid information retrieval, multilingual NLP, low-resource language processing, and digital humanities. It represents both an application of established principles to a new context and an extension of those principles to address context-specific challenges. By systematically evaluating lexical  overlap,  semantic  similarity,  and  thematic  relevance  for  Filipino  literary  texts,  this  study

provides  empirical  evidence  that  advances  understanding  of  how  retrieval  systems  should  be designed for linguistically and culturally distinctive content.

The findings  from  this  research  have  implications  extending  beyond  Filipino  literature  to other low-resource literary traditions facing similar challenges in digital access and computational analysis. The methodological framework developed here combining language-specific preprocessing, multilingual embeddings, and multi-dimensional relevance scoring offers a template adaptable to other underserved literary contexts.

## 3.1 Research Design

This  study  employs  a  design  and  development  research  approach  with  comparative evaluation  to  create  and  assess  a  dual-formula  hybrid  information  retrieval  system  for  thematic analysis of José Rizal's Noli Me Tangere and El Filibusterismo . The research design consists of four primary  phases:  (1)  domain-adaptive  pre-training  for  semantic  grounding  validation,  (2)  system development with dual-formula architecture, (3) implementation and testing, and (4) comparative evaluation. This methodology  directly addresses  the  research objectives by systematically developing,  implementing,  and  evaluating  a  hybrid  retrieval  architecture  that  integrates  XLMRoBERTa semantic embeddings with weighted lexical matching and Filipino-specific preprocessing through two specialized scoring formulas, enhanced by domain-adaptive semantic validation.

The research is grounded in four theoretical frameworks established in Chapter 2: Hybrid Retrieval  Theory  (Gao  et  al.,  2021;  Kuzi  et  al.,  2020),  Cross-lingual  Transfer  Learning  Theory (Conneau et al., 2019; Imperial, 2021), Thematic Coherence Theory (Yoo et al., 2024; Ignaco &amp; Ballera, 2025; Aamir et al., 2025), and Domain-Adaptive Pre-training Theory (Krieger et al., 2022). These frameworks inform the architectural decisions, preprocessing strategies, dual-formula design, semantic validation mechanisms, and evaluation criteria throughout the methodology.

