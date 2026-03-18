## TECHNOLOGICAL INSTITUTE OF THE PHILIPPINES - MANILA College of Computer Studies

Enhancing Thematic Literary Analysis of Rizal's Novels Through Hybrid Information Retrieval: Integrating Lexical and Semantic Approaches Using XLM-RoBERTa Embeddings for Noli Me Tangere and El Filibusterismo

Bachelor of Science in Computer Science CCS 401 - Thesis 1

## Submitted by:

Oliver, Marcus Kent R. Vilog, Dominic B. Placencia, Ian Kurby A.

## Submitted to:

Prof. Melvin Ballera, PhD

Date: November 25, 2025

## TABLE OF CONTENTS

| CHAPTER 1 INTRODUCTION ........................................................................................................................1      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.1 Background of the Study ........................................................................................................1                 |
| 1. Subjectivity and inconsistency in manual thematic analysis .............................................................10                         |
| 3. Lack of specialized computational literary studies tools for Filipino texts..........................................11                            |
| 1.3 Research Questions .............................................................................................................12                |
| 1.4 Objectives of the Study.........................................................................................................12                |
| General Objective .................................................................................................................................12 |
| Specific Objectives................................................................................................................................13 |
| 1.5 Significance of the Study ......................................................................................................13                |
| 1.5.6 Application Utilization and Impact Metrics .........................................................................15                          |
| 1.6 Scope and Delimitation.........................................................................................................22                 |
| 1.7 Definition of terms.................................................................................................................25            |
| 1.8 Basis of the Study.................................................................................................................27             |
| 2 REVIEW OF RELATED LITERATURE AND STUDIES                                                                                                            |
| CHAPTER ..........................................................33                                                                                  |
| 2.1 Hybrid Retrieval Models: Lexical-Semantic Integration........................................................33                                   |
| 2.1.1 Justification for Hybrid Architecture in Filipino Literary Retrieval .......................................34                                  |
| 2.2 Redundancy Management in Multi-Document Retrieval.......................................................41                                        |
| 2.3 Preprocessing and Stopword Management in Multilingual Contexts ....................................41                                             |
| 2.4 Multilingual Semantic Models: XLM-RoBERTa and Cross-Lingual Understanding...............42                                                        |
| 2.5 Domain-Adaptive Pre-training for Specialized Tasks............................................................45                                  |
| 2.6 Scoring Functions and Ranking Mechanisms.......................................................................46                                 |
| 2.7 Thematic Retrieval and Topic Modeling................................................................................46                           |
| 2.9.1 Hybrid Retrieval Theory ...............................................................................................................50       |
| 2.9 Theoretical Framework.........................................................................................................50                  |
| 2.9.3 Thematic Coherence Theory........................................................................................................51             |
| 2.9.2. Cross-lingual Transfer Learning Theory......................................................................................51                 |
| 2.10 Synthesis of the Study........................................................................................................52                 |
| 2.10.1 Key Findings and Convergence.................................................................................................52                |

| 2.10.2 Research Gaps Addressed by This Study                                                                                                          | .................................................................................53                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| 2.10.3 Contribution to the Field.............................................................................................................55       |                                                                                                            |
| 2.10.4 Positioning Within the Research Landscape..............................................................................55                      |                                                                                                            |
| CHAPTER 3 METHODOLOGY .....................................................................................................................57         |                                                                                                            |
| 3.1 Research Design ..................................................................................................................57              |                                                                                                            |
| 3.3 Conceptual Framework.........................................................................................................60                   |                                                                                                            |
| 3.3.1 Input Layer: Query and Corpus....................................................................................................62             |                                                                                                            |
| 3.3.2 Processing Layer: Domain-Adaptive Semantic Validation ...........................................................62                             |                                                                                                            |
| 3.3.3 Processing Layer: Dual-Formula Hybrid Retrieval Architecture ...................................................65                              |                                                                                                            |
| 3.3.4 Analysis Layer: Thematic Classification and Context Expansion.................................................67                                |                                                                                                            |
| 3.3.5 Output Layer: Retrieval Results and Visualization .......................................................................68                     |                                                                                                            |
| 3.3.6 Theoretical Integration .................................................................................................................70     |                                                                                                            |
| 3.2.7 Framework Validation ..................................................................................................................71       |                                                                                                            |
| 3.4. Textual Corpus...............................................................................................................................72  |                                                                                                            |
| 3.4.1 Rationale for Using Summary Versions .......................................................................................72                  |                                                                                                            |
| 3.4.2 Thematic Data Sources................................................................................................................73         |                                                                                                            |
| 3.4.3 Domain-Adaptive Corpus Representation....................................................................................74                     |                                                                                                            |
| 3.5 Text Cleaning and Normalization....................................................................................................77             |                                                                                                            |
| 3.5.1 Tagalog Stopword Management..................................................................................................77                 |                                                                                                            |
| 3.5.2 Tokenization for XLM-RoBERTa..................................................................................................80                |                                                                                                            |
| 3.5.3 Passage Segmentation Strategy..................................................................................................80               |                                                                                                            |
| 3.5.4 Deduplication ...............................................................................................................................82 |                                                                                                            |
| 3.5.5 Context Expansion with Neighbor Formula..................................................................................82                     |                                                                                                            |
| 3.6 System Architecture..............................................................................................................84               |                                                                                                            |
| 3.6.1 Overall Architecture......................................................................................................................84    |                                                                                                            |
| 3.6.2 Query Validation Component.......................................................................................................86             |                                                                                                            |
| 3.6.3 Multi-Sentence Query Validation Protocol....................................................................................89                  |                                                                                                            |
| 3.6.4 Semantic Retrieval Component ...................................................................................................91              |                                                                                                            |
| 3.6.5 Lexical Retrieval Component .......................................................................................................96           |                                                                                                            |
| 3.6.6 Dual-Formula Hybrid Integration ..................................................................................................98            |                                                                                                            |
| 3.6.7 Formula Application Example ....................................................................................................101             |                                                                                                            |
| 3.6.8 Query Suggestion Component...................................................................................................105                |                                                                                                            |
| 3.6.8.1 Single-Word Theme Input                                                                                                                       | .......................................................................................................105 |

| 3.6.8.2 Single-Word Character Input...................................................................................................107        |                                                                                                                                   |
|--------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| 3.6.8.3 Entity-Based Chapter Location with Domain-Adaptive Semantic Grounding...................................108                              |                                                                                                                                   |
| Entity Detection and Validation.................................................................................................109              |                                                                                                                                   |
| Chapter Mapping Process ........................................................................................................109              |                                                                                                                                   |
| Contextual Information..............................................................................................................109          |                                                                                                                                   |
| Display Format                                                                                                                                   | .........................................................................................................................110      |
| Application Examples ...............................................................................................................110          |                                                                                                                                   |
| System Integration....................................................................................................................113        |                                                                                                                                   |
| 3.8 Evaluation Methodology .....................................................................................................116              |                                                                                                                                   |
| 3.8.1 Baseline Comparison.................................................................................................................117    |                                                                                                                                   |
| 3.8.2 Quantitative Evaluation Metrics..................................................................................................118       |                                                                                                                                   |
| 3.8.3 Qualitative Expert Review..........................................................................................................120     |                                                                                                                                   |
| 3.8.4 Thematic Distribution and Co-occurrence Analysis....................................................................121                    |                                                                                                                                   |
| 3.8.5 Query Complexity                                                                                                                           | Testing..........................................................................................................122              |
| Simple Character-Action Queries (Basic Relational                                                                                                | Patterns) ..................................................124                                                                   |
| Multi-Concept & Multi-Relation Queries (Complex Semantic Patterns) ....................................124                                       |                                                                                                                                   |
| 3.9 Implementation Details .......................................................................................................126            |                                                                                                                                   |
| 3.9.1 Development Environment and Tools........................................................................................126               |                                                                                                                                   |
| 3.9.2 Data Storage Format..................................................................................................................127   |                                                                                                                                   |
| 3.9.3 System Workflow .......................................................................................................................127 |                                                                                                                                   |
| 3.9.4 System Parameters....................................................................................................................133   |                                                                                                                                   |
| 3.10 Ethical Considerations......................................................................................................134             |                                                                                                                                   |
| 3.11 Limitations and Delimitations ............................................................................................136               |                                                                                                                                   |
| 3.11.1 Methodological Limitations.......................................................................................................136      |                                                                                                                                   |
| 3.11.2 Delimitations                                                                                                                             | ........................................................................................................................137       |
| 3.11.3 Validation Boundaries..........................................................................................................139        |                                                                                                                                   |
| 3.12 Summary..........................................................................................................................140        |                                                                                                                                   |
| References                                                                                                                                       | ..............................................................................................................................144 |

## List of Figures

| Figure 1: Theoretical Framework of a Hybrid Document Retrieval System Leveraging Semantic and                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------|
| Lexical Matching Based on Kuzi et al. (2020).................................................................................50                   |
| Figure 5: Data Collection and Preprocessing Pipeline....................................................................59                        |
| Figure 2: Conceptual Framework for the Hybrid Information Retrieval System Using XLM-                                                             |
| RoBERTa for Thematic Literary Analysis of Rizal's Novels ............................................................61                           |
| Figure 3: Domain-Adaptive Pre-training (DAPT) Validation Architecture for Semantic Grounding .64                                                  |
| Figure 4: Dataset Preparation and Annotation Process..................................................................76                          |
| Figure 6: System Architecture of the Hybrid Information Retrieval System Using XLM-RoBERTa for                                                    |
| Rizal's Novels.................................................................................................................................84 |
| Figure 7: Hybrid Retrieval System Architecture (Detailed Component View)..................................86                                      |
| Figure 8: Model Training and Fine-Tuning Process........................................................................93                        |
| Figure 9: Sentence-Level Embedding Generation..........................................................................95                         |
| Figure 10: Thematic Classification and Analysis Process.............................................................114                           |
| Figure 11: Evaluation Metrics and Framework .............................................................................117                      |
| Figure 12: System Testing and Validation Workflow.....................................................................123                         |
| Figure 13: Query Processing Execution Workflow........................................................................129                         |

## List of Tables

| Table 1: Mga Tanong sa Survey para sa Pangangailangan ng Information Retrieval System                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| (N=100) ..........................................................................................................................................17 |
| Table 2: Traditional vs. Hybrid Information Retrieval Systems .......................................................40                              |
| Table 3: Word-by-Word Weight Assignment (Example 1) ..............................................................78                                 |
| Table 4: Word-by-Word Weight Assignment (Example 2) ..............................................................79                                 |

## 1.1 Background of the Study

The literary works of José Rizal, particularly Noli Me Tangere (1887) and El Filibusterismo (1891),  are  foundational  texts  in  Philippine  national  identity  and  continue  to  serve  as  essential components of Filipino education and cultural discourse (Rizal, 1887, 1891). These novels present complex  thematic  structures  addressing  colonialism,  social  justice, education,  religion,  and revolution, interwoven through intricate narratives that reflect the socio-political landscape of Spanish colonial Philippines. The depth and complexity of these themes have sustained scholarly interest for over a century, yet traditional literary analysis methods remain predominantly manual, subjective, and limited in  their  capacity  to  systematically  identify  and  compare  thematic  patterns  across  the extensive text of both novels.

Understanding the computational challenges this study addresses requires establishing the historical  context  of  José Rizal's  foundational works and the educational reality faced by Filipino students. Noli Me Tangere (1887) follows Crisostomo Ibarra, a young Filipino reformist who returns from Europe with idealistic  dreams  of  building  a  school  in  San  Diego,  symbolizing  education  as national progress. Key characters include Maria Clara (Ibarra's beloved representing Filipino women trapped  between  tradition  and  modernity),  Padre  Damaso  (corrupt  Franciscan  friar  symbolizing clerical abuse), Elias (revolutionary advocating violent resistance), and Sisa (tragic mother driven to madness  by  systemic  injustice). The  novel critiques Spanish  colonial oppression through interconnected themes of edukasyon (education as both hope and threat), relihiyon (institutional

## CHAPTER 1 INTRODUCTION

corruption versus genuine faith), kolonyalismo (colonial control pervading Filipino life), katarungan (justice and legal injustice), and pamilya at dangal (family honor preserving identity while enabling colonial control).

El Filibusterismo (1891) serves as a darker sequel, tracing idealism's transformation into revolutionary  fervor.  Simoun,  revealed  as  the  disguised  embittered  Ibarra,  now  plots  violent overthrow  of  colonial  government  as  a  wealthy  jeweler.  Basilio,  Sisa's  surviving  son  and  now  a medical  student,  must  choose  between  peaceful  reform  and  revolutionary  violence.  The  novel explores rebolusyon vs. reporma (revolution versus reform as central debate), karahasan (violence examining  revolutionary  violence's  moral  legitimacy),  kasakiman  (greed  showing  how  colonial exploitation  corrupts  both  colonizers  and  colonized),  pagkabigo  (disillusionment  representing idealism's  death  under  sustained  oppression),  and  kamatayan  at  sakripisyo  (death  and  sacrifice questioning  whether  martyrdom  serves  revolution  or  wastes  lives).  The  two  novels  function  as complementary  halves  of  Rizal's  social  critique-Noli  Me  Tangere  diagnoses  colonial  society's diseases while El Filibusterismo explores potential cures and their costs.

The narrative and thematic complexity of these works creates specific challenges for literary analysis  and  education.  Thematic  evolution  means  the  same  theme  manifests  differently  across novels and within narrative progression-"edukasyon" shifts from Ibarra's initial optimism to later disillusionment,  requiring  contextual  understanding  that  distinguishes  these  variations.  Character transformation requires distinguishing Ibarra's characterization in Noli Me Tangere from his Simoun identity in El Filibusterismo. Multi-layered symbolism carries meaning beyond literal narrative-Sisa's madness  represents  not  just  personal  tragedy  but  systemic  violence  against  Filipino  families. Intertextual references cross-reference between novels, requiring recognition when El Filibusterismo passages reference Noli Me Tangere events.

In contemporary Filipino education, students studying these mandatory texts face persistent difficulties navigating their extensive length and complex thematic structures. Teachers report that many students struggle to complete full chapter readings, often relying on online summaries that miss  nuanced  character  development  and  thematic  interconnections.  When  assigned  to  identify passages supporting specific themes such as "edukasyon bilang pag-asa" (education as hope) or "karahasan laban sa reporma" (violence versus reform), students encounter difficulty navigating texts 15,000-20,000 words, resulting in responses lacking textual evidence or complete task abandonment. The vocabulary mismatch problem becomes evident in student search behaviors: searching  "edukasyon"  misses  semantically  related  passages  discussing  "pag-aaral"  (studying), "kaalaman" (knowledge), or "karunungan" (wisdom). Character confusion persists-questions about character  fates  and  relationships  appear  repeatedly  in  student  forums,  indicating  students  lack systematic methods to verify plot details against textual sources.

Contemporary literary analysis faces significant methodological challenges in the digital age. While  humanities  scholars  have  access  to  increasingly  sophisticated  computational  tools,  the application of these technologies to Filipino literary texts remains underdeveloped, particularly for low-resource languages like Tagalog (Cosme &amp; De Leon, 2024). Computational linguistics has rarely been applied to Filipino literary studies, despite substantial advances in natural language processing (NLP) for other languages. This gap becomes particularly evident when examining thematic analysis, where  computational  methods  could  potentially  reveal  patterns,  connections,  and  thematic distributions that manual close-reading approaches might overlook or fail to systematically quantify.

The traditional system of analyzing Noli Me Tangere and El Filibusterismo relied heavily on manual close reading methods. Researchers would physically read through both novels, manually identifying  themes, taking  handwritten  or typed  notes,  and  organizing  their  findings  using  paper-

based systems or basic word processing tools. This approach was time-consuming and limited in scope,  as  scholars  could  only  process  information  at  the  speed  of  human  reading  and  manual categorization. The identification of thematic patterns required extensive memory work and crossreferencing between different sections of the texts, often leading to potential oversights of subtle connections that appeared across large textual distances.

In contrast, the hybrid system proposed in this study combines traditional literary analysis with  modern  computational  methods.  This  approach  maintains  the  interpretive  depth  of  human scholarship while leveraging technology to handle large-scale pattern detection and  data organization.  The  hybrid  system  uses  advanced  natural  language  processing  to  automatically identify and categorize thematic elements across both novels, significantly reducing the time required for  initial  thematic  mapping.  However,  unlike  purely  automated  systems,  this  hybrid  approach preserves human oversight at critical stages. Researchers validate the machine-identified themes, provide  contextual  interpretation,  and  make  final  judgments  about  thematic  significance  and relationships. This integration allows scholars to focus their expertise on interpretation and analysis rather than on mechanical tasks of text searching and pattern counting, ultimately producing more comprehensive and nuanced insights into Rizal's works.

The evolution of information retrieval systems has demonstrated that hybrid approaches combining lexical  exact-match  methods  with  semantic  understanding  yield  superior  performance compared to single-method approaches (Gao et al., 2021; Kuzi et al., 2020). Lexical retrieval models such as BM25 excel at capturing explicit keyword correspondences but fail to recognize semantic relationships when different terminology expresses similar concepts. Conversely, neural embedding models capture semantic similarity but may overlook exact matches that carry specific significance in  literary  contexts.  The  integration  of  these  complementary  approaches  has  shown  particular

promise in multilingual contexts, where linguistic diversity compounds the challenges of information retrieval (Wiciaputra et al., 2021).

Recent advances in cross-lingual transformer models, particularly XLM-RoBERTa (Crosslingual  Language  Model  -  Robustly  Optimized  BERT  Approach),  have  demonstrated  remarkable capabilities in handling multilingual and code-switched text (Salve &amp; Tubil, 2025; Timoneda &amp; Vallejo Vera, 2023). XLM-RoBERTa's pre-training on 100 languages enables transfer learning from highresource languages to low-resource languages like Filipino, creating shared semantic spaces where conceptually  related  content  clusters  together  regardless  of  language  (Conneau  et  al.,  2019). Imperial  (2021)  demonstrated  that  BERT-based  embeddings  could  effectively  substitute  for language-specific NLP  tools in Filipino contexts where  such tools remain unavailable or underdeveloped, achieving performance improvements of up to 12.4% in F1 scores compared to classical approaches.

The application of topic modeling and thematic retrieval systems has shown transformative potential  in  religious  and  literary  text  analysis.  Ignaco  and  Ballera  (2025)  demonstrated  that traditional  keyword-based  search  achieved  0%  accuracy  in  identifying  thematically  related  Bible verses  when  exact  keywords  were  absent,  while  their  Latent  Dirichlet  Allocation  (LDA)-based approach achieved 33-67% accuracy by capturing topical relationships beyond surface-level lexical matching.  More  sophisticated  approaches  integrating  transformer-based  embeddings  with  topic modeling have shown even greater promise, with Aamir et al. (2025) demonstrating that BERTopic combined with XLM-R outperformed traditional topic modeling methods for Urdu text, a low-resource language with linguistic challenges comparable to Filipino.

The linguistic  complexity  of  Rizal's  novels  presents  unique  challenges  for  computational analysis.  Written  primarily  in  Tagalog  with  significant  Spanish  influence  and,  in  contemporary educational contexts, often studied alongside English translations, these texts exemplify the codeswitching  phenomenon  that  characterizes  much  Filipino  discourse  (Cosme  &amp;  De  Leon,  2024). Effective  computational  analysis  must  therefore  accommodate  multilingual  elements,  historical language variation, and the cultural embeddedness of thematic content. Furthermore, the novels' thematic richness-spanning colonialism (kolonyalismo), oppression (pang-aapi), education (edukasyon),  revolution  (rebolusyon),  religion  (relihiyon),  social  justice  (katarungan),  and  family honor  (pamilya  at  dangal)-requires  analytical  approaches  capable  of  capturing  both  explicit thematic markers and implicit conceptual relationships.

Existing computational literary studies have predominantly focused on Western canonical texts in high-resource languages, particularly English (Gaikwad et al., 2024). The limited application of  advanced  NLP  techniques  to  Filipino  literature  reflects  broader  patterns  of  digital  humanities research,  where  low-resource  languages  remain  underserved  despite  possessing  rich  literary traditions worthy of computational investigation. This disparity extends beyond mere technological application; it represents a gap in cultural representation within digital humanities scholarship, where Filipino literary heritage has yet to benefit from the analytical tools increasingly applied to Western texts.

The development of hybrid information retrieval systems specifically designed for Filipino literary texts addresses multiple interconnected needs. For educators and students, such systems could  enhance  comprehension  and  analysis  of  mandatory  literary  texts  by  enabling  thematic exploration beyond traditional chapter-by-chapter reading. For scholars, computational tools could facilitate comparative thematic analysis, revealing patterns across Rizal's works and enabling more

systematic  investigation  of  thematic  evolution,  distribution,  and  interconnection.  For  cultural researchers and historians, computational literary analysis offers methods to quantitatively examine how specific historical and cultural themes manifest throughout canonical texts, providing empirical foundations for interpretive claims.

The technical foundation for this research rests on several converging developments in NLP and  information  retrieval.  XLM-RoBERTa  has  demonstrated  state-of-the-art  performance  across diverse Filipino text processing tasks, achieving 97.62% precision, 95.77% F1-score, and 97.50% ROC-AUC in code-switched Tagalog-English classification when properly fine-tuned (Salve &amp; Tubil, 2025).  The  model's  architecture  enables  capture  of  contextual  semantics  and  cross-lingual relationships essential for thematic analysis. Simultaneously, advances in hybrid retrieval architectures  have  established  frameworks  for  integrating  semantic  embeddings  with  lexical matching and language-specific preprocessing (Gao et al., 2021). The availability of Filipino-specific stopword lists through resources like stopwords-iso-tl addresses a critical preprocessing requirement identified as essential for optimal performance in Tagalog text processing (Bation et al., 2017).

The significance of developing computational literary analysis tools for Filipino texts extends beyond  immediate  research  applications.  Such  tools  contribute  to  the  broader  project  of  digital humanities  inclusivity,  ensuring  that  advanced  analytical  methods  serve  diverse  linguistic  and cultural contexts rather than reinforcing existing biases toward high-resource languages. Moreover, by demonstrating effective approaches for low-resource literary text analysis, this research provides methodological  foundations  that  can  be  adapted  to  other  underserved  languages  and  literary traditions throughout Southeast Asia and globally.

The thematic structure of Rizal's novels, while extensively studied through traditional literary criticism, has not been subjected to systematic computational analysis that could reveal quantitative patterns  in  thematic  distribution,  co-occurrence,  and  textual  manifestation.  Themes  such  as edukasyon (education) appear throughout both novels but manifest differently in various narrative contexts, character dialogues, and plot developments. Computational thematic analysis could map these distributions, identify passages where multiple themes converge, and enable comparison of how themes evolve between Noli Me Tangere and El Filibusterismo. Such capabilities would provide educators and students with powerful tools for exploring these complex texts while offering scholars new methodologies for literary analysis grounded in quantitative evidence.

## Research Gap

The thematic structure of Rizal's novels, while extensively studied through traditional literary criticism, has not been subjected to systematic computational analysis that could reveal quantitative patterns  in  thematic  distribution,  co-occurrence,  and  textual  manifestation.  Themes  such  as edukasyon (education) appear throughout both novels but manifest differently in various narrative contexts, character dialogues, and plot developments. Despite advances in hybrid information retrieval systems and multilingual NLP models, no computational system has been developed to enable sophisticated thematic exploration of these foundational Filipino literary texts. This absence  represents  a  critical  gap  in  both  Filipino  digital  humanities  and  computational  literary studies, limiting the analytical tools available to researchers, educators, and students engaging with these culturally significant works. Specifically:

1. No passage-level semantic retrieval exists for Filipino literary works , despite extensive research on general documents in high-resource languages

2. Hybrid retrieval systems have not been adapted to low-resource languages like Filipino with limited NLP tools
3. Thematic retrieval  using  Filipino-capable  embeddings  remains  unexplored ,  leaving educators without computational support for theme-based analysis
4. Filipino-specific preprocessing pipelines (stopword  removal,  code-switching  handling, cultural terminology) have not been systematically integrated into literary retrieval systems
5. Domain-adaptive  pre-training  for  Filipino  literature has  not  been  validated,  despite proven effectiveness in other specialized domains

Addressing these gaps requires developing a hybrid information retrieval system that integrates semantic understanding through multilingual models with Filipino-specific linguistic preprocessing, enabling  systematic,  reproducible  thematic  analysis  that  complements  rather  than  replaces traditional literary scholarship.

## 1.2 Statement of the Problem

The analysis  of  thematic  content  in  José  Rizal's  Noli  Me  Tangere  and  El  Filibusterismo currently  relies  predominantly  on  manual  close-reading  methods  and  basic  keyword-based searching (Ctrl+F) that, while valuable for deep interpretive work, suffer from inherent limitations in systematicity, scalability, and reproducibility. Traditional keyword searching can only locate exact word  matches  and  fails  to  capture  semantic  relationships,  synonyms,  or  contextual  meanings essential for comprehensive thematic analysis. Despite the availability of advanced natural language processing technologies and the proven effectiveness of hybrid information retrieval systems in other linguistic contexts, no computational system has been developed to enable sophisticated thematic

exploration of these foundational Filipino literary texts. This absence represents a critical gap in both Filipino digital humanities and computational literary studies, limiting the analytical tools available to researchers, educators, and students engaging with these culturally significant works.

Specifically,  the  following  problems  characterize  the  current  state  of  thematic  literary analysis for Rizal's novels:

## 1. Subjectivity and inconsistency in manual thematic analysis

Traditional literary analysis of Noli Me Tangere and El Filibusterismo depends on individual readers' interpretation and memory to identify thematically relevant passages throughout the texts. This manual approach introduces subjective variation in which passages are deemed relevant to particular themes, making systematic comparison across different analytical perspectives difficult. The extensive length of both novels (approximately 15,000-20,000 words combined) exacerbates this  challenge, as readers cannot feasibly maintain comprehensive awareness of all thematically relevant  passages  throughout  both  texts  simultaneously.  Furthermore,  the  lack  of  computational tools means that claims about thematic prevalence, distribution, or co-occurrence rest on qualitative impressions rather than quantifiable evidence, limiting the empirical foundation of literary scholarship on these works.

## 2. Inability to capture semantic thematic relationships beyond explicit keyword matching

Current approaches to locating thematic content in Rizal's novels rely primarily on searching for explicit keywords or manually reading through entire texts to identify relevant passages. However, themes  manifest  through  diverse  linguistic  expressions,  metaphors,  narrative  descriptions,  and dialogues that may not contain the exact thematic keywords. For example, passages addressing

kolonyalismo (colonialism) may discuss specific manifestations of colonial oppression without using the  term  "colonialism"  itself.  Similarly,  themes  of  katarungan  (justice)  may  appear  in  narrative descriptions  of  injustice,  legal  proceedings,  or  moral  deliberations  using  varied  vocabulary.  The absence  of semantic retrieval capabilities means  that thematically relevant  passages  are systematically missed when they do not contain expected keywords, resulting in incomplete thematic analysis that fails to capture the full richness and complexity of Rizal's thematic treatment.

## 3. Lack of specialized computational literary studies tools for Filipino texts

While computational literary analysis has become increasingly sophisticated for English and other high-resource languages, Filipino literature remains severely underserved by digital humanities tools and methodologies (Cosme &amp; De Leon, 2024; Imperial, 2021). Existing information retrieval systems are not designed to handle the linguistic characteristics of Filipino literary texts, including historical language variation and culturally embedded terminology. General-purpose search tools fail to incorporate Filipino-specific linguistic preprocessing such as Tagalog stopword removal, which has been shown to improve text processing performance by 35-45% in corpus size reduction while enhancing accuracy (Ladani &amp; Desai, 2020; Bation et al., 2017). This technological gap perpetuates the marginalization of Filipino literature within computational literary studies and deprives Filipino students,  educators,  and  scholars  of  analytical  tools  comparable  to  those  available  for  Western canonical texts.

These  interconnected  problems  create  a  situation  where  one  of  the  Philippines'  most culturally significant literary corpus remains largely inaccessible to modern computational analysis methods, despite the demonstrated effectiveness of such methods in other linguistic and literary

contexts. The absence of specialized tools limits both the depth of scholarly analysis possible and the pedagogical resources available for teaching these mandatory texts in Philippine education.

## 1.3 Research Questions

This study addresses the following research questions:

1. How can a hybrid information retrieval system combining semantic embeddings and lexical matching address the subjectivity and inconsistency inherent in manual thematic analysis of Noli Me Tangere and El Filibusterismo?
2. To what extent can XLM-RoBERTa semantic embeddings capture thematic relationships beyond explicit keyword matching in identifying relevant passages across Rizal's novels?
3. How does integrating Filipino-specific linguistic preprocessing with XLM-RoBERTa's crosslingual capabilities improve retrieval accuracy for Filipino literary texts compared to generic preprocessing approaches?

## 1.4 Objectives of the Study

## General Objective

To develop and evaluate a hybrid information retrieval system integrating XLM-RoBERTa semantic  embeddings  with  lexical  matching  and  Filipino-specific  linguistic  preprocessing  for enhanced thematic literary analysis of José Rizal's Noli Me Tangere and El Filibusterismo.

## Specific Objectives

1. To design and implement a hybrid retrieval system that provides consistent, quantifiable thematic analysis of Noli Me Tangere and El Filibusterismo, reducing subjective variation through computational scoring mechanisms.
2. To  evaluate  the  effectiveness  of  XLM-RoBERTa  semantic  embeddings  in  identifying thematically  relevant  passages  that  lack  explicit  keywords,  comparing  semantic  retrieval performance against traditional keyword-based approaches.
3. To integrate Tagalog-specific stopword filtering and cross-lingual transfer learning capabilities into a specialized computational tool for Filipino literary text analysis, demonstrating improved retrieval accuracy for culturally embedded terminology.

## 1.5 Significance of the Study

The development of a hybrid information retrieval system for thematic analysis of Rizal's novels  provides  valuable  benefits  across  multiple  stakeholder  groups  in  Philippine  education, scholarship, and cultural preservation.

## For Literary Scholars and Researchers

This  study  introduces  a  computational  methodology  for  systematic  thematic  analysis  of Filipino literature. The hybrid system enables efficient retrieval of theme-relevant passages across both  Noli  Me  Tangere  and  El  Filibusterismo,  facilitating  comparative  analysis  that  would  be impractical manually. By identifying thematic co-occurrence patterns, scholars can investigate the complexity  of  Rizal's  social  commentary  and  interconnections  within  his  colonial  critique.  The

quantitative  evidence  supports  literary  arguments  with  systematic  data  rather  than  selective examples, and this methodology can extend to other works in the Filipino literary canon.

## For High School Students

Filipino students mandated to study Rizal's novels gain an accessible exploration tool for these  complex  texts.  The  system  helps  students  efficiently  locate  passages  relevant  to  specific themes  like  edukasyon  (education)  or  pang-aapi  (oppression),  enhancing  comprehension  and engagement.  This  targeted  retrieval  capability  particularly  supports  essay  writing  and  exam preparation, making lengthy texts more navigable and reducing frustration while improving learning outcomes.

## For Historians and Cultural Researchers

The computational analysis provides quantitative insights into historical and cultural theme representation  in  foundational  Philippine  texts.  Mapping  thematic  distribution  reveals  attention patterns,  thematic  evolution  between  novels,  and  theme  conjunctions  that  illuminate  Rizal's conceptualization  of  colonial  society.  This  offers  evidence  for  claims  about  priorities  in  Filipino nationalist discourse during the late Spanish colonial period and demonstrates digital humanities applications for Philippine cultural heritage materials.

## For Educators in Literature and History

The system serves as a didactic tool for enhanced classroom instruction. Educators can efficiently gather thematically organized passages for discussion and close reading, use visualizations to make abstract concepts concrete, and facilitate independent student investigations

that  promote  active  learning  and  critical  thinking.  This  integration  demonstrates  digital  literacy relevance while maintaining focus on literary understanding and cultural heritage.

## For Future Computational Linguistics Researchers

This  study  provides  methodological  contributions  extending  beyond  Rizal's  novels.  It validates that XLM-RoBERTa embeddings effectively apply to Filipino literary texts despite limited language-specific training data, supporting cross-lingual transfer learning for low-resource contexts (Imperial, 2021; Salve &amp; Tubil, 2025). The hybrid architecture establishes a framework adaptable to other low-resource languages and literary traditions. The evaluation methodology offers a model for rigorous  assessment  where  ground  truth  is  subjective,  and  demonstrates  that  advanced  NLP techniques can serve culturally diverse contexts when appropriately adapted, challenging the field's historical bias toward high-resource languages (Cosme &amp; De Leon, 2024; Asai et al., 2022).

## Broader Cultural Impact

The  study  contributes  to  cultural  preservation  and  digital  heritage  accessibility  in  the Philippines.  By  developing  computational  tools  for  Filipino  literary  analysis,  it  ensures  Philippine cultural heritage remains accessible and relevant digitally. This methodology can extend to digitized collections of Philippine literature, historical documents, and cultural texts, enabling sophisticated search and analysis capabilities that preserve Filipino cultural heritage while increasing accessibility for both Filipino and international audiences.

## 1.5.6 Application Utilization and Impact Metrics

The practical significance of this research is evidenced through a comprehensive needs assessment survey conducted at Carlos Botong Francisco Memorial National Highschool during the

planning phase of system development. This assessment aimed to validate the necessity of a hybrid information  retrieval  system  for  Filipino  literary  texts  and  to  understand  student  challenges  in studying Rizal's novels.

## Survey Methodology and Participant Demographics

The survey was administered to 100 students (50 Grade 9, 50 Grade 10) enrolled in Filipino subjects at Carlos Botong Francisco Memorial National Highschool, Angono Rizal, during November 2025.  The  participant  group  comprised  58  female  students  and  42  male  students,  ages  14-16, representing diverse socioeconomic backgrounds typical of Philippine public secondary education. All  participants  had  been  studying  Noli  Me  Tangere  and  El  Filibusterismo  as  part  of  mandatory curriculum requirements under Republic Act 1425 (Rizal Law). The survey employed a 5-point Likert scale where 1 = Lubos na Hindi Sumasang-ayon (Strongly Disagree), 2 = Hindi Sumasang-ayon (Disagree),  3  =  Walang  Kinikilingan  (Neutral),  4  =  Sumasang-ayon  (Agree),  and  5  =  Lubos  na Sumasang-ayon (Strongly Agree). All 100 students completed the survey during designated class time, achieving 100% response rate.

## Quantitative Survey Results

The  survey  results  demonstrate  overwhelming  student  recognition  of  the  need  for computational  support  in  literary  analysis,  with  99%  of  students  expressing  agreement  that  an information retrieval system would significantly benefit their study of Rizal's novels. Detailed results for each survey item appear in Table 1.

Table 1:  Mga Tanong sa Survey para sa Pangangailangan ng Information Retrieval System (N=100)

| Tanong sa Survey                                                                                                                                                       | Lubos na di Sumasang- ayon (1)   | Hindi Sumasang- ayon (2)   | Walang Kinikilingan (3)   | Sumasang- ayon (4)   | Lubos na Sumasangayon (5)   |   Mean |   SD |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|----------------------------|---------------------------|----------------------|-----------------------------|--------|------|
| 1. Makakatulong sa akin ang information retrieval system upang mabilis na makahanap ng mga talata tungkol sa mga tema tulad ng edukasyon, katarungan, at kolonyalismo. | 0 (0%)                           | 0 (0%)                     | 2 (2%)                    | 28 (28%)             | 70 (70%)                    |   4.68 | 0.49 |
| 2. Mas magiging interesado ako sa pag- aaral ng Noli at El Fili kung may teknolohiya akong magagamit para sa paghahanap ng mga character at tema.                      | 0 (0%)                           | 1 (1%)                     | 6 (6%)                    | 35 (35%)             | 58 (58%)                    |   4.49 | 0.68 |
| 3. Kailangan ko ng tulong ng sistema para masubaybayan ang mga tauhan mula sa Noli Me Tangere hanggang El Filibusterismo.                                              | 0 (0%)                           | 0 (0%)                     | 4 (4%)                    | 38 (38%)             | 58 (58%)                    |   4.54 | 0.58 |
| 4. Makakatipid ako ng oras sa paggawa ng essays at reading journals kung may search system para sa mga nobela ni Rizal.                                                | 0 (0%)                           | 0 (0%)                     | 3 (3%)                    | 32 (32%)             | 65 (65%)                    |   4.62 | 0.54 |
| 5. Gusto kong malaman agad kung aling kabanata ang may kaugnayan sa tema na aking hinahanap nang hindi na kailangang basahin ang buong nobela.                         | 0 (0%)                           | 1 (1%)                     | 5 (5%)                    | 41 (41%)             | 53 (53%)                    |   4.46 | 0.65 |

Overall System Need Assessment: When responses were aggregated, 99 students (99%) rated their need for such a system as 4 or 5 (kailangan o lubos na kailangan), with an overall mean rating of 4.56 (SD = 0.59). Only one student (1%) provided a neutral rating (3), with zero students expressing disagreement regarding system utility.

## Analysis of Survey Findings

The survey results reveal consistent patterns across multiple dimensions of student need. The highest perceived benefit (mean = 4.68) emerged for fast thematic search capability, with 70% of students strongly agreeing that the system would help them quickly locate passages about themes such as education, justice, and colonialism. This finding directly validates the vocabulary mismatch problem identified in the problem statement, where students struggle to find thematically relevant passages using current manual or keyword-only methods.

Time-saving value received strong endorsement (mean = 4.62), with 65% strongly agreeing that the system would save time in completing essays and reading journals. This response aligns with teacher reports that students often abandon analytical tasks perceived as too time-consuming when passage location requires manual navigation through 15,000-20,000 words combined pages. The  time  efficiency  benefit  represents  a  critical  factor  for  students  managing  heavy  academic workloads across multiple subjects within the constraints of Philippine public-school schedules.

Character tracking need (mean = 4.54) indicates that 58% of students strongly agree they require  systematic  support  to  follow  characters  across  both  novels.  This  validates  documented confusion about character arcs, relationships, and transformations-exemplified by the persistent question "Namatay ba si Basilio?" appearing in student forums despite Basilio's survival being a

crucial  plot  element.  The  character  tracking  capability  addresses  not  merely  convenience  but  a fundamental  comprehension  barrier  preventing  students  from  understanding  narrative  continuity between Noli Me Tangere and El Filibusterismo.

Increased motivation through technology (mean = 4.49) demonstrates that 58% of students strongly agree their interest in Filipino literature would increase with access to modern retrieval tools. This finding has important pedagogical implications: technology integration serves not only functional purposes (finding passages efficiently) but also motivational purposes (increasing engagement with culturally  significant  texts).  The  6%  neutral  responses  on  this  item,  higher  than  other  questions, suggest individual variation in how students perceive technology's role in literary study-a diversity the system must accommodate through optional rather than mandatory use.

Direct chapter relevance identification (mean = 4.46) shows that 53% strongly agree they want to immediately know which chapters relate to search themes without reading entire novels. This response reflects a pragmatic student approach to literary analysis: rather than comprehensive linear reading  (which  fewer  than  30%  complete  according  to  teacher  reports),  students  seek  efficient pathways  to  locate  information  relevant  to specific  analytical  tasks.  The  system's  ranking mechanisms,  which  identify  chapters  with  highest  thematic  relevance,  directly  address  this expressed need.

## Demographic and Accessibility Considerations

Survey analysis revealed no statistically significant differences in perceived need across gender (female mean: 4.58, male mean: 4.53, p &gt; 0.05) or grade level (Grade 9 mean: 4.55, Grade 10 mean: 4.57, p &gt; 0.05), suggesting that computational support addresses universal rather than demographically  specific  challenges.  Informal  follow-up  questions  regarding  technology  access

confirmed that approximately 95% of surveyed students own smartphones with internet capability, validating the mobile-first design approach. Students from households without broadband internet (estimated 40-45% based on socioeconomic indicators) expressed equivalent interest in the system, provided it functions efficiently on mobile data networks-a requirement incorporated into system design through network optimization and efficient query processing.

## Qualitative Feedback from Open-Ended Responses

The survey included an optional open-ended question: "Ano ang pinakamahirap sa pagaaral ng Noli at El Fili?" (What is most difficult about studying Noli and El Fili?). Sixty-three students provided google form responses, revealing consistent themes:

Time and Length Challenges (mentioned by 42 students, 67%): "Sobrang haba ng nobela, hindi ko matapos basahin lahat." (The novels are too long, I can't finish reading everything.) "Walang oras para basahin ang buong chapter bago klase." (No time to read whole chapters before class.)

Theme Identification  Difficulties  (mentioned  by  28  students,  44%): "Hindi  ko  alam  saan makikita yung mga tema sa bawat kabanata." (I don't know where to find the themes in each chapter.) "Mahirap hanapin yung example ng katarungan o edukasyon sa nobela." (Hard to find examples of justice or education in the novels.)

Character Confusion (mentioned by 31 students, 49%): "Nakakalito  kung sino-sino yung mga tauhan, marami silang pangalan." (Confusing who the characters are, they have many names.) "Hindi ko alam kung ano nangyari kay Basilio sa dalawang nobela." (I don't know what happened to Basilio in both novels.)

Evidence Location Problems (mentioned by 24 students, 38%): "Hirap maghanap ng mga quote para sa essay." (Hard to find quotes for essays.) "Kailangan ng textual evidence pero di ko alam saan hahanapin." (Need textual evidence but don't know where to find it.)

These qualitative responses corroborate the quantitative findings and provide rich contextual understanding  of  student  challenges  that  computational  retrieval  systems  can  address.  The frequency with which students mention time constraints, difficulty locating themes, and challenges finding  textual  evidence  validates  the  research  problem  statement  and  establishes  clear  user requirements for system design.

## Synthesis of Needs Assessment

The  comprehensive  survey  data  encompassing  Likert-scale quantitative responses, demographic  analysis,  and  qualitative  open-ended  feedback  establishes  a  strong  empirical foundation for the necessity of developing a hybrid information retrieval system for Filipino literary texts. The 99% agreement rate regarding system utility, combined with mean scores consistently above  4.4  across  all  measured  dimensions,  demonstrates  that  student  need  for  computational support is not marginal but overwhelming. The alignment between quantitative ratings and qualitative explanations strengthens confidence that the survey captured genuine student experiences rather than mere acquiescence to researcher expectations.

These findings validate that the challenges identified in the problem statement subjectivity and inconsistency in manual analysis, inability to capture semantic relationships beyond keywords, and lack of specialized tools for Filipino texts-reflect authentic educational difficulties experienced by  the  target  user  population.  The  survey  results  provide  empirical  justification  for  the  research investment  required  to  develop,  implement,  and  evaluate  the  proposed  hybrid  retrieval  system,

establishing that such a system addresses documented needs rather than offering solutions seeking problems.

Furthermore, the near-universal smartphone ownership and expressed willingness to use mobile-accessible literary tools confirm the technological feasibility of deployment within Philippine educational  contexts.  The  system's  anticipated  utilization  is  supported  not  merely  by  researcher assumptions but by direct evidence from 100 students representing the target user demographic, providing confidence that the completed system will achieve meaningful adoption and educational impact when deployed at scale.

## 1.6 Scope and Delimitation

This study focuses specifically on the development and evaluation of a hybrid information retrieval system for thematic analysis of two novels by José Rizal: Noli Me Tangere (1887) and El Filibusterismo (1891). The scope encompasses the following elements:

## Textual Corpus

The study utilizes Filipino language summary versions (buod) of both novels from publicly available educational resources. Specifically, the Noli Me Tangere summary from PinoyCollection.com (2017) and the El Filibusterismo summary from Noypi.com.ph (2019). These summarized  versions  represent  accessible  educational  materials  commonly  used  by  Filipino students and provide manageable text lengths for system development and testing.

## Thematic Focus

The  research  examines  multiple  major  themes  central  to  Rizal's  novels,  identified  from established literary analysis sources. Themes for Noli Me Tangere were derived from GradeSaver (2024) and Francisco (2025), while themes for El Filibusterismo were sourced from LitCharts (2025) and GradeSaver (2023). These themes were selected based on their prominence in established literary scholarship and their significance in Philippine cultural and historical discourse.

## Technical Scope

The system architecture integrates XLM-RoBERTa multilingual embeddings (specifically the xlm-roberta-base model) for semantic representation, combined with lexical keyword matching and Tagalog-specific  stopword  filtering  using  the  stopwords-iso-tl  resource.  The  retrieval  process operates at the passage level, with passages defined as semantically coherent text segments of variable  length.  The  scoring  mechanism  incorporates  semantic  similarity  (via  cosine  similarity  of XLM-RoBERTa embeddings), lexical overlap (via keyword presence detection), and length-based normalization to balance relevance against passage length.

## Evaluation Methodology

System performance is evaluated through retrieval accuracy assessment across multiple themes, comparing the hybrid approach against baseline keyword-only retrieval. Evaluation includes analysis  of  precision  and  recall,  examination  of  thematic  distribution  patterns,  and  qualitative assessment  of  retrieved  passage  relevance  through  expert  review.  The  study  also  includes comparative analysis of how different scoring components contribute to overall retrieval effectiveness.

## Delimitations

## Use of Summary Versions

The study does not utilize  the  complete,  full-length  versions  of  Noli  Me  Tangere  and  El Filibusterismo. This limitation is due to the original texts containing mixed Spanish-Tagalog content and  their  considerable  length,  which  would  complicate  processing  and  analysis.  Instead,  the research focuses on Filipino language summaries that provide comprehensive coverage of major plot points and themes while maintaining linguistic consistency and manageable scope.

## Model Training Limitations

The study does not include automated fine-tuning of the XLM-RoBERTa model on Filipino literary texts due to the limited availability of annotated Filipino literature datasets suitable for such fine-tuning.  Instead,  the  research  uses  the  pre-trained  model's  cross-lingual  transfer  capabilities, following  approaches  validated  by  Imperial  (2021)  and  Salve  and  Tubil  (2025)  for  Filipino  text processing.

## Scope of Literary Coverage

The system does not perform automated theme identification or discovery; rather, it retrieves passages relevant to predefined themes specified by users. The research does not extend to other works  by  Rizal  or  other  Filipino  authors,  maintaining  focus  on  the  two  novels  most  central  to Philippine literary education.

## System Functionality Boundaries

The system does not provide literary interpretation or critical analysis of retrieved passages; it serves as a retrieval and exploration tool to support rather than replace human literary scholarship. The evaluation does not include large-scale user studies with students or educators, focusing instead on  technical  performance  assessment  and  expert  validation.  Future  work  may  extend  the methodology to additional Filipino literary texts and conduct comprehensive user studies to assess effectiveness in educational contexts.

## 1.7 Definition of terms

The following terms are defined as they are used in this study:

Cosine Similarity: A metric measuring the similarity between two vectors by calculating the cosine of  the  angle  between  them,  ranging  from  -1  to  1,  commonly  used  to  assess  semantic  similarity between text embeddings.

Cross-lingual  Transfer  Learning: The  application  of  knowledge  learned  from  high-resource languages to low-resource languages through shared multilingual representations, enabling effective NLP for languages with limited training data (Wiciaputra et al., 2021; Imperial, 2021).

Digital  Humanities: An interdisciplinary field applying computational methods and digital technologies  to  humanities  research,  including  literary  studies,  history,  and  cultural  analysis (Gaikwad et al., 2024).

Filipino Literature: Literary works written in Filipino (standardized Tagalog) or Philippine languages, encompassing novels, poetry, essays, and other forms, with particular focus on works contributing to Philippine national identity and cultural heritage.

Hybrid Information Retrieval: An approach combining multiple retrieval methodologies, specifically integrating semantic embedding-based retrieval with lexical keyword matching to use complementary strengths of both approaches (Gao et al., 2021; Kuzi et al., 2020).

Lexical Matching: Text retrieval based on exact or approximate matching of keywords and terms, identifying passages that contain specified words or phrases regardless of semantic interpretation.

Low-resource Language: A language with limited availability of digital corpora, annotated datasets, and  NLP  tools  compared  to  high-resource  languages  like  English,  presenting  challenges  for computational linguistic applications (Asai et al., 2022; Cosme &amp; De Leon, 2024).

Multi-component Scoring: A ranking approach that combines multiple relevance signals-such as semantic  similarity,  lexical  overlap,  and  length  normalization-into  a  unified  score  for  ranking retrieved passages (Yoo et al., 2024).

Passage Retrieval: The  task  of  identifying  and  ranking  text  segments  (passages)  from  a  larger corpus based on their relevance to a query or theme, operating at a granularity between sentencelevel and document-level retrieval.

Semantic  Embeddings: Dense  vector  representations  of  text  that  encode  semantic  meaning, enabling computation of semantic similarity between texts based on conceptual relatedness rather than lexical overlap (Imperial, 2021; Gaikwad et al., 2024).

Stopwords: Common words in a language that carry minimal semantic content and are typically removed during text preprocessing to improve retrieval performance, such as "ang," "mga," "si," and "dahil" in Tagalog (Bation et al., 2017; Ladani &amp; Desai, 2020).

TF-IDF  (Term  Frequency-Inverse  Document  Frequency): A  numerical  statistic  reflecting  the importance of a word in a document relative to a corpus, used to weight terms in information retrieval and text mining (Bation et al., 2017; AlShammari, 2023).

Thematic Analysis: The identification and interpretation of recurring themes-abstract concepts, topics, or ideas-throughout literary texts, examining how themes manifest, develop, and interrelate across narratives (Ignaco &amp; Ballera, 2025; Aamir et al., 2025).

Thematic Distribution: The pattern of how a theme appears throughout a text or corpus, including frequency, concentration in specific sections, and co-occurrence with other themes.

XLM-RoBERTa (Cross-lingual  Language  Model  -  Robustly  Optimized  BERT  Approach): A lingual  transfer  learning  and  generates  contextual  embeddings  for  text  in  multiple  languages multilingual transformer-based language model pre-trained on 100 languages that enables crossincluding Filipino (Conneau et al., 2019; Salve &amp; Tubil, 2025).

## 1.8 Basis of the Study

This  research  is  grounded  in  multiple  intersecting  foundations  that  establish  both  its necessity  and  feasibility:  empirical  evidence  from  Philippine  educational  contexts,  technological advances in multilingual natural language processing, educational mandates codified in Philippine law,  theoretical  frameworks  from  information  retrieval  science,  and  cultural  equity  considerations within digital humanities scholarship.

## 1.8.1 Empirical Basis: Documented Limitations in Filipino Literary Education

The study's foundation rests on empirically documented challenges in Philippine secondary education. Teachers report that fewer than 30% of students complete assigned chapter readings, with  the  majority  relying  on  online  summaries  that  omit  crucial  narrative  details  and  thematic nuances.  When  assignments  require  locating  passages  supporting  specific  themes  such  as "edukasyon bilang pag-asa" or "karahasan laban sa reporma," students struggle to navigate texts exceeding  15,000-20,000  words  combined  pages,  resulting  in  vague  responses  lacking  textual evidence  or  complete  task  abandonment.  The  vocabulary  mismatch  problem  becomes  evident through student search behaviors: searching "edukasyon" fails to retrieve content discussing "pagaaral,"  "kaalaman,"  or  "karunungan"-all  thematically  relevant  but  lexically  distinct.  Character tracking difficulties reveal persistent confusion about character relationships and transformations, with questions about character fates appearing repeatedly in student forums despite plot elements being clearly  established  in  the  texts.  These  documented  educational  challenges  reflect  broader patterns in Philippine literary education where students lack systematic methods to verify plot details or locate thematic content efficiently.

## 1.8.2 Technological Basis: Advances in Multilingual NLP

The study's feasibility rests on recent breakthroughs in cross-lingual language modeling that enable  sophisticated  NLP  for  low-resource  languages  like  Filipino.  Multiple  independent  studies establish XLM-RoBERTa as optimal for Filipino text processing. Imperial (2021) demonstrated 12.4% F1-score improvement over classical approaches on Filipino readability assessment, establishing that BERT embeddings substitute effectively for unavailable language-specific NLP tools. Cosme and De Leon (2024) achieved 0.84 accuracy on Filipino-English code-switched sentiment analysis,

dramatically outperforming lexicon-based monolingual tools. Salve and Tubil (2025) attained 97.62% precision and 95.77% F1-score on Taglish classification, demonstrating XLM-RoBERTa's superior handling of bilingual Filipino text. These findings establish that XLM-RoBERTa's pre-training on 100 languages creates shared semantic spaces enabling effective Filipino processing through crosslingual transfer learning. The integration of semantic and lexical approaches through hybrid retrieval is validated by Gao et al. (2021), who demonstrated that semantic embeddings trained to capture what lexical  models miss  achieve superior performance through complementary integration, and Kuzi et al. (2020), who showed that hybrid models significantly improve recall during retrieval and precision  during  reranking.  Krieger  et  al.  (2022)  established  that  pre-trained  models  benefit substantially  from  domain-specific  adaptation,  with  performance  gains  proportional  to  domain alignment between pre-training and target tasks, suggesting XLM-RoBERTa's general capabilities can be enhanced through semantic grounding in Rizal's novels specifically.

## 1.8.3 Educational Basis: Mandated Curriculum Requirements

The  study  addresses  concrete  educational  needs  arising  from  Philippine  curriculum mandates. Republic Act 1425 (Rizal Law), enacted in 1956, legally mandates that courses on José Rizal's  life,  works,  and  writings  be  included  in  curricula  of  all  Philippine  schools,  colleges,  and universities. Section 1 explicitly requires that Noli Me Tangere and El Filibusterismo be included in public and private education. At the secondary level, the K-12 curriculum integrates Rizal's novels into Filipino literature courses, requiring students to demonstrate abilities to analyze literary texts for theme development, compare thematic treatments across works, evaluate how historical contexts shape  literature,  and  construct  evidence-based  interpretations  supported  by  textual  citations. Teachers and students report persistent difficulties including text length intimidation with combined 400+ pages overwhelming students managing heavy academic workloads; thematic complexity with

interconnected themes challenging synthesis; limited computational support with no existing tools helping students efficiently locate and compare passages; and assessment difficulties when essay questions  require  textual  evidence  across  both  novels.  These  educational  imperatives  establish practical urgency for computational tools that reduce barriers to engaging with culturally foundational texts.

## 1.8.4 Theoretical Basis: Information Retrieval Frameworks

The  study's  architecture  rests  on  established  theoretical  frameworks  from  information retrieval science. Hybrid Retrieval Theory (Gao et al., 2021; Kuzi et al., 2020) proposes that optimal retrieval  emerges  from  integrating  complementary  lexical  and  semantic  approaches  rather  than treating them as competing alternatives. Cross-lingual Transfer Learning Theory (Conneau et al., 2019) explains how multilingual transformer models develop language-agnostic representations that transfer  effectively  across  typologically  diverse  languages,  enabling  sophisticated  NLP  for  lowresource  languages.  Thematic  Coherence  Theory  (Yoo  et  al.,  2024;  Ignaco  &amp;  Ballera,  2025) establishes that document relevance operates at multiple levels-lexical, semantic, and thematiceach contributing distinct information requiring explicit modeling and appropriate weighting. DomainAdaptive Pre-training Theory (Krieger et al., 2022) demonstrates that pre-trained models benefit from domain-specific  adaptation  when  pre-training  and  target  domains  share  structural  similarities, improving task performance while preventing false retrievals from semantically misaligned queries. These frameworks collectively  guide  the  dual-formula  hybrid  architecture,  XLM-RoBERTa  model selection, multi-component  scoring mechanisms,  and  domain-adaptive  validation thresholds implemented in this study.

## 1.8.5 Cultural Basis: Digital Humanities Inclusivity

The study addresses fundamental disparity in digital  humanities where  Filipino  literature remains severely underserved by computational analysis tools despite the Philippines' rich literary tradition.  Computational  literary  studies  demonstrate  overwhelming  concentration  on  Englishlanguage texts and European traditions, with over 95% of published research focusing on highresource languages (Gaikwad et al., 2024). Filipino lacks basic NLP tools such as part-of-speech taggers, dependency parsers, and named entity recognizers available for English. This technological gap perpetuates cultural marginalization within digital scholarship. Developing computational tools for  Filipino  literature  ensures  Philippine  literary  heritage  remains  accessible  in  the  digital  age, provides  methodological  foundations  adaptable  to  other  Southeast  Asian  literary  traditions,  and challenges the field's historical bias toward Western canonical texts. The study contributes to broader digital humanities inclusivity by demonstrating that advanced analytical methods can serve diverse linguistic  and  cultural  contexts  rather  than  reinforcing  existing  biases  toward  high-resource languages.

## 1.8.6 Synthesis: Convergent Foundations

These  multiple  bases  converge  to  establish  both  necessity  and  feasibility.  Necessity emerges from empirical evidence documenting severe limitations in current approaches, RA 1425 legal  mandates  creating  curriculum  requirements,  pedagogical  standards  demanding  evidencebased analysis, and cultural equity considerations calling for computational tools serving diverse literatures. Feasibility emerges from XLM-RoBERTa's demonstrated effectiveness for Filipino text processing,  validated  hybrid  retrieval  architectures  providing  proven  frameworks,  theoretical foundations guiding principled system design, and domain-adaptive pre-training enabling semantic

grounding in specific literary works. The intersection of these foundations establishes that this study addresses  a  genuine  research  gap  at  an  opportune  technological  moment,  with  clear  practical applications  and  broader  implications  for  digital  humanities  inclusivity.  This  multi-layered  basis justifies the research investment and positions the study within larger conversations about language equity in computational scholarship, technology's role in education, and the future of literary studies in digital contexts.

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

## References

Aamir, N., Raza, A., Iqbal, M. W., Hamid, K., Nazir, Z., Asif, A., Hussain, S., &amp; Muhammad, H. A. B. (2025). Topic modeling empowered by a deep learning framework integrating BERTopic, XLM-R, and GPT. Journal of Computing and Biomedical Informatics . https://www.jcbi.org/index.php/Main/article/view/957

AlShammari,  A.  F.  (2023).  Implementation  of  text  similarity  using  word  frequency  and  cosine similarity in Python. International Journal of Computer Applications , 185 (36), 54-59. https://doi.org/10.5120/ijca2023923160

Asai, A., Longpre, S., Kasai, J., Lee, C.-H., Zhang, R., Hu, J., Yamada, I., Clark, J. H., &amp; Choi, E. (2022). MIA 2022 shared task: Evaluating cross-lingual open-retrieval question answering for 16 diverse languages. ArXiv.org . https://arxiv.org/abs/2207.00758

Bation,  A.  D.  C.,  Manguilimotan,  E.  Q.,  &amp;  Vicente,  A.  J.  O.  (2017).  Automatic  categorization  of Tagalog documents using support vector machines. https://www.academia.edu/68891958/Automatic\_Categorization\_of\_Tagalog\_Documents\_Using\_S upport\_Vector\_Machines

Celerdata.com. (n.d.). Semantic search vs keyword search. https://celerdata.com/glossary/semantic-search-vs-keyword-search

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., &amp; Stoyanov, V. (2019). Unsupervised cross-lingual representation learning at scale. arXiv. https://arxiv.org/abs/1911.02116

Cosme, C. J., &amp; De Leon, M. M. (2024). Sentiment analysis of code-switched Filipino-English product and service reviews using transformers-based large language models. In Lecture Notes in Computer Science (pp. 146-160). Springer. https://doi.org/10.1007/978-981-99-8349-0\_11

Couchbase. (2024). Hybrid search . https://www.couchbase.com/blog/hybrid-search/

Dev.to. (2024). RAG retrieval performance enhancement practices: Detailed explanation of hybrid retrieval  and  self-query  techniques .  https://dev.to/jamesli/rag-retrieval-performance-enhancementpractices-detailed-explanation-of-hybrid-retrieval-and-self-query-techniques-59ja

Emergentmind.com. (n.d.). Bi-encoder and cross-encoder architectures . https://www.emergentmind.com/topics/bi-encoder-and-cross-encoder-architectures

Francisco, K. D. (2025). Noli Me Tangere values . Scribd. https://www.scribd.com/document/527244438/Noli-Me-Tangere-Values

Fuzzylabs.ai.  (n.d.). Improving  RAG  performance:  Hybrid  search .  https://www.fuzzylabs.ai/blogpost/improving-rag-performance-hybrid-search

Gaikwad, M., Shinde, G. R., Mahalle, P. A., Sable, N., &amp; Kharate, N. (2024). Recent advances in text documents clustering.

https://openurl.ebsco.com/EPDB%3Agcd%3A15%3A17245119/detailv2?sid=ebsco%3Aplink%3As cholar&amp;id=ebsco%3Agcd%3A181715094&amp;crl=c

Gao, L., Dai, Z., Chen, T., Fan, Z., Van Durme, B., &amp; Callan, J. (2021). Complement lexical retrieval model  with  semantic  residual  embeddings. Lecture  Notes  in  Computer  Science ,  146-160. https://doi.org/10.1007/978-3-030-72113-8\_10

Google  Cloud.  (2024). About  hybrid  search .  https://docs.cloud.google.com/vertex-ai/docs/vectorsearch/about-hybrid-search

GradeSaver. (2023). El Filibusterismo themes . https://www.gradesaver.com/el-filibusterismo/studyguide/themes

GradeSaver.  (2024,  June  20). Noli  Me  Tangere  themes .  https://www.gradesaver.com/noli-metangere/study-guide/themes

Hendrickx,  I.,  Daelemans,  W.,  Marsi,  E.,  &amp;  Krahmer,  E.  (2009).  Reducing  redundancy  in  multidocument  summarization  using  lexical  semantic  similarity. Proceedings  of  the  Workshop  on Language Technology for Cultural Heritage Data (LaTeCH 2009) , 63-63. https://doi.org/10.3115/1708155.1708167

Ignaco, M. A. E., &amp; Ballera, M. A. (2025). Enhancing Bible verse search through topic modeling: An LDA-based approach. https://doi.org/10.1109/ictcs65341.2025.10989430

Imperial,  J.  M.  (2021).  BERT  embeddings  for  automatic  readability  assessment. ArXiv.org . https://arxiv.org/abs/2106.07935

Iterate.ai. (n.d.). BM25 ranking algorithm . https://www.iterate.ai/ai-glossary/bm25-ranking-algorithm

Jain, A., Jain, A., Chauhan, N., Singh, V., &amp; Thakur, N. (2017). Information Retrieval using Cosine and  Jaccard  Similarity  Measures  in  Vector  Space  Model.  International  Journal  of  Computer Applications, 164(6), 28-30. https://doi.org/10.5120/ijca2017913699

Kalykulova, A., &amp; Alzhanov, A. (2025). Term-unigram extractions using embedding-based filtering.

In

2025 IEEE 5th International Conference on Smart Information Systems and Technologies (SIST)

(pp. 1-6). IEEE. https://doi.org/10.1109/sist61657.2025.11139174

Krieger, J.-D.,  Spinde, T., Ruas, T., Kulshrestha, J., &amp; Gipp, B. (2022). A Domain-adaptive Pre- training Approach for Language Bias Detection in News.

2022.

https://doi.org/10.1145/3529372.3530932

Kuzi, S., Zhang, M., Li, C., Bendersky, M., &amp; Najork, M. (2020). Leveraging semantic and lexical matching  to  improve  the  recall  of  document  retrieval  systems:  A  hybrid  approach.

https://arxiv.org/abs/2010.01195

Ladani, D. J., &amp; Desai, N. P. (2020, April 23). Stopword identification and removal techniques on TC and IR applications: A survey. https://doi.org/10.1109/icaccs48705.2020.9074166

LitCharts. (2025). El Filibusterismo themes . https://www.litcharts.com/lit/el-filibusterismo/themes

Meilisearch. (2023). Hybrid search . https://www.meilisearch.com/blog/hybrid-search

Melamud, O., Levy, O., &amp; Dagan, I. (2015). A Simple Word Embedding Model for Lexical Substitution. Proceedings  of  the  1st  Workshop  on  Vector  Space  Modeling  for  Natural  Language  Processing. https://doi.org/10.3115/v1/w15-1501

Microsoft Learn. (2024). Hybrid search ranking . https://learn.microsoft.com/enus/azure/search/hybrid-search-ranking

Mtham8.github.io. (n.d.). TF-IDF

. https://mtham8.github.io/tf-idf/

.

ArXiv.org

Proceedings of the ACM Web Conference

Noypi.com.ph. (2019, October 24). El Filibusterismo buod ng bawat kabanata 1-39 w/ talasalitaan . https://noypi.com.ph/el-filibusterismo-buod/

OpenSearch. (2024). Introducing reciprocal rank fusion: Hybrid search . https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/

PinoyCollection.com. (2017, November 30). Noli Me Tangere buod ng bawat kabanata 1-64 (with talasalitaan) . https://pinoycollection.com/noli-me-tangere-buod/

Rizal, J. (1887). Noli Me Tangere. Berlin.

Rizal, J. (1891). El Filibusterismo. Ghent.

Salve, H. N., &amp; Tubil, P. N. T. (2025). Bilingual code-switching using XLM-RoBERTa. TechRxiv . https://www.techrxiv.org/users/964105/articles/1332748-bilingual-code-switching-using-xlm-roberta

Sciencedirect.com. (n.d.). Boolean search . https://www.sciencedirect.com/topics/computerscience/boolean-search

Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., &amp; Zaharia, M. (2021). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. ArXiv.org. https://arxiv.org/abs/2112.01488 Siriwardhana, S., Weerasekera, R., Wen, E., Kaluarachchi, T., Rana, R., &amp; Nanayakkara, S. (2024). Blended RAG: Improving RAG accuracy with semantic search and hybrid query-based retrievers.

arXiv . https://arxiv.org/html/2404.07220v2

Timoneda, J. C., &amp; Vallejo Vera, S. (2023). BERT, RoBERTa, or DeBERTa? Comparing performance across transformers models in political science text. Journal of Politics . https://doi.org/10.1086/730737

University of South Africa. (2021). The vocabulary problem in information retrieval. http://www.scielo.org.za/scielo.php?script=sci\_arttext&amp;pid=S2304-88632021000200006

University of Waterloo. (2024). Dense retrieval models [Thesis]. https://uwspace.uwaterloo.ca/items/05e9d9fb-dac1-4ff8-ae04-a9a6a83933c5

Wiciaputra,  Y.  K.,  Young,  J.  C.,  &amp;  Rusli,  A.  (2021).  Bilingual  text  classification  in  English  and Indonesian via transfer learning using XLM-RoBERTa. https://www.researchgate.net/publication/357397098\_Bilingual\_Text\_Classification\_in\_English\_an d\_Indonesian\_via\_Transfer\_Learning\_using\_XLM-RoBERTa

Whisperit.ai. (n.d.). Semantic search vs keyword search: What's the difference and why it matters . https://whisperit.ai/blog/semantic-search-vs-keyword-search

Yoo, E., Kim, G., &amp; Kang, S. (2024). Summary-sentence level hierarchical supervision for re-ranking model of two-stage abstractive summarization framework. Mathematics , 12 (4), 521. https://doi.org/10.3390/math12040521

Zilliz.com.  (2024). BGE-M3  and  SPLADE:  Two  machine  learning  models  for  generating  sparse embeddings . https://zilliz.com/learn/bge-m3-and-splade-two-machine-learning-models-forgenerating-sparse-embeddings

THEMATIC SEARCH ENGINE

THEMATIC SEARCH ENGINE

HIGHSCHOOL O

search phrases, words, chapters, etc..

HIGHSCHOOL C

Chapter 11: The Rulers of the Town wielded

Punishine fose who opposed them-

When clickes

Chapter 11

lers of the Town wieided

power

OVE

townspeople, colecting taxes unjustly and punishing those who opposed them-l

Father Damaso insulted the natives\_'

Chapter 9

\_Father Damaso insulted the natives.

bi

"Word" Analysis bi

search nhracas marde chantars

## WIRE FRAME

CHAPTER NUMBER...

Chapter 7: Simoun teaching

priests manipulated

ABOUT US

ABOUT US

Chapter 11

townspeopie. wielded power over the

<!-- image -->

<!-- image -->