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

