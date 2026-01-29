"""
Configuration constants for the Rizal Exploration Engine.
"""

CORE_ENTITIES = {
    'ibarra', 'crisostomo', 'crisóstomo', 'simoun', 'maria', 'clara', 'elias',
    'basilio', 'sisa', 'kapitan', 'tiago', 'tiyago', 'tasio', 'pilosopo',
    'juli', 'isagani', 'paulita', 'salvi', 'damaso', 'camorra', 'camora',
    'kabesang', 'tales', 'kundiman', 'ben', 'zayb', 'donya', 'victorina'
}

# Similarity and Search Thresholds
MIN_SEMANTIC_THRESHOLD = 0.20
THEMATIC_THRESHOLD = 0.45
NEIGHBOR_RELEVANCE_THRESHOLD = 0.40
SHORT_SENTENCE_THRESHOLD = 5
SHORT_SENTENCE_PENALTY = 0.08
MAX_CONTEXT_EXPANSION = 5
HIGH_STOPWORD_RATIO = 0.6
STOPWORD_PENALTY_FACTOR = 0.5

# Domain Coherence
DOMAIN_COHERENCE_THRESHOLD = 0.38
DOMAIN_MIN_WORDS = 2
DOMAIN_OUTLIER_DELTA = 0.10

# Relation Validation
RELATION_SIM_THRESHOLD = 0.40
RELATION_COOCC_THRESHOLD = 3
RELATION_COOCC_THRESHOLD_NAMED = 1
RELATION_ENABLE_TSNE = False  # default to PCA due to speed

# Semantic Query Validation
SEMANTIC_SIMILARITY_THRESHOLD = 0.4
HIGH_SEMANTIC_SIMILARITY_THRESHOLD = 0.75
MIN_COOCCURRENCE_NORMAL = 1
MIN_COOCCURRENCE_STRICT = 3

# File Paths
BOOKS_CONFIG = [
    ('noli', 'csvFiles/noli_chapters.csv', 'csvFiles/noli_themes.csv'),
    ('elfili', 'csvFiles/elfili_chapters.csv', 'csvFiles/elfili_themes.csv')
]

# Models
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
SPACY_MODEL_NAME = "xx_sent_ud_sm"
