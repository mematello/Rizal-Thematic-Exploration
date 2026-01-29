
import pytest
import numpy as np
from unittest.mock import MagicMock
from rizal.engine import RizalEngine
from rizal.loader import DataLoader
from rizal.errors import RizalError

@pytest.fixture
def mock_loader():
    loader = MagicMock(spec=DataLoader)
    loader.books_data = {'noli': {'chapters': [], 'embeddings': []}}
    loader.model = MagicMock()
    # Mock encode to return a numpy array of shape (1, 384)
    loader.model.encode.return_value = np.zeros((1, 384))
    
    loader.ready = True
    loader.global_vocabulary = {'ibarra', 'noli', 'el', 'fili'}
    loader.corpus_vocabulary = {'noli': {'ibarra'}}
    
    # Use real numpy arrays for matrices to satisfy sklearn checks
    loader.passage_embeddings_matrix = np.zeros((2, 384)) # 2 passages
    loader.theme_embeddings_matrix = np.zeros((0, 384))
    
    loader.global_passages = [
        {'sentence_text': 'Passage 1 text'},
        {'sentence_text': 'Passage 2 text'}
    ]
    
    # Mock methods
    loader.get_passage_id.return_value = (1, 1)
    return loader

def test_engine_initialization(mock_loader):
    engine = RizalEngine(mock_loader)
    assert engine.data == mock_loader

def test_query_stopwords_only(mock_loader):
    engine = RizalEngine(mock_loader)
    # Mock query analyzer to return no content words
    engine.query_analyzer.get_content_words = MagicMock(return_value=[])
    
    result = engine.query("ang sa mga")
    assert result['status'] == 'error'
    assert result['error_type'] == 'no_lexical_grounding'

def test_query_unknown_words(mock_loader):
    engine = RizalEngine(mock_loader)
    engine.query_analyzer.validate_filipino_query = MagicMock(return_value=(True, {}))
    engine.query_analyzer.get_content_words = MagicMock(return_value=['unknownword'])
    
    result = engine.query("unknownword")
    assert result['status'] == 'error'
    assert result['error_type'] == 'no_lexical_grounding'
    assert 'missing_words' in result['overlap_info']
