from fastapi.testclient import TestClient
from app.main import app
from app.core.engine import get_engine

client = TestClient(app)

# Mock Engine
class MockEngine:
    def search(self, db, query, top_k=10):
        return {
            'noli': [
                {
                    'id': 1,
                    'chapter_number': 1,
                    'chapter_title': 'Mock Chapter',
                    'sentence_text': 'Mock text',
                    'scores': {'semantic': 90, 'lexical': 80, 'final': 85},
                    'themes': [{'id': '1', 'label': 'Test Theme', 'score': 0.9}]
                }
            ],
            'elfili': []
        }

def test_search_endpoint():
    # Override dependency
    app.dependency_overrides[get_engine] = lambda: MockEngine()
    
    response = client.get("/api/v1/search?q=test")
    assert response.status_code == 200
    data = response.json()
    assert 'results' in data
    assert len(data['results']['noli']) == 1
    assert data['results']['noli'][0]['chapter_title'] == 'Mock Chapter'
