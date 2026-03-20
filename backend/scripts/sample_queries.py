import sys
import os
from sqlalchemy import select, or_, and_
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.database import SessionLocal, Sentence
from app.core.config import get_settings
db = SessionLocal()
def get_samples(filters, limit=5, source_type='full'):
    stmt = select(Sentence).filter(Sentence.source_type == source_type).filter(*filters).limit(limit * 3)
    results = db.execute(stmt).scalars().all()
    if results:
        random.shuffle(results)
        return results[:limit]
    return []

print('--- 1. Single character doing action ---')
s1 = get_samples([Sentence.sentence_text.ilike('%Ibarra%'), Sentence.sentence_text.like('%tumayo%')], 3, 'full')
s1 += get_samples([Sentence.sentence_text.ilike('%Simoun%'), Sentence.sentence_text.like('%kinuha%')], 2, 'full')
for s in s1: print(f'{s.id}: {s.sentence_text}')

print('\n--- 2. Multiple characters interacting ---')
s2 = get_samples([Sentence.sentence_text.ilike('%Ibarra%'), Sentence.sentence_text.ilike('%Elias%')], 3, 'full')
s2 += get_samples([Sentence.sentence_text.ilike('%Simoun%'), Sentence.sentence_text.ilike('%Basilio%')], 2, 'full')
for s in s2: print(f'{s.id}: {s.sentence_text}')

print('\n--- 3. No character names but thematic content ---')
s3 = get_samples([Sentence.sentence_text.ilike('%buhay%'), Sentence.sentence_text.ilike('%kamatayan%'), ~Sentence.sentence_text.ilike('%Ibarra%'), ~Sentence.sentence_text.ilike('%Maria%')], 3, 'full')
s3 += get_samples([Sentence.sentence_text.ilike('%bayan%'), Sentence.sentence_text.ilike('%laya%'), ~Sentence.sentence_text.ilike('%Simoun%')], 2, 'full')
for s in s3: print(f'{s.id}: {s.sentence_text}')

print('\n--- 4. Purely descriptive or atmospheric ---')
s4 = get_samples([or_(Sentence.sentence_text.ilike('%himpapawid%'), Sentence.sentence_text.ilike('%gabi%')), Sentence.sentence_text.ilike('%maliwanag%')], 5, 'full')
for s in s4: print(f'{s.id}: {s.sentence_text}')

print('\n--- 5. Transitional or filler (short) ---')
s5 = get_samples([Sentence.sentence_text.ilike('Ngunit%')], 5, 'full')
for s in s5: print(f'{s.id}: {s.sentence_text}')

print('\n--- 6. Buod heavily summarizing a scene ---')
s6 = get_samples([Sentence.sentence_text.ilike('%nag-usap%')], 5, 'summary')
for s in s6: print(f'{s.id}: {s.sentence_text}')

print('\n--- 7. Buod vague or abstract ---')
s7 = get_samples([Sentence.sentence_text.ilike('%napansin%')], 5, 'summary')
for s in s7: print(f'{s.id}: {s.sentence_text}')

db.close()
