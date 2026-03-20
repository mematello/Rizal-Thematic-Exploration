import sys
import os
import random
from sqlalchemy import select, or_

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.models.database import SessionLocal, Sentence
from app.api.v1.content import _get_paksa_data, _get_sanggunian_data

def evaluate():
    db = SessionLocal()
    
    def get_samples(filters, limit=3, source_type='full'):
        stmt = select(Sentence).filter(Sentence.source_type == source_type).filter(*filters).limit(limit * 10)
        results = db.execute(stmt).scalars().all()
        if results:
            random.shuffle(results)
            return results[:limit]
        return []

    categories = [
        ("Single character doing action", get_samples([Sentence.sentence_text.ilike('%Ibarra%'), Sentence.sentence_text.ilike('%tumingin%')], 3, 'full') or get_samples([Sentence.sentence_text.ilike('%Ibarra%')], 3, 'full')),
        ("Multiple characters interacting", get_samples([Sentence.sentence_text.ilike('%Ibarra%'), Sentence.sentence_text.ilike('%Elias%')], 3, 'full') or get_samples([Sentence.sentence_text.ilike('%Simoun%'), Sentence.sentence_text.ilike('%Basilio%')], 3, 'full')),
        ("No character names but thematic content", get_samples([Sentence.sentence_text.ilike('%buhay%'), Sentence.sentence_text.ilike('%kamatayan%')], 3, 'full') or get_samples([Sentence.sentence_text.ilike('%bayan%'), Sentence.sentence_text.ilike('%laya%')], 3, 'full')),
        ("Purely descriptive or atmospheric", get_samples([or_(Sentence.sentence_text.ilike('%himpapawid%'), Sentence.sentence_text.ilike('%gabi%')), Sentence.sentence_text.ilike('%maliwanag%')], 3, 'full') or get_samples([Sentence.sentence_text.ilike('%araw%'), Sentence.sentence_text.ilike('%langit%')], 3, 'full')),
        ("Transitional or filler", get_samples([Sentence.sentence_text.ilike('Ngunit%')], 3, 'full') or get_samples([Sentence.sentence_text.ilike('Subalit%')], 3, 'full')),
        ("Buod heavily summarizing a scene", get_samples([Sentence.sentence_text.ilike('%nag-usap%')], 3, 'summary') or get_samples([Sentence.sentence_text.ilike('%nagpasya%')], 3, 'summary')),
        ("Buod vague or abstract", get_samples([Sentence.sentence_text.ilike('%napansin%')], 2, 'summary') or get_samples([Sentence.sentence_text.ilike('%naramdaman%')], 2, 'summary'))
    ]

    report = "# Manual Evaluation Report\n\n"
    
    for cat_name, sentences in categories:
        report += f"## {cat_name}\n\n"
        for s in sentences:
            report += f"**Sentence [{s.id} ({s.book} - {s.source_type})]**: {s.sentence_text}\n\n"
            
            # Paksa
            try:
                paksa = _get_paksa_data(s.id, db)
                if paksa.has_theme and paksa.themes:
                    t = paksa.themes[0]
                    report += f"- **Theme Assigned**: {t.label} (Confidence: {t.confidence:.2f})\n"
                    report += f"- **Theme Evidence**: {t.evidence}\n"
                else:
                    report += f"- **Theme Assigned**: Walang Tema\n"
            except Exception as e:
                report += f"- **Theme Error**: {e}\n"
                
            # Sanggunian
            try:
                sang = _get_sanggunian_data(s.id, db)
                if sang.has_reference:
                    report += f"- **Sanggunian**: {sang.reference_text}\n"
                else:
                    report += f"- **Sanggunian**: Walang Sanggunian\n"
            except Exception as e:
                report += f"- **Sanggunian Error**: {e}\n"
                
            report += f"- **Judgment**: [PENDING MANUAL JUDGMENT]\n\n"
            
    with open(os.path.join(os.path.dirname(__file__), 'raw_report.md'), 'w', encoding='utf-8') as f:
        f.write(report)
        
    db.close()
    print("Report written to scripts/raw_report.md")

if __name__ == '__main__':
    evaluate()
