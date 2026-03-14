import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.core.engine import RizalEngine
from app.models.database import SessionLocal, Sentence
from sqlalchemy import select

def get_passage_context(db, center, window=8):
    range_start, range_end = center.sentence_index - window, center.sentence_index + window
    candidates = db.scalars(select(Sentence).filter(
        Sentence.book == center.book, 
        Sentence.chapter_number == center.chapter_number, 
        Sentence.source_type == center.source_type,
        Sentence.sentence_index >= range_start, 
        Sentence.sentence_index <= range_end
    ).order_by(Sentence.sentence_index)).all()
    
    text = " ".join([s.sentence_text for s in candidates])
    return text

def format_themes(themes_raw):
    if not themes_raw: return "None (0.00)"
    if isinstance(themes_raw[0], dict):
        scored = [f"{t.get('label')} ({abs(round(float(t.get('score', 0)), 2))})" for t in themes_raw]
        return ", ".join(scored)
    return ", ".join(themes_raw)

def main():
    queries = [
        "edukasyon",
        "pang-aapi ng kastila",
        "simbahan",
        "kalayaan",
        "elias",
        "maria clara"
    ]
    
    engine = RizalEngine()
    output_lines = ["# Context Window Classification Experiment\n"]
    
    with SessionLocal() as db:
        for q in queries:
            output_lines.append(f"## Query: `{q}`\n")
            try:
                res = engine.search(db=db, query=q, source_type="full")
                results = res.get("results", {})
                combined = results.get("noli", []) + results.get("elfili", [])
                
                if not combined:
                    output_lines.append("*No results found.*\n")
                    continue
                    
                # Take top 2 results to keep report concise
                for i, item in enumerate(combined[:2]):
                    sent_id = item.get("id")
                    if not sent_id: continue
                    
                    center_sent = db.scalars(select(Sentence).filter(Sentence.id == sent_id)).first()
                    if not center_sent: continue
                    
                    # 1. Exact Sentence
                    exact_text = center_sent.sentence_text
                    exact_emb = engine.base_model.encode(exact_text, show_progress_bar=False)
                    themes_exact = engine._classify_themes(db, exact_emb, exact_text)
                    
                    # 2. Sentence + Neighbors (using engine's native _expand_context algorithm)
                    neighbor_text = engine._expand_context(db, center_sent)
                    # Strip HTML tags like <strong> that engine inserts
                    clean_neighbor_text = neighbor_text.replace("<strong>", "").replace("</strong>", "")
                    neighbor_emb = engine.base_model.encode(clean_neighbor_text, show_progress_bar=False)
                    themes_neighbor = engine._classify_themes(db, neighbor_emb, clean_neighbor_text)
                    
                    # 3. Full Passage Block (+/- 8 sentences)
                    passage_text = get_passage_context(db, center_sent, window=8)
                    passage_emb = engine.base_model.encode(passage_text, show_progress_bar=False)
                    themes_passage = engine._classify_themes(db, passage_emb, passage_text)
                    
                    output_lines.append(f"**Result {i+1}:**")
                    output_lines.append(f"> {exact_text}")
                    output_lines.append(f"- **Sentence Only:** {format_themes(themes_exact)}")
                    output_lines.append(f"- **Sentence + Neighbors:** {format_themes(themes_neighbor)}")
                    output_lines.append(f"- **Passage Block:** {format_themes(themes_passage)}")
                    output_lines.append("\n*Judgment*: [Fill in post-review]")
                    output_lines.append("\n---\n")
                    
            except Exception as e:
                import traceback
                output_lines.append(f"*Error during search: {e}*\n ```\n{traceback.format_exc()}\n```\n")
                
    artifact_path = '/Users/marcusoliver/.gemini/antigravity/brain/b1ab1c8a-66f1-4193-942b-b09ee6af16fe/context_window_experiment.md'
    with open(artifact_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
        
    print(f"Done. Wrote to {artifact_path}")

if __name__ == "__main__":
    main()
