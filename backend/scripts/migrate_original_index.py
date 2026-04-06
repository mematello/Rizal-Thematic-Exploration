from app.models.database import engine
from sqlalchemy import text

def migrate():
    with engine.connect() as conn:
        print("Adding original_sentence_number column...")
        try:
            conn.execute(text("ALTER TABLE sentences ADD COLUMN original_sentence_number INTEGER;"))
            conn.commit()
            print("Successfully added column.")
        except Exception as e:
            print(f"Column might already exist: {e}")

if __name__ == "__main__":
    migrate()
