from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    # Check for duplicate sentences in Chapter 1
    result = conn.execute(text("""
        SELECT 
            sentence_index, 
            sentence_text,
            COUNT(*) as count
        FROM sentences
        WHERE book = 'noli' AND chapter_number = 1
        GROUP BY sentence_index, sentence_text
        HAVING COUNT(*) > 1
        ORDER BY sentence_index
    """))
    
    duplicates = result.fetchall()
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate sentence groups in Noli Chapter 1:")
        for dup in duplicates[:5]:  # Show first 5
            print(f"  Index {dup[0]}: appears {dup[2]} times - {dup[1][:60]}...")
    else:
        print("No duplicates found in database!")
        
    # Check total count
    result2 = conn.execute(text("""
        SELECT COUNT(*) FROM sentences
        WHERE book = 'noli' AND chapter_number = 1
    """))
    total = result2.fetchone()[0]
    print(f"\nTotal rows in DB for Noli Ch.1: {total}")
