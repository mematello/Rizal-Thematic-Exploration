"""
EXPLAIN ANALYZE Query Plan Analysis

Captures query plans for common search patterns to identify
index usage, sequential scans, and potential optimizations.

Outputs to: docs/perf/query_plans.txt
"""

import psycopg2
import json
from pathlib import Path

# Database connection from docker-compose.yml
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 5433,
    "database": "rizal_db",
    "user": "rizal",
    "password": "rizalpass123"
}

OUTPUT_PATH = Path(__file__).parent.parent / "docs" / "perf" / "query_plans.txt"


def get_sample_embedding():
    """Get a real embedding vector from the database for realistic query plans"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get embedding as text representation
    cur.execute("SELECT embedding::text FROM sentences WHERE embedding IS NOT NULL LIMIT 1")
    result = cur.fetchone()

    cur.close()
    conn.close()

    if result:
        # PostgreSQL returns the vector as a string like '[0.1,0.2,...]'
        # This is the correct format for the query
        return result[0]
    return None


def run_explain_query(conn, query_name, sql):
    """Execute EXPLAIN (ANALYZE, BUFFERS) and return formatted output"""
    cur = conn.cursor()

    try:
        cur.execute(sql)
        plan = cur.fetchall()

        output = [f"\n{'=' * 80}"]
        output.append(f"Query: {query_name}")
        output.append('=' * 80)
        output.append(sql)
        output.append('-' * 80)
        output.append("EXPLAIN Output:")
        output.append('-' * 80)

        for row in plan:
            output.append(row[0])

        return "\n".join(output)

    except Exception as e:
        return f"\nError executing {query_name}: {e}\n"
    finally:
        cur.close()


def get_table_stats(conn):
    """Get row counts and index information for sentences and themes tables"""
    cur = conn.cursor()

    output = ["\n" + "=" * 80]
    output.append("DATABASE STATISTICS")
    output.append("=" * 80)

    # Table row counts
    queries = {
        "Total sentences": "SELECT COUNT(*) FROM sentences",
        "Summary sentences": "SELECT COUNT(*) FROM sentences WHERE source_type = 'summary'",
        "Full text sentences": "SELECT COUNT(*) FROM sentences WHERE source_type = 'full'",
        "Noli sentences": "SELECT COUNT(*) FROM sentences WHERE book LIKE '%Noli%'",
        "Fili sentences": "SELECT COUNT(*) FROM sentences WHERE book LIKE '%Fili%'",
        "Total themes": "SELECT COUNT(*) FROM themes",
        "Sentences with embeddings": "SELECT COUNT(*) FROM sentences WHERE embedding IS NOT NULL",
        "Sentences with DAPT embeddings": "SELECT COUNT(*) FROM sentences WHERE embedding_dapt IS NOT NULL"
    }

    for desc, sql in queries.items():
        cur.execute(sql)
        count = cur.fetchone()[0]
        output.append(f"{desc:<35} {count:>10,}")

    # Index information
    output.append("\n" + "-" * 80)
    output.append("INDEXES")
    output.append("-" * 80)

    cur.execute("""
        SELECT
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes
        WHERE schemaname = 'public'
        AND tablename IN ('sentences', 'themes')
        ORDER BY tablename, indexname
    """)

    for schema, table, index, definition in cur.fetchall():
        output.append(f"\n{table}.{index}:")
        output.append(f"  {definition}")

    cur.close()
    return "\n".join(output)


def main():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)

    # Get sample embedding for realistic vector queries
    sample_embedding = get_sample_embedding()
    if not sample_embedding:
        print("Warning: Could not retrieve sample embedding")
        sample_embedding = "[" + ",".join(["0.1"] * 768) + "]"

    print(f"Using embedding vector (first 50 chars): {sample_embedding[:50]}...")

    print(f"Running EXPLAIN ANALYZE queries...")

    results = []

    # Get database statistics first
    results.append(get_table_stats(conn))

    # Test queries based on actual search patterns from engine.py

    # 1. Semantic search with vector similarity (most common in fallback)
    results.append(run_explain_query(
        conn,
        "Semantic Search - Vector Similarity",
        f"""
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, sentence_index, chapter_number, chapter_title, embedding
        FROM sentences
        WHERE book = 'noli'
          AND source_type = 'summary'
        ORDER BY embedding <=> {sample_embedding!r}::vector
        LIMIT 150;
        """
    ))

    # 2. Lexical ILIKE search (single word)
    results.append(run_explain_query(
        conn,
        "Lexical Search - Single ILIKE",
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, sentence_index, chapter_number, embedding
        FROM sentences
        WHERE book = 'noli'
          AND source_type = 'summary'
          AND sentence_text ILIKE '%kalayaan%'
        LIMIT 200;
        """
    ))

    # 3. Multi-filter lexical search (multi-word query)
    results.append(run_explain_query(
        conn,
        "Lexical Search - Multi-word (multiple ILIKE)",
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, sentence_index, chapter_number, embedding
        FROM sentences
        WHERE book = 'noli'
          AND source_type = 'summary'
          AND sentence_text ILIKE '%pag-ibig%'
          AND sentence_text ILIKE '%Maria Clara%'
        LIMIT 200;
        """
    ))

    # 4. Context expansion - range query on sentence_index
    results.append(run_explain_query(
        conn,
        "Context Expansion - Neighbor Sentences",
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, sentence_index, embedding
        FROM sentences
        WHERE book = 'noli'
          AND chapter_number = 10
          AND source_type = 'summary'
          AND sentence_index BETWEEN 45 AND 61
        ORDER BY sentence_index;
        """
    ))

    # 5. Theme classification - theme matrix lookup
    results.append(run_explain_query(
        conn,
        "Theme Lookup - All Themes",
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, tagalog_title, meaning, embedding
        FROM themes;
        """
    ))

    # 6. Cross-lingual with OR filters (bridge tokens)
    results.append(run_explain_query(
        conn,
        "Cross-lingual Search - OR filters",
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, embedding
        FROM sentences
        WHERE book = 'noli'
          AND source_type = 'summary'
          AND (sentence_text ILIKE '%korapsyon%'
               OR sentence_text ILIKE '%katiwalian%'
               OR sentence_text ILIKE '%kasakiman%')
        LIMIT 200;
        """
    ))

    # 7. Semantic search with multiple filters (HNSW + WHERE concern)
    results.append(run_explain_query(
        conn,
        "Semantic + Filters - HNSW Recall Test",
        f"""
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, chapter_number
        FROM sentences
        WHERE book = 'El Filibusterismo'
          AND source_type = 'summary'
          AND chapter_number BETWEEN 10 AND 20
        ORDER BY embedding <=> {sample_embedding!r}::vector
        LIMIT 100;
        """
    ))

    # 8. Character chapters query pattern
    results.append(run_explain_query(
        conn,
        "Character Search - Text Match",
        """
        EXPLAIN (ANALYZE, BUFFERS)
        SELECT id, sentence_text, chapter_number, sentence_index
        FROM sentences
        WHERE source_type = 'summary'
          AND (sentence_text ILIKE '%Crisostomo%'
               OR sentence_text ILIKE '%Ibarra%')
        LIMIT 500;
        """
    ))

    conn.close()

    # Write results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(results))

    print(f"\n✓ Query plans saved to: {OUTPUT_PATH}")
    print("\nKey findings to look for:")
    print("  - Sequential scans vs index scans")
    print("  - ILIKE queries cannot use B-tree indexes (consider pg_trgm GIN)")
    print("  - Vector search: check if HNSW index is used")
    print("  - HNSW + WHERE filters: check filter order (recall risk)")
    print("  - Buffer usage and shared hit ratios")


if __name__ == "__main__":
    main()
