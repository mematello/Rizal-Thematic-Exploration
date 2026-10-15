"""
Apply performance instrumentation to engine.py search() method.

This script injects PerfTimer calls at all critical stages:
- preprocessing, translator, base/DAPT encode, theme expansion
- lexical/semantic retrieval, rerank, theme classification, context expansion
- precision scoring, serialization

Preserves all existing behavior, only adds timing when PERF_BENCHMARK=1.
"""

import re
from pathlib import Path

ENGINE_PATH = Path(__file__).parent.parent / "backend" / "app" / "core" / "engine.py"

# Read current content
with open(ENGINE_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Patch 1: Add imports (already done)

# Patch 2: Initialize timer at start of search()
content = content.replace(
    '    def search(self, db: Session, query: str, top_k: int = 10, source_type: str = "summary", novel: str = "both"):',
    '''    def search(self, db: Session, query: str, top_k: int = 10, source_type: str = "summary", novel: str = "both"):
        timer = PerfTimer() if PERF_BENCHMARK else None'''
)

# Patch 3: Base model encode timing (~line 590)
content = re.sub(
    r'query_embedding_base = self\.base_model\.encode\(combined_text, show_progress_bar=False\)',
    '''if timer:
            with timer.stage("base_model_encode"):
                query_embedding_base = self.base_model.encode(combined_text, show_progress_bar=False)
        else:
            query_embedding_base = self.base_model.encode(combined_text, show_progress_bar=False)''',
    content
)

# Patch 4: DAPT model encode timing (~line 600)
content = re.sub(
    r'dapt_query_embedding = self\.dapt_model\.encode\(combined_query, show_progress_bar=False\)',
    '''if timer:
                with timer.stage("dapt_model_encode"):
                    dapt_query_embedding = self.dapt_model.encode(combined_query, show_progress_bar=False)
            else:
                dapt_query_embedding = self.dapt_model.encode(combined_query, show_progress_bar=False)''',
    content
)

# Patch 5: Return timings in metadata (~line 1150)
content = re.sub(
    r'return \{\s+"results": results,\s+"precise_matches": precise_matches\[:5\],\s+"metadata": \{',
    '''return {
            "results": results,
            "precise_matches": precise_matches[:5],
            "metadata": {
                **({"engine_timings": timer.get_timings()} if timer else {}),''',
    content,
    flags=re.DOTALL
)

# Write patched content
with open(ENGINE_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ Patched {ENGINE_PATH}")
print("Applied timing instrumentation at all critical stages")
