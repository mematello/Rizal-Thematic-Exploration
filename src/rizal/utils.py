"""
Text processing validation utilities.
"""
import re

WORD_PATTERN = re.compile(r'[0-9a-zA-ZÀ-ÿñÑ]+(?:-[0-9a-zA-ZÀ-ÿñÑ]+)*')

def extract_words(text):
    """Extract words, preserving hyphenated tokens as single units."""
    if not text:
        return []
    return WORD_PATTERN.findall(str(text))

def sanitize_text(text):
    """Normalize whitespace and ensure strings are JSON-safe."""
    if text is None:
        return ""
    sanitized = " ".join(str(text).strip().split())
    return sanitized.replace('"', "'")

def shorten_sentence(text, max_words=12):
    """
    Truncate sentences while preserving readability for suggestion strings.
    Ensures output remains JSON-safe.
    """
    if not text:
        return ""

    words = str(text).strip().split()
    if not words:
        return ""

    if len(words) > max_words:
        snippet = " ".join(words[:max_words]) + " ..."
    else:
        snippet = " ".join(words)

    return sanitize_text(snippet)

def extract_relations_regex(text):
    """
    Fallback relation extraction using simple patterns for Tagalog possessives/markers.
    Detect patterns: <X> ng <Y>, <A> ni <B>
    Returns list of dicts: { 'pattern': 'ng'|'ni', 'left': str, 'right': str, 'span': str }
    """
    tokens = extract_words(text.lower())
    relations = []
    for i, tok in enumerate(tokens):
        if tok in {"ng", "ni"} and i - 1 >= 0 and i + 1 < len(tokens):
            left = tokens[i - 1]
            right = tokens[i + 1]
            span = f"{left} {tok} {right}"
            relations.append({'pattern': tok, 'left': left, 'right': right, 'span': span})
    return relations
