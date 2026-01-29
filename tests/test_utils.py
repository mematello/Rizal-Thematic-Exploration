
import pytest
from rizal.utils import extract_words, sanitize_text, shorten_sentence, extract_relations_regex

def test_extract_words():
    text = "Hello-World this is a test!"
    words = extract_words(text)
    assert "Hello-World" in words
    assert "test" in words
    assert "!" not in words

def test_sanitize_text():
    raw = "  Diff   spaces  \n "
    assert sanitize_text(raw) == "Diff spaces"

def test_shorten_sentence():
    long_text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    short = shorten_sentence(long_text, max_words=5)
    assert short == "one two three four five ..."
    assert len(extract_words(short)) == 5 # 5 words (ellipses ignored by regex)

def test_extract_relations_regex():
    text = "Ang sumbrero ni Rizal ay bago."
    rels = extract_relations_regex(text)
    assert len(rels) >= 1
    assert rels[0]['pattern'] == 'ni'
    assert rels[0]['left'] == 'sumbrero'
    assert rels[0]['right'] == 'rizal'
