import spacy
try:
    nlp = spacy.load('xx_ent_wiki_sm')
    print("SUCCESS: Model loaded")
except Exception as e:
    print("ERROR:", e)
