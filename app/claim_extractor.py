# simple heuristic claim extractor: split into sentences and return ones with verbs/entities
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def extract_claims(text: str, max_claims: int = 5):
    sents = sent_tokenize(text)
    # simple heuristic: return up to max_claims longest sentences (likely containing claim)
    sents = sorted(sents, key=lambda s: -len(s))
    return sents[:max_claims]
