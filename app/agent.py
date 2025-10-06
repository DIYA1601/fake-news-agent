from .claim_extractor import extract_claims
from .retriever import Retriever
from .verifier import Verifier
from .utils import search_wikipedia

class FactAgent:
    def __init__(self, seed_corpus=None):
        # seed corpus can be a list of strings
        self.retriever = Retriever(corpus_texts=seed_corpus or [])
        self.verifier = Verifier()

    def verify_claim(self, text, url=None):
        claims = extract_claims(text)
        results = []
        for claim in claims:
            # 1) quick wiki search for candidate evidence (fast bootstrap)
            wiki_docs = search_wikipedia(claim, max_results=3)
            # 2) retrieve from local index
            retrieved = self.retriever.get_top_docs(claim, k=5)
            # prefer retrieved; fallback to wiki
            evidence = retrieved if retrieved else [{"text":d} for d in wiki_docs]
            verdict = self.verifier.verify(claim, evidence)
            results.append({
                "claim": claim,
                "verdict": verdict["label"],
                "scores": verdict.get("scores"),
                "explanation": verdict.get("explanation"),
                "evidence": evidence[:3]
            })
        return {"results": results}
