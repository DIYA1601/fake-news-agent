# FAISS + sentence-transformers wrapper for local in-memory retrieval
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, corpus_texts=None, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus_texts or []
        self.embeddings = None
        self.index = None
        if self.corpus:
            self._build_index(self.corpus)

    def _build_index(self, texts):
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        # normalize for cosine-ish similarity
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.embeddings = embs
        self.corpus = texts

    def add_corpus(self, texts):
        if not texts:
            return
        self.corpus = texts
        self._build_index(texts)

    def get_top_docs(self, query, k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        if self.index is None or self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.corpus):
                continue
            results.append({"text": self.corpus[idx], "score": float(score)})
        return results
