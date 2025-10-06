# Simple verifier using a zero-shot NLI model (MNLI) via Hugging Face pipeline.
from transformers import pipeline

class Verifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        # uses NLI labels: ENTAILMENT ~ Supported, CONTRADICTION ~ Refuted, NEUTRAL ~ NotEnoughInfo
        self.pipe = pipeline("text-classification", model=model_name, return_all_scores=True, truncation=True)

    def map_nli_label(self, hf_label):
        # model label mapping to our schema
        mapping = {
            "ENTAILMENT": "SUPPORTED",
            "CONTRADICTION": "REFUTED",
            "NEUTRAL": "NOT_ENOUGH_INFO"
        }
        return mapping.get(hf_label.upper(), hf_label)

    def verify(self, claim, evidence_docs):
        context = "\n\n".join([d["text"] for d in evidence_docs]) if evidence_docs else ""
        premise = f"Evidence: {context}" if context else "Evidence:"
        hypothesis = claim
        input_text = {"premise": premise, "hypothesis": hypothesis}
        # Many HF pipelines accept text pair; for simplicity, use concatenated prompt:
        combined = premise + "\n\nHypothesis: " + hypothesis
        outputs = self.pipe(combined[:1000])  # truncate to avoid huge input
        # outputs is list of dicts with label & score; convert best
        # outputs example: [{'label':'ENTAILMENT', 'score':0.7}, ...] but because return_all_scores True it's list of lists
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
            scores = {o['label']: o['score'] for o in outputs[0]}
        else:
            # fallback
            scores = {o['label']: o['score'] for o in outputs}
        best_label = max(scores.items(), key=lambda x: x[1])[0]
        return {
            "label": self.map_nli_label(best_label),
            "scores": scores,
            "explanation": f"Used top {len(evidence_docs)} evidence docs. See evidence snippets."
        }
