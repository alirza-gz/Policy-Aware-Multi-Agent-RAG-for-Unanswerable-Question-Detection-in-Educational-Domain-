"""Separated hallucination metrics + explicitly-named groundedness.

Definitions (all computed from one predictions file; gold labels used ONLY for scoring):

  unanswerable_answer_rate  P(action=ANSWER | gold_answerable=False)
                            == false_acceptance_rate (same quantity; both names kept).
  wrong_answer_rate         P(answer lacks gold span | answered & gold_answerable=True).
                            Proxy: token-F1 / containment against gold answers.
  unsupported_answer_rate   P(support < tau | answered). support = lexical_groundedness by
                            default, or NLI entailment probability when an NLI scorer is
                            supplied (then ``support_method`` = "nli").
  contradiction_rate        P(NLI contradiction >= tau | answered). Requires NLI;
                            otherwise None and ``contradiction_method`` = "not_computed".
  hallucination_rate_any    P(answered row is unanswerable-answer OR wrong OR unsupported
                            OR contradicted | answered).
  hallucination_rate_legacy == unanswerable_answer_rate (the pre-refactor metric whose
                            name over-claimed; kept for regression comparison only).

``lexical_groundedness`` is token overlap of the answer with retrieved passages. It is
NOT entailment and is inflated by extractive answers (a system that copies a passage
scores ~1.0 regardless of whether the passage answers the question).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from eval.answer_quality import token_f1, tokenize

ANSWER = "ANSWER"


def lexical_groundedness(answer: str, passage_texts: Sequence[str]) -> float:
    pred = set(tokenize(answer))
    if not pred:
        return 0.0
    ctx = set()
    for t in passage_texts:
        ctx.update(tokenize(t or ""))
    return len(pred & ctx) / len(pred) if ctx else 0.0


def semantic_groundedness(answer: str, passage_texts: Sequence[str],
                          entail_prob: Callable[[str, str], float]) -> float:
    """Max NLI entailment probability of the answer given any passage.
    Independent of gold answers (uses answer + retrieved evidence only)."""
    if not (answer or "").strip() or not passage_texts:
        return 0.0
    return max(float(entail_prob(p, answer)) for p in passage_texts)


class NLIScorer:
    """Lazy cross-encoder NLI. Not loadable offline; tests inject fakes instead."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small"):
        from sentence_transformers import CrossEncoder
        self.m = CrossEncoder(model_name)
        self.model_name = model_name

    def _probs(self, premise: str, hypothesis: str):
        import numpy as np
        logits = self.m.predict([(premise, hypothesis)])[0]
        e = np.exp(logits - np.max(logits))
        return e / e.sum()  # label order for this model: contradiction, entailment, neutral

    def entail_prob(self, premise, hypothesis): return float(self._probs(premise, hypothesis)[1])
    def contradict_prob(self, premise, hypothesis): return float(self._probs(premise, hypothesis)[0])


def _is_wrong(row: Dict, f1_floor: float = 0.3) -> bool:
    ans = (row.get("final_answer") or "").lower()
    golds = [g for g in (row.get("gold_answers") or []) if g]
    if not golds:
        return False
    if any(g.lower() in ans for g in golds):
        return False
    return token_f1(ans, golds) < f1_floor


def compute_hallucination_metrics(rows: List[Dict], support_tau: float = 0.6,
                                  entail_prob: Optional[Callable] = None,
                                  contradict_prob: Optional[Callable] = None,
                                  nli_tau: float = 0.5) -> Dict:
    n = len(rows)
    answered = [r for r in rows if r.get("action") == ANSWER and (r.get("final_answer") or "").strip()]
    unans = [r for r in rows if not r.get("gold_answerable", True)]
    ans_on_unans = [r for r in unans if r.get("action") == ANSWER]
    ans_on_ans = [r for r in answered if r.get("gold_answerable", True)]

    def support(r):
        ps = r.get("passage_texts") or []
        if entail_prob is not None:
            return semantic_groundedness(r.get("final_answer", ""), ps, entail_prob)
        return lexical_groundedness(r.get("final_answer", ""), ps)

    flags = []
    for r in answered:
        un = not r.get("gold_answerable", True)
        wrong = r.get("gold_answerable", True) and _is_wrong(r)
        uns = support(r) < (nli_tau if entail_prob is not None else support_tau)
        con = False
        if contradict_prob is not None:
            con = max((contradict_prob(p, r["final_answer"]) for p in (r.get("passage_texts") or [])),
                      default=0.0) >= nli_tau
        flags.append((un, wrong, uns, con))

    def rate(k, den): return round(sum(1 for f in flags if f[k]) / den, 4) if den else float("nan")
    na = len(answered)
    out = {
        "n": n, "n_answered": na, "n_gold_unanswerable": len(unans),
        "unanswerable_answer_rate": round(len(ans_on_unans) / len(unans), 4) if unans else float("nan"),
        "wrong_answer_rate": round(sum(1 for r in ans_on_ans if _is_wrong(r)) / len(ans_on_ans), 4) if ans_on_ans else float("nan"),
        "unsupported_answer_rate": rate(2, na),
        "support_method": "nli" if entail_prob is not None else "lexical_groundedness",
        "contradiction_rate": rate(3, na) if contradict_prob is not None else None,
        "contradiction_method": "nli" if contradict_prob is not None else "not_computed (requires NLI model)",
        "hallucination_rate_any": round(sum(1 for f in flags if any(f)) / na, 4) if na else float("nan"),
    }
    out["false_acceptance_rate"] = out["unanswerable_answer_rate"]
    out["hallucination_rate_legacy"] = out["unanswerable_answer_rate"]
    return out
