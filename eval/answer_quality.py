"""Answer-quality metrics for gold-answerable questions.

Separates decision correctness from answer correctness:
- Decision: did the system ANSWER?
- Quality (only when it answered): EM, token F1, semantic similarity, groundedness.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def exact_match(prediction: str, golds: Sequence[str]) -> float:
    pred = (prediction or "").strip().lower()
    if not pred:
        return 0.0
    return 1.0 if any(pred == (g or "").strip().lower() for g in golds if g) else 0.0


def token_f1(prediction: str, golds: Sequence[str]) -> float:
    pred_toks = tokenize(prediction)
    if not pred_toks:
        return 0.0
    best = 0.0
    for g in golds:
        gold_toks = tokenize(g or "")
        if not gold_toks:
            continue
        common = set(pred_toks) & set(gold_toks)
        if not common:
            continue
        # Multiset-aware overlap via counts.
        from collections import Counter

        pc, gc = Counter(pred_toks), Counter(gold_toks)
        overlap = sum(min(pc[t], gc[t]) for t in pc)
        if overlap == 0:
            continue
        precision = overlap / len(pred_toks)
        recall = overlap / len(gold_toks)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return float(best)


def semantic_similarity(prediction: str, golds: Sequence[str], model=None) -> float:
    """Cosine similarity via the shared embedding model (lazy)."""
    if not prediction or not golds:
        return 0.0
    if model is None:
        return float("nan")  # caller may skip
    import numpy as np

    texts = [prediction] + [g for g in golds if g]
    emb = model.encode(texts, convert_to_numpy=True)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = emb[0] @ emb[1:].T
    return float(np.max(sims)) if len(sims) else 0.0


def groundedness(prediction: str, passage_texts: Sequence[str]) -> float:
    """Token overlap of answer against retrieved passages (lexical faithfulness).

    Not an NLI entailment score; documented as a lexical groundedness proxy.
    """
    pred = set(tokenize(prediction))
    if not pred:
        return 0.0
    ctx = set()
    for t in passage_texts:
        ctx.update(tokenize(t or ""))
    if not ctx:
        return 0.0
    return len(pred & ctx) / len(pred)


def compute_answer_quality(rows: List[Dict], embed_model=None) -> Dict:
    """Metrics over gold-answerable questions only."""
    answerable = [r for r in rows if r.get("gold_answerable", False)]
    n = len(answerable)
    if n == 0:
        return {}

    answered = [r for r in answerable if r.get("action") == "ANSWER"]
    n_ans = len(answered)

    em_vals, f1_vals, sem_vals, ground_vals = [], [], [], []
    for r in answered:
        pred = str(r.get("final_answer", "") or "")
        golds = r.get("gold_answers") or []
        em_vals.append(exact_match(pred, golds))
        f1_vals.append(token_f1(pred, golds))
        if embed_model is not None:
            sem_vals.append(semantic_similarity(pred, golds, model=embed_model))
        # Groundedness uses retrieved passage texts if present; else skip.
        passages = r.get("passage_texts") or []
        if passages:
            ground_vals.append(groundedness(pred, passages))

    def _avg(xs):
        return round(float(sum(xs) / len(xs)), 4) if xs else float("nan")

    return {
        "n_answerable": n,
        "n_answered": n_ans,
        "answer_rate_on_answerable": round(n_ans / n, 4) if n else 0.0,
        "exact_match": _avg(em_vals),
        "token_f1": _avg(f1_vals),
        "semantic_similarity": _avg(sem_vals) if sem_vals else float("nan"),
        "groundedness": _avg(ground_vals) if ground_vals else float("nan"),
        # Denominator = all gold-answerable (unanswered count as 0 quality).
        "exact_match_all_answerable": round(sum(em_vals) / n, 4) if n else 0.0,
        "token_f1_all_answerable": round(sum(f1_vals) / n, 4) if n else 0.0,
    }
