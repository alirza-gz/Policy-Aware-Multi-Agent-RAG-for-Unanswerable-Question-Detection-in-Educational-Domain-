"""Clarification metrics + dedicated Partially-Answerable evaluation.

Definitions (positive class for P/R/F1 = expected_action == CLARIFY):
  clarification_rate          CLARIFY / N
  clarification_precision     TP / predicted CLARIFY       (TP: predicted and expected CLARIFY)
  clarification_recall        TP / expected CLARIFY
  clarification_f1            harmonic mean
  correct_clarification_rate  TP / N   (population-level; NOT the same as recall)
  clarification_quality       mean rule-based score in [0,1] over predicted-CLARIFY rows:
                              specific (not generic) + names missing info + asks for input
                              (+ overlaps the gold out-of-scope sub-question when available).
                              A heuristic; it is NOT a human judgement.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from app.agents.answerability_agent import content_terms, is_generic_clarification

_ASK = re.compile(r"\b(rephrase|provide|specify|which|what exactly|point me|would you like|can you|could you)\b", re.I)


def _safe(a, b): return round(a / b, 4) if b else float("nan")


def clarification_quality_row(row: Dict) -> float:
    text = (row.get("clarification_question") or row.get("final_answer") or "")
    if row.get("action") != "CLARIFY" or not text.strip():
        return float("nan")
    score = [0.0 if is_generic_clarification(text) else 1.0,
             1.0 if (row.get("missing_information") or row.get("missing_parts")) else 0.0,
             1.0 if _ASK.search(text) else 0.0]
    gold_missing = row.get("out_of_scope_part_question")
    if gold_missing:
        g = set(content_terms(gold_missing))
        score.append(len(g & set(content_terms(text))) / len(g) if g else 0.0)
    return sum(score) / len(score)


def compute_clarification_metrics(rows: List[Dict]) -> Dict:
    n = len(rows)
    pred = [r for r in rows if r.get("action") == "CLARIFY"]
    exp = [r for r in rows if r.get("expected_action") == "CLARIFY"]
    tp = sum(1 for r in pred if r.get("expected_action") == "CLARIFY")
    p, rc = (tp / len(pred) if pred else 0.0), (tp / len(exp) if exp else 0.0)
    f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
    q = [v for v in (clarification_quality_row(r) for r in pred) if v == v]
    return {
        "n": n, "n_pred_clarify": len(pred), "n_expected_clarify": len(exp), "tp": tp,
        "clarification_rate": _safe(len(pred), n),
        "correct_clarification_rate": _safe(tp, n),
        "clarification_precision": round(p, 4), "clarification_recall": round(rc, 4),
        "clarification_f1": round(f1, 4),
        "clarification_quality": round(sum(q) / len(q), 4) if q else float("nan"),
        "clarification_quality_method": "rule_based_heuristic",
    }


def compute_partial_answerable_eval(rows: List[Dict]) -> Dict:
    pa = [r for r in rows if r.get("category") == "partially_answerable"]
    n = len(pa)
    acts = Counter(r.get("action") for r in pa)
    labelled = [r for r in pa if r.get("answerability_label")]
    q = [v for v in (clarification_quality_row(r) for r in pa) if v == v]
    return {
        "n": n, "action_counts": dict(acts),
        "partially_answerable_action_accuracy": _safe(acts.get("CLARIFY", 0), n),
        "partially_answerable_accuracy": _safe(
            sum(1 for r in labelled if r["answerability_label"] == "partially_answerable"), len(labelled))
        if labelled else None,
        "partial_label_method": "answerability_label logged" if labelled else "not_logged_in_these_predictions",
        "answer_rate_on_partial": _safe(acts.get("ANSWER", 0), n),   # answering a partially-answerable question
        "abstain_rate_on_partial": _safe(acts.get("ABSTAIN", 0), n),
        "clarification_quality_on_partial": round(sum(q) / len(q), 4) if q else float("nan"),
    }
