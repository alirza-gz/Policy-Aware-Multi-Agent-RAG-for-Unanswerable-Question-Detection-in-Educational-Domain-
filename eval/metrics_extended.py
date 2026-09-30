"""Extended thesis metrics for Policy-Aware Multi-Agent RAG evaluation.

Includes classification (with ROC/PR-AUC from continuous scores), abstention,
hallucination (global + per-category), policy evaluation, and helpers used by
``eval.report``.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from app.config import Config

ACTIONS = ["ANSWER", "CLARIFY", "ABSTAIN"]

_GOV = dict(getattr(Config, "GOVERNANCE", {}) or {})
_RETR_ABSTAIN = float(_GOV.get("retriever_abstain_below", 0.2))
_REAS_ABSTAIN = float(_GOV.get("reasoner_abstain_below", 0.3))
_CLARIFY_BAND = list(_GOV.get("clarify_band", [0.3, 0.5]) or [0.3, 0.5])
_BANNED = [str(p).lower() for p in (getattr(Config, "BANNED_PHRASES", []) or [])]


def policy_expected_action(row: Dict) -> str:
    """Re-apply configured policy mirroring GovernanceAgent.decide() precedence.

    Precedence (must match live agent):
      safety -> model CLARIFY -> answerability ABSTAIN gates -> mid-band CLARIFY -> ANSWER
    """
    answer = (row.get("raw_answer", "") or "").lower()
    if any(p and p in answer for p in _BANNED):
        return "ABSTAIN"
    # Prefer answerability_confidence when present (same as GovernanceAgent).
    conf = float(
        row.get(
            "answerability_confidence",
            row.get("reasoner_confidence", 0.0),
        )
    )
    retr = float(row.get("retriever_confidence", 0.0))
    # Model-requested clarification BEFORE answerability abstain.
    if row.get("model_needs_clarification", False):
        return "CLARIFY"
    if not row.get("model_is_answerable", False):
        return "ABSTAIN"
    if retr < _RETR_ABSTAIN or conf < _REAS_ABSTAIN:
        return "ABSTAIN"
    if _CLARIFY_BAND[0] <= conf < _CLARIFY_BAND[1]:
        return "CLARIFY"
    return "ANSWER"


def _answered_correctly(row: Dict) -> bool:
    if row.get("action") != "ANSWER":
        return False
    answer = (row.get("final_answer", "") or "").lower()
    return any(g.lower() in answer for g in (row.get("gold_answers") or []) if g)


def _safe_auc(y_true: List[int], scores: List[float]) -> float:
    """ROC-AUC; returns NaN if undefined (single class or empty)."""
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def _safe_ap(y_true: List[int], scores: List[float]) -> float:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return float("nan")
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return float("nan")


def compute_extended_metrics(rows: List[Dict]) -> Dict:
    """All scalar metrics + confusion matrices for one system's rows."""
    n = len(rows)
    if n == 0:
        return {}

    # --- Binary unanswerable detection --------------------------------------
    y_true = [0 if r.get("gold_answerable", True) else 1 for r in rows]
    y_pred = [0 if r.get("action") == "ANSWER" else 1 for r in rows]
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    cm_binary = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    # Continuous score for AUC: unanswerable_score = 1 - answerability_confidence.
    # Documented: higher = more likely unanswerable. NEVER derived from hard labels.
    scores = [
        float(
            r.get(
                "unanswerable_score",
                1.0
                - float(
                    r.get(
                        "answerability_confidence",
                        r.get("reasoner_confidence", 0.0),
                    )
                ),
            )
        )
        for r in rows
    ]
    roc_auc = _safe_auc(y_true, scores)
    pr_auc = _safe_ap(y_true, scores)

    n_answerable = sum(1 for t in y_true if t == 0)
    n_unanswerable = n - n_answerable

    # --- Behaviour / abstention ---------------------------------------------
    action_counts = Counter(r.get("action") for r in rows)
    abstention_rate = action_counts.get("ABSTAIN", 0) / n
    clarification_rate = action_counts.get("CLARIFY", 0) / n
    answer_rate = action_counts.get("ANSWER", 0) / n
    coverage = answer_rate  # proportion of questions that receive an answer

    # Abstention Precision / Recall (positive = gold unanswerable; pred abstain).
    # Definition: ABSTAIN only (not CLARIFY) for abstention metrics.
    abstain_rows = [r for r in rows if r.get("action") == "ABSTAIN"]
    n_abstain = len(abstain_rows)
    true_abstain = sum(1 for r in abstain_rows if not r.get("gold_answerable", True))
    abstention_precision = (true_abstain / n_abstain) if n_abstain else 0.0
    abstention_recall = (
        (true_abstain / n_unanswerable) if n_unanswerable else 0.0
    )

    # --- Hallucination ------------------------------------------------------
    # Hallucination = substantive ANSWER on a gold-unanswerable question.
    hallucinated = sum(
        1
        for r in rows
        if not r.get("gold_answerable", True)
        and r.get("action") == "ANSWER"
        and (r.get("final_answer", "") or "").strip()
    )
    hallucination_rate = (hallucinated / n_unanswerable) if n_unanswerable else 0.0
    unsupported_answer_rate = hallucination_rate  # alias under current annotation budget
    correct_abstention = sum(
        1
        for r in rows
        if not r.get("gold_answerable", True) and r.get("action") == "ABSTAIN"
    )
    correct_abstention_rate = (
        (correct_abstention / n_unanswerable) if n_unanswerable else 0.0
    )

    # --- False rejection / loose answer quality -----------------------------
    false_rejections = sum(
        1 for r in rows if r.get("gold_answerable", True) and r.get("action") != "ANSWER"
    )
    false_rejection_rate = (false_rejections / n_answerable) if n_answerable else 0.0
    false_acceptance = sum(
        1
        for r in rows
        if not r.get("gold_answerable", True) and r.get("action") == "ANSWER"
    )
    false_acceptance_rate = (
        (false_acceptance / n_unanswerable) if n_unanswerable else 0.0
    )
    answered_correct = sum(
        1 for r in rows if r.get("gold_answerable", True) and _answered_correctly(r)
    )
    answered_correct_rate = (answered_correct / n_answerable) if n_answerable else 0.0

    # --- Policy evaluation --------------------------------------------------
    compliant = sum(1 for r in rows if r.get("action") == policy_expected_action(r))
    policy_compliance_rate = compliant / n

    exp = [r.get("expected_action", "ABSTAIN") for r in rows]
    act = [r.get("action", "ANSWER") for r in rows]
    cm_decision = confusion_matrix(exp, act, labels=ACTIONS).tolist()

    correct_answer = sum(
        1 for r in rows if r.get("expected_action") == "ANSWER" and r.get("action") == "ANSWER"
    )
    n_exp_answer = sum(1 for r in rows if r.get("expected_action") == "ANSWER")
    correct_clarify = sum(
        1
        for r in rows
        if r.get("expected_action") == "CLARIFY" and r.get("action") == "CLARIFY"
    )
    n_exp_clarify = sum(1 for r in rows if r.get("expected_action") == "CLARIFY")
    correct_abstain = sum(
        1
        for r in rows
        if r.get("expected_action") == "ABSTAIN" and r.get("action") == "ABSTAIN"
    )
    n_exp_abstain = sum(1 for r in rows if r.get("expected_action") == "ABSTAIN")

    clarification_quality = (
        (correct_clarify / n_exp_clarify) if n_exp_clarify else float("nan")
    )

    return {
        "n": n,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(roc_auc, 4) if roc_auc == roc_auc else float("nan"),
        "pr_auc": round(pr_auc, 4) if pr_auc == pr_auc else float("nan"),
        "auc_score_name": "unanswerable_score=1-answerability_confidence",
        "hallucination_rate": round(float(hallucination_rate), 4),
        "unsupported_answer_rate": round(float(unsupported_answer_rate), 4),
        "correct_abstention_rate": round(float(correct_abstention_rate), 4),
        "abstention_rate": round(float(abstention_rate), 4),
        "abstention_precision": round(float(abstention_precision), 4),
        "abstention_recall": round(float(abstention_recall), 4),
        "clarification_rate": round(float(clarification_rate), 4),
        "answer_rate": round(float(answer_rate), 4),
        "coverage": round(float(coverage), 4),
        "false_rejection_rate": round(float(false_rejection_rate), 4),
        "false_acceptance_rate": round(float(false_acceptance_rate), 4),
        "answered_correct_rate": round(float(answered_correct_rate), 4),
        "policy_compliance_rate": round(float(policy_compliance_rate), 4),
        "correct_answer_rate": round(
            (correct_answer / n_exp_answer) if n_exp_answer else 0.0, 4
        ),
        "correct_clarify_rate": round(
            (correct_clarify / n_exp_clarify) if n_exp_clarify else 0.0, 4
        ),
        "correct_abstain_rate": round(
            (correct_abstain / n_exp_abstain) if n_exp_abstain else 0.0, 4
        ),
        "clarification_quality": round(float(clarification_quality), 4)
        if clarification_quality == clarification_quality
        else float("nan"),
        "confusion_matrix_binary": cm_binary,
        "confusion_matrix_decision": cm_decision,
    }


def compute_hallucination_by_category(rows: List[Dict]) -> Dict[str, Dict]:
    """Per-category hallucination / abstention rates for unanswerable-like cats."""
    result = {}
    cats = sorted({r.get("category", "unlabeled") for r in rows})
    for c in cats:
        cat_rows = [r for r in rows if r.get("category") == c]
        n = len(cat_rows)
        if n == 0:
            continue
        # For answerable, hallucination vs gold is not defined the same way.
        n_unans = sum(1 for r in cat_rows if not r.get("gold_answerable", True))
        halluc = sum(
            1
            for r in cat_rows
            if not r.get("gold_answerable", True)
            and r.get("action") == "ANSWER"
            and (r.get("final_answer", "") or "").strip()
        )
        correct_abs = sum(
            1
            for r in cat_rows
            if not r.get("gold_answerable", True) and r.get("action") == "ABSTAIN"
        )
        result[c] = {
            "n": n,
            "n_unanswerable_in_cat": n_unans,
            "hallucination_rate": round((halluc / n_unans) if n_unans else 0.0, 4),
            "correct_abstention_rate": round(
                (correct_abs / n_unans) if n_unans else 0.0, 4
            ),
            "answer": sum(1 for r in cat_rows if r.get("action") == "ANSWER"),
            "clarify": sum(1 for r in cat_rows if r.get("action") == "CLARIFY"),
            "abstain": sum(1 for r in cat_rows if r.get("action") == "ABSTAIN"),
        }
    return result


def compute_by_mode(rows: List[Dict]) -> Dict[str, Dict]:
    modes = sorted({r.get("mode", "full_system") for r in rows})
    return {m: compute_extended_metrics([r for r in rows if r.get("mode") == m]) for m in modes}


def compute_per_category(rows: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    modes = sorted({r.get("mode", "full_system") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        cats = sorted({r.get("category", "unlabeled") for r in mode_rows})
        result[m] = {}
        for c in cats:
            cat_rows = [r for r in mode_rows if r.get("category") == c]
            n = len(cat_rows)
            detected = sum(1 for r in cat_rows if r.get("action") != "ANSWER")
            correct_action = sum(
                1 for r in cat_rows if r.get("action") == r.get("expected_action")
            )
            actions = Counter(r.get("action") for r in cat_rows)
            hall = compute_hallucination_by_category(cat_rows).get(c, {})
            result[m][c] = {
                "n": n,
                "detection_rate": round(detected / n, 4) if n else 0.0,
                "expected_action_accuracy": round(correct_action / n, 4) if n else 0.0,
                "answer": actions.get("ANSWER", 0),
                "clarify": actions.get("CLARIFY", 0),
                "abstain": actions.get("ABSTAIN", 0),
                "hallucination_rate": hall.get("hallucination_rate", 0.0),
                "correct_abstention_rate": hall.get("correct_abstention_rate", 0.0),
            }
    return result


def compute_per_course(rows: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    modes = sorted({r.get("mode", "full_system") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        courses = sorted({str(r.get("course") or "unknown") for r in mode_rows})
        result[m] = {}
        for c in courses:
            crs = [r for r in mode_rows if str(r.get("course") or "unknown") == c]
            metrics = compute_extended_metrics(crs)
            result[m][c] = {
                k: metrics[k]
                for k in (
                    "n",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "coverage",
                    "hallucination_rate",
                    "policy_compliance_rate",
                )
                if k in metrics
            }
    return result


def compute_by_seed(rows: List[Dict]) -> Dict[str, Dict[int, Dict]]:
    result: Dict[str, Dict[int, Dict]] = {}
    modes = sorted({r.get("mode", "full_system") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        seeds = sorted({r.get("seed", 0) for r in mode_rows})
        result[m] = {
            s: compute_extended_metrics([r for r in mode_rows if r.get("seed") == s])
            for s in seeds
        }
    return result
