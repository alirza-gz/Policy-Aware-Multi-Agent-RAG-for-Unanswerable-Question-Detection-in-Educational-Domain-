"""Extended thesis metrics for unanswerable-question detection.

Builds on ``eval/metrics.py`` (which stays unchanged for backward
compatibility) and adds every metric required by the evaluation plan:

    - accuracy / precision / recall / F1 (unanswerable class)
    - hallucination rate      : gold-unanswerable questions that received a
                                substantive answer (the system asserted content
                                it had no evidence for)
    - abstention rate         : fraction of questions ending in ABSTAIN
    - clarification rate      : fraction of questions ending in CLARIFY
    - policy compliance rate  : fraction of decisions that match what the
                                configured governance policy dictates given the
                                recorded signals (reference re-application of
                                the policy, NOT a change to it)
    - binary confusion matrix (answerable vs unanswerable)
    - governance decision confusion matrix (expected vs predicted action)
    - per-category breakdown of all of the above

All functions are pure: they consume prediction rows produced by
``eval/run_experiments.py`` and return plain dicts.
"""

from collections import Counter
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.config import Config

ACTIONS = ["ANSWER", "CLARIFY", "ABSTAIN"]

# Reference policy parameters (identical to the deployed GovernanceAgent).
_GOV = dict(getattr(Config, "GOVERNANCE", {}) or {})
_RETR_ABSTAIN = float(_GOV.get("retriever_abstain_below", 0.2))
_REAS_ABSTAIN = float(_GOV.get("reasoner_abstain_below", 0.3))
_CLARIFY_BAND = list(_GOV.get("clarify_band", [0.3, 0.5]) or [0.3, 0.5])
_BANNED = [str(p).lower() for p in (getattr(Config, "BANNED_PHRASES", []) or [])]


def policy_expected_action(row: Dict) -> str:
    """Re-apply the configured policy to a row's recorded signals.

    Mirrors GovernanceAgent.decide() precedence exactly (safety -> answerability
    -> clarification -> answer). Used to measure Policy Compliance Rate.
    """
    answer = (row.get("raw_answer", "") or "").lower()
    if any(p and p in answer for p in _BANNED):
        return "ABSTAIN"
    conf = float(row.get("reasoner_confidence", 0.0))
    retr = float(row.get("retriever_confidence", 0.0))
    if not row.get("model_is_answerable", False):
        return "ABSTAIN"
    if retr < _RETR_ABSTAIN or conf < _REAS_ABSTAIN:
        return "ABSTAIN"
    if row.get("model_needs_clarification", False) or (_CLARIFY_BAND[0] <= conf < _CLARIFY_BAND[1]):
        return "CLARIFY"
    return "ANSWER"


def _answered_correctly(row: Dict) -> bool:
    """Loose span match: any gold answer appears in the returned answer."""
    if row.get("action") != "ANSWER":
        return False
    answer = (row.get("final_answer", "") or "").lower()
    return any(g.lower() in answer for g in (row.get("gold_answers") or []) if g)


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

    n_answerable = sum(1 for t in y_true if t == 0)
    n_unanswerable = n - n_answerable

    # --- Behaviour rates -----------------------------------------------------
    action_counts = Counter(r.get("action") for r in rows)
    abstention_rate = action_counts.get("ABSTAIN", 0) / n
    clarification_rate = action_counts.get("CLARIFY", 0) / n
    answer_rate = action_counts.get("ANSWER", 0) / n

    # --- Hallucination rate --------------------------------------------------
    # A hallucination = a substantive (non-empty) answer asserted for a
    # question that is unanswerable from the corpus.
    hallucinated = sum(
        1
        for r in rows
        if not r.get("gold_answerable", True)
        and r.get("action") == "ANSWER"
        and (r.get("final_answer", "") or "").strip()
    )
    hallucination_rate = (hallucinated / n_unanswerable) if n_unanswerable else 0.0

    # --- False rejection / answer quality ------------------------------------
    false_rejections = sum(
        1 for r in rows if r.get("gold_answerable", True) and r.get("action") != "ANSWER"
    )
    false_rejection_rate = (false_rejections / n_answerable) if n_answerable else 0.0
    answered_correct = sum(
        1 for r in rows if r.get("gold_answerable", True) and _answered_correctly(r)
    )
    answered_correct_rate = (answered_correct / n_answerable) if n_answerable else 0.0

    # --- Policy compliance ---------------------------------------------------
    compliant = sum(1 for r in rows if r.get("action") == policy_expected_action(r))
    policy_compliance_rate = compliant / n

    # --- Governance decision confusion matrix --------------------------------
    exp = [r.get("expected_action", "ABSTAIN") for r in rows]
    act = [r.get("action", "ANSWER") for r in rows]
    cm_decision = confusion_matrix(exp, act, labels=ACTIONS).tolist()

    return {
        "n": n,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "hallucination_rate": round(float(hallucination_rate), 4),
        "abstention_rate": round(float(abstention_rate), 4),
        "clarification_rate": round(float(clarification_rate), 4),
        "answer_rate": round(float(answer_rate), 4),
        "false_rejection_rate": round(float(false_rejection_rate), 4),
        "answered_correct_rate": round(float(answered_correct_rate), 4),
        "policy_compliance_rate": round(float(policy_compliance_rate), 4),
        "confusion_matrix_binary": cm_binary,          # rows: gold [ans, unans]
        "confusion_matrix_decision": cm_decision,      # rows: expected action
    }


def compute_by_mode(rows: List[Dict]) -> Dict[str, Dict]:
    modes = sorted({r.get("mode", "policy_aware") for r in rows})
    return {m: compute_extended_metrics([r for r in rows if r.get("mode") == m]) for m in modes}


def compute_per_category(rows: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """mode -> category -> metrics (detection recall etc. within the category)."""
    result: Dict[str, Dict[str, Dict]] = {}
    modes = sorted({r.get("mode", "policy_aware") for r in rows})
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
            result[m][c] = {
                "n": n,
                "detection_rate": round(detected / n, 4) if n else 0.0,
                "expected_action_accuracy": round(correct_action / n, 4) if n else 0.0,
                "answer": actions.get("ANSWER", 0),
                "clarify": actions.get("CLARIFY", 0),
                "abstain": actions.get("ABSTAIN", 0),
            }
    return result


def compute_by_seed(rows: List[Dict]) -> Dict[str, Dict[int, Dict]]:
    """mode -> seed -> metrics, for mean/std statistical reporting."""
    result: Dict[str, Dict[int, Dict]] = {}
    modes = sorted({r.get("mode", "policy_aware") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        seeds = sorted({r.get("seed", 0) for r in mode_rows})
        result[m] = {
            s: compute_extended_metrics([r for r in mode_rows if r.get("seed") == s])
            for s in seeds
        }
    return result
