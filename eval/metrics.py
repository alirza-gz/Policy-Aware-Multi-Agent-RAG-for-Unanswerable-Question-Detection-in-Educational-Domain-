"""Quantitative metrics for unanswerable-question detection.

Task framing (binary detection of the *unanswerable* class):
    y_true = 1 when the gold label is unanswerable (answerable == False)
    y_pred = 1 when the system did NOT provide an answer (action != ANSWER),
             i.e. it abstained or asked for clarification.

Reported metrics per mode:
    - accuracy                : overall unanswerable-detection accuracy
    - precision / recall / f1 : for the unanswerable (positive) class
    - false_rejection_rate    : answerable questions that were wrongly not answered
    - abstain_rate            : fraction of questions ending in ABSTAIN
    - clarify_rate            : fraction of questions ending in CLARIFY
    - answer_rate             : fraction of questions ending in ANSWER
    - answered_correct_rate   : answerable questions answered with the gold span
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

ACTION_ANSWER = "ANSWER"
ACTION_CLARIFY = "CLARIFY"
ACTION_ABSTAIN = "ABSTAIN"


def load_predictions(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _answered_correctly(row: Dict) -> bool:
    """Loose correctness check: any gold span appears in the returned answer."""
    if row.get("action") != ACTION_ANSWER:
        return False
    answer = (row.get("final_answer", "") or "").lower()
    gold = row.get("gold_answers", []) or []
    return any(g.lower() in answer for g in gold if g)


def compute_metrics(rows: List[Dict]) -> Dict:
    n = len(rows)
    if n == 0:
        return {}

    y_true = [0 if r.get("gold_answerable", True) else 1 for r in rows]   # 1 = unanswerable
    y_pred = [0 if r.get("action") == ACTION_ANSWER else 1 for r in rows]  # 1 = not answered

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )

    # Confusion matrix over classes [0=answerable, 1=unanswerable].
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    n_answerable = sum(1 for r in rows if r.get("gold_answerable", True))
    n_unanswerable = n - n_answerable

    # False rejection: answerable questions the system refused to answer.
    false_rejections = sum(
        1 for r in rows if r.get("gold_answerable", True) and r.get("action") != ACTION_ANSWER
    )
    false_rejection_rate = (false_rejections / n_answerable) if n_answerable else 0.0

    abstain_rate = sum(1 for r in rows if r.get("action") == ACTION_ABSTAIN) / n
    clarify_rate = sum(1 for r in rows if r.get("action") == ACTION_CLARIFY) / n
    answer_rate = sum(1 for r in rows if r.get("action") == ACTION_ANSWER) / n

    correct_on_answerable = sum(
        1 for r in rows if r.get("gold_answerable", True) and _answered_correctly(r)
    )
    answered_correct_rate = (correct_on_answerable / n_answerable) if n_answerable else 0.0

    return {
        "n": n,
        "n_answerable": n_answerable,
        "n_unanswerable": n_unanswerable,
        "accuracy": round(float(accuracy), 4),
        "precision_unanswerable": round(float(precision), 4),
        "recall_unanswerable": round(float(recall), 4),
        "f1_unanswerable": round(float(f1), 4),
        "false_rejection_rate": round(float(false_rejection_rate), 4),
        "abstain_rate": round(float(abstain_rate), 4),
        "clarify_rate": round(float(clarify_rate), 4),
        "answer_rate": round(float(answer_rate), 4),
        "answered_correct_rate": round(float(answered_correct_rate), 4),
        "confusion_matrix": cm,
    }


def compute_by_mode(rows: List[Dict]) -> Dict[str, Dict]:
    modes = sorted({r.get("mode", "policy_aware") for r in rows})
    return {mode: compute_metrics([r for r in rows if r.get("mode") == mode]) for mode in modes}


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics.")
    parser.add_argument("--predictions", default="results/predictions.jsonl")
    parser.add_argument("--out", default="results/metrics.json")
    args = parser.parse_args()

    rows = load_predictions(Path(args.predictions))
    by_mode = compute_by_mode(rows)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(by_mode, f, indent=2, ensure_ascii=False)

    print(f"[metrics] Wrote metrics to {args.out}\n")
    for mode, m in by_mode.items():
        print(f"=== {mode} ===")
        for k, v in m.items():
            if k == "confusion_matrix":
                continue
            print(f"  {k:28s}: {v}")
        print()


if __name__ == "__main__":
    main()
