"""Confidence-threshold sensitivity analysis.

The thesis asks for the "percentage of answers dropped based on the confidence
threshold". This script sweeps the reasoner confidence threshold and, for each
value, re-derives the abstain decision from the recorded signals and reports the
trade-off between answering and refusing:

    answer_rate           : fraction of questions the system chooses to answer
    dropped_rate          : fraction dropped/abstained (= 1 - answer_rate)
    false_rejection_rate  : answerable questions wrongly dropped
    f1_unanswerable       : F1 for detecting unanswerable questions

A question is answered at threshold tau iff:
    model_is_answerable AND retriever_confidence >= retriever_floor
    AND reasoner_confidence >= tau

Inputs come from the predictions file produced by ``run_eval.py`` (the
policy_aware rows, which carry the raw signals).

Usage:
    python -m eval.threshold_sweep --predictions results/predictions.jsonl
    python -m eval.threshold_sweep --retriever-floor 0.2 --step 0.05
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from eval.metrics import load_predictions


def _rows_with_signals(rows: List[Dict]) -> List[Dict]:
    """Prefer policy_aware rows; they carry the retriever/reasoner signals."""
    policy = [r for r in rows if r.get("mode") == "policy_aware"]
    return policy if policy else rows


def sweep(rows: List[Dict], retriever_floor: float, step: float) -> pd.DataFrame:
    rows = _rows_with_signals(rows)
    n = len(rows)
    n_answerable = sum(1 for r in rows if r.get("gold_answerable", True))

    thresholds = [round(i * step, 4) for i in range(int(1.0 / step) + 1)]
    records = []
    for tau in thresholds:
        y_true, y_pred = [], []  # 1 = unanswerable / not-answered
        answered = 0
        false_rejections = 0
        for r in rows:
            gold_unans = 0 if r.get("gold_answerable", True) else 1
            will_answer = (
                bool(r.get("model_is_answerable", False))
                and float(r.get("retriever_confidence", 0.0)) >= retriever_floor
                and float(r.get("reasoner_confidence", 0.0)) >= tau
            )
            y_true.append(gold_unans)
            y_pred.append(0 if will_answer else 1)
            if will_answer:
                answered += 1
            elif r.get("gold_answerable", True):
                false_rejections += 1

        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average="binary", zero_division=0
        )
        answer_rate = answered / n if n else 0.0
        records.append(
            {
                "threshold": tau,
                "answer_rate": round(answer_rate, 4),
                "dropped_rate": round(1 - answer_rate, 4),
                "false_rejection_rate": round(
                    (false_rejections / n_answerable) if n_answerable else 0.0, 4
                ),
                "f1_unanswerable": round(float(f1), 4),
            }
        )
    return pd.DataFrame(records)


def plot_sweep(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    for col, marker in [
        ("answer_rate", "o"),
        ("dropped_rate", "s"),
        ("false_rejection_rate", "^"),
        ("f1_unanswerable", "d"),
    ]:
        plt.plot(df["threshold"], df[col], marker=marker, label=col)
    plt.title("Confidence-Threshold Sensitivity (Answer vs. Refuse Trade-off)")
    plt.xlabel("Reasoner confidence threshold (tau)")
    plt.ylabel("Rate / F1")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[threshold_sweep] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Confidence-threshold sensitivity analysis.")
    parser.add_argument("--predictions", default="results/predictions.jsonl")
    parser.add_argument("--retriever-floor", type=float, default=0.2)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    rows = load_predictions(Path(args.predictions))
    df = sweep(rows, args.retriever_floor, args.step)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "threshold_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"[threshold_sweep] Saved {csv_path}\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(df)
    print()

    plot_sweep(df, out_dir / "threshold_sweep.png")
    print("[threshold_sweep] Done.")


if __name__ == "__main__":
    main()
