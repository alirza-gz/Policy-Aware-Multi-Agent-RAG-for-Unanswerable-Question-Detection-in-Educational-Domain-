"""Aggregate evaluation results into tables, figures, and a qualitative report.

Outputs (default ``results/`` directory):
    - metrics_comparison.csv    : per-mode quantitative metrics (Pandas)
    - metrics_comparison.png    : grouped bar chart, policy_aware vs baseline
    - action_distribution.png   : ANSWER / CLARIFY / ABSTAIN breakdown per mode
    - trace_report.md           : qualitative trace analysis (transparency of decisions)

Usage:
    python -m eval.analyze --predictions results/predictions.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import pandas as pd

from eval.metrics import load_predictions, compute_by_mode

# Metrics shown side-by-side in the comparison bar chart.
COMPARISON_METRICS = [
    "accuracy",
    "precision_unanswerable",
    "recall_unanswerable",
    "f1_unanswerable",
    "false_rejection_rate",
    "abstain_rate",
    "clarify_rate",
]
ACTIONS = ["ANSWER", "CLARIFY", "ABSTAIN"]


def build_metrics_table(by_mode: Dict[str, Dict]) -> pd.DataFrame:
    df = pd.DataFrame(by_mode).T
    df.index.name = "mode"
    return df


def plot_metrics_comparison(df: pd.DataFrame, out_path: Path) -> None:
    metrics = [m for m in COMPARISON_METRICS if m in df.columns]
    plot_df = df[metrics]

    ax = plot_df.T.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Policy-Aware vs Baseline - Unanswerable Question Detection")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Mode")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[analyze] Saved {out_path}")


def plot_action_distribution(rows: List[Dict], out_path: Path) -> None:
    modes = sorted({r.get("mode", "policy_aware") for r in rows})
    counts = {mode: {a: 0 for a in ACTIONS} for mode in modes}
    for r in rows:
        mode = r.get("mode", "policy_aware")
        action = r.get("action", "ANSWER")
        if action in counts[mode]:
            counts[mode][action] += 1

    dist_df = pd.DataFrame(counts).T[ACTIONS]
    ax = dist_df.plot(kind="bar", stacked=True, figsize=(8, 6))
    ax.set_title("Governance Action Distribution per Mode")
    ax.set_ylabel("Number of questions")
    ax.set_xlabel("Mode")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[analyze] Saved {out_path}")


def write_trace_report(rows: List[Dict], out_path: Path, max_examples: int = 15) -> None:
    """Qualitative report: inspect the transparency of policy-aware decisions."""
    policy_rows = [r for r in rows if r.get("mode") == "policy_aware"]

    def section(title: str, subset: List[Dict]) -> List[str]:
        lines = [f"## {title} ({len(subset)} cases)\n"]
        for r in subset[:max_examples]:
            gold = "answerable" if r.get("gold_answerable") else "unanswerable"
            lines.append(f"- **Q:** {r.get('question')}")
            lines.append(f"  - gold: `{gold}` | action: `{r.get('action')}`")
            lines.append(f"  - reason: {r.get('reason')}")
            lines.append(
                f"  - retriever_conf: {r.get('retriever_confidence')} | "
                f"reasoner_conf: {r.get('reasoner_confidence')} | "
                f"model_is_answerable: {r.get('model_is_answerable')}"
            )
            if r.get("trace"):
                lines.append(f"  - trace: `{json.dumps(r.get('trace'), ensure_ascii=False)}`")
            lines.append("")
        return lines

    abstained = [r for r in policy_rows if r.get("action") == "ABSTAIN"]
    clarified = [r for r in policy_rows if r.get("action") == "CLARIFY"]
    # Correctly abstained on genuinely unanswerable questions.
    correct_abstain = [r for r in abstained if not r.get("gold_answerable")]
    # Wrongly refused answerable questions (transparency failures worth inspecting).
    wrong_refuse = [
        r for r in policy_rows
        if r.get("gold_answerable") and r.get("action") != "ANSWER"
    ]

    lines = [
        "# Qualitative Trace Report\n",
        "This report supports the qualitative evaluation described in the research: ",
        "it inspects governance decisions and their stated reasons to assess the ",
        "transparency of the policy-aware system and its compliance with the ",
        "answerability policy.\n",
    ]
    lines += section("Correctly abstained on unanswerable questions", correct_abstain)
    lines += section("Requested clarification", clarified)
    lines += section("Wrongly refused answerable questions (false rejections)", wrong_refuse)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[analyze] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results.")
    parser.add_argument("--predictions", default="results/predictions.jsonl")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_predictions(Path(args.predictions))
    by_mode = compute_by_mode(rows)

    df = build_metrics_table(by_mode)
    csv_path = out_dir / "metrics_comparison.csv"
    df.to_csv(csv_path)
    print(f"[analyze] Saved {csv_path}")
    print("\n[analyze] Metrics comparison:\n")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.drop(columns=["confusion_matrix"], errors="ignore"))
    print()

    plot_metrics_comparison(df, out_dir / "metrics_comparison.png")
    plot_action_distribution(rows, out_dir / "action_distribution.png")
    write_trace_report(rows, out_dir / "trace_report.md")

    print("\n[analyze] Done.")


if __name__ == "__main__":
    main()
