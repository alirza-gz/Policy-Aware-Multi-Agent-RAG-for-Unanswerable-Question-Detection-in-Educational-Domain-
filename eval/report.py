"""Statistical aggregation and thesis-ready CSV / Markdown report generation.

Consumes the combined predictions written by ``eval/run_experiments.py`` and
produces, in the output directory:

    metrics_summary.csv        one row per system: mean +/- std of every metric
                               over the seeded runs
    baseline_improvement.csv   % improvement of policy_aware over each baseline
    ablation_comparison.csv    full system vs each ablation (delta per metric)
    per_category.csv           per-category behaviour of every system
    confusion_binary_<sys>.csv / confusion_decision_<sys>.csv
    EVALUATION_REPORT.md       consolidated Markdown report with all tables

Usage:
    python -m eval.report
    python -m eval.report --predictions results/predictions_all.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from eval.metrics_extended import (
    ACTIONS,
    compute_by_mode,
    compute_by_seed,
    compute_per_category,
)

# Metrics reported with mean/std and used in comparisons.
SCALAR_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "hallucination_rate",
    "abstention_rate",
    "clarification_rate",
    "answer_rate",
    "false_rejection_rate",
    "answered_correct_rate",
    "policy_compliance_rate",
]
# For these, LOWER is better (improvement sign is flipped).
LOWER_IS_BETTER = {"hallucination_rate", "false_rejection_rate"}

BASELINES = ["vanilla_rag", "single_agent_rag"]
FULL = "policy_aware"
ABLATIONS = ["full_system", "no_governance", "no_clarification", "no_retriever_confidence", "no_policy_rules"]


def _save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a CSV, tolerating files locked by e.g. an open Excel window."""
    try:
        df.to_csv(path, **kwargs)
        print(f"[report] Saved {path}")
    except PermissionError:
        print(f"[report] WARNING: {path} is locked by another program (Excel?); skipped. "
              "Close it and re-run to refresh this file.")


def load_predictions(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summary_table(by_seed: Dict[str, Dict[int, Dict]]) -> pd.DataFrame:
    """One row per system, ``metric_mean`` / ``metric_std`` columns."""
    records = []
    for mode, seed_metrics in by_seed.items():
        rec = {"system": mode, "n_runs": len(seed_metrics)}
        for metric in SCALAR_METRICS:
            vals = [m[metric] for m in seed_metrics.values() if metric in m]
            rec[f"{metric}_mean"] = round(float(np.mean(vals)), 4) if vals else float("nan")
            rec[f"{metric}_std"] = round(float(np.std(vals)), 4) if vals else float("nan")
        records.append(rec)
    return pd.DataFrame(records).set_index("system")


def improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """% improvement of the full system over each baseline, per metric."""
    records = []
    if FULL not in summary.index:
        return pd.DataFrame()
    for baseline in BASELINES:
        if baseline not in summary.index:
            continue
        rec = {"baseline": baseline}
        for metric in SCALAR_METRICS:
            base = summary.loc[baseline, f"{metric}_mean"]
            full = summary.loc[FULL, f"{metric}_mean"]
            if base == 0:
                rec[metric] = float("nan")
                continue
            change = (full - base) / abs(base) * 100.0
            if metric in LOWER_IS_BETTER:
                change = -change
            rec[metric] = round(change, 2)
        records.append(rec)
    return pd.DataFrame(records).set_index("baseline")


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Absolute metric deltas of each ablation vs the full system."""
    if "full_system" not in summary.index:
        return pd.DataFrame()
    records = []
    for abl in ABLATIONS:
        if abl not in summary.index:
            continue
        rec = {"configuration": abl}
        for metric in SCALAR_METRICS:
            full = summary.loc["full_system", f"{metric}_mean"]
            val = summary.loc[abl, f"{metric}_mean"]
            rec[f"{metric}"] = round(val, 4)
            rec[f"{metric}_delta"] = round(val - full, 4)
        records.append(rec)
    return pd.DataFrame(records).set_index("configuration")


def per_category_table(per_cat: Dict[str, Dict[str, Dict]]) -> pd.DataFrame:
    records = []
    for mode, cats in per_cat.items():
        for cat, m in cats.items():
            records.append({"system": mode, "category": cat, **m})
    return pd.DataFrame(records)


def confusion_frames(by_mode: Dict[str, Dict]) -> Dict[str, Dict[str, pd.DataFrame]]:
    frames = {}
    for mode, m in by_mode.items():
        if not m:
            continue
        binary = pd.DataFrame(
            m["confusion_matrix_binary"],
            index=["gold_answerable", "gold_unanswerable"],
            columns=["pred_answered", "pred_not_answered"],
        )
        decision = pd.DataFrame(
            m["confusion_matrix_decision"],
            index=[f"expected_{a}" for a in ACTIONS],
            columns=[f"pred_{a}" for a in ACTIONS],
        )
        frames[mode] = {"binary": binary, "decision": decision}
    return frames


def _md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def write_markdown_report(
    out_path: Path,
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    ablation: pd.DataFrame,
    per_cat_df: pd.DataFrame,
    confusions: Dict[str, Dict[str, pd.DataFrame]],
    n_rows: int,
    seeds: List[int],
    reasoning_modes: List[str],
) -> None:
    lines = [
        "# Evaluation Report - Policy-Aware Multi-Agent RAG",
        "",
        "Automatically generated by `python -m eval.report`. All systems share the",
        "same retriever, embedding model, LLM, prompts, and dataset; they differ only",
        "in the decision layer. Runs are reproducible via `eval/experiments.yml`",
        f"(seeds: {seeds}). Total prediction rows: {n_rows}.",
        f"Reasoning backend: **{', '.join(reasoning_modes) or 'unknown'}**.",
        "",
        "## Metric definitions",
        "",
        "- **accuracy / precision / recall / f1** - binary unanswerable-question",
        "  detection (positive class = unanswerable; predicted positive = the system",
        "  did not answer).",
        "- **hallucination_rate** - unanswerable questions that received a substantive",
        "  answer (lower is better).",
        "- **abstention_rate / clarification_rate / answer_rate** - fraction of",
        "  questions ending in ABSTAIN / CLARIFY / ANSWER.",
        "- **false_rejection_rate** - answerable questions wrongly not answered",
        "  (lower is better).",
        "- **policy_compliance_rate** - decisions consistent with the configured",
        "  governance policy given the recorded signals.",
        "",
        "## 1. Baseline comparison (mean over runs, std in metrics_summary.csv)",
        "",
        _md(summary[[f"{m}_mean" for m in SCALAR_METRICS]].rename(columns=lambda c: c[:-5])),
        "",
        "## 2. Improvement of the policy-aware system over the baselines (%)",
        "",
        "Positive = the policy-aware system is better (sign flipped for",
        "lower-is-better metrics).",
        "",
        _md(improvement),
        "",
        "## 3. Ablation study (vs full system, `_delta` columns)",
        "",
        _md(ablation),
        "",
        "## 4. Per-category analysis",
        "",
        "`detection_rate` = fraction of the category's questions not answered;",
        "`expected_action_accuracy` = fraction receiving the gold action.",
        "",
        _md(per_cat_df),
        "",
        "## 5. Confusion matrices",
        "",
    ]
    for mode, frames in confusions.items():
        lines += [
            f"### {mode}",
            "",
            "Answerable vs unanswerable:",
            "",
            _md(frames["binary"]),
            "",
            "Governance decisions (expected vs predicted):",
            "",
            _md(frames["decision"]),
            "",
        ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate CSV + Markdown evaluation reports.")
    parser.add_argument("--predictions", default="results/predictions_all.jsonl")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_predictions(Path(args.predictions))
    if not rows:
        raise SystemExit(f"No predictions found in {args.predictions}")
    seeds = sorted({r.get("seed", 0) for r in rows})
    reasoning_modes = sorted({r.get("reasoning_mode", "unknown") for r in rows})

    by_seed = compute_by_seed(rows)
    by_mode = compute_by_mode(rows)
    per_cat = compute_per_category(rows)

    summary = summary_table(by_seed)
    _save_csv(summary, out_dir / "metrics_summary.csv")

    improvement = improvement_table(summary)
    _save_csv(improvement, out_dir / "baseline_improvement.csv")

    ablation = ablation_table(summary)
    _save_csv(ablation, out_dir / "ablation_comparison.csv")

    per_cat_df = per_category_table(per_cat)
    _save_csv(per_cat_df, out_dir / "per_category.csv", index=False)

    confusions = confusion_frames(by_mode)
    for mode, frames in confusions.items():
        _save_csv(frames["binary"], out_dir / f"confusion_binary_{mode}.csv")
        _save_csv(frames["decision"], out_dir / f"confusion_decision_{mode}.csv")

    write_markdown_report(
        out_dir / "EVALUATION_REPORT.md",
        summary,
        improvement,
        ablation,
        per_cat_df,
        confusions,
        len(rows),
        seeds,
        reasoning_modes,
    )
    print("[report] Done.")


if __name__ == "__main__":
    main()
