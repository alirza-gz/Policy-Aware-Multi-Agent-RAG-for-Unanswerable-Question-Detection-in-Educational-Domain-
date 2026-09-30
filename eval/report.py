"""Thesis-ready CSV / Markdown report generation (advisor evaluation matrix).

Consumes predictions from ``eval.run_experiments`` and produces dedicated
tables for baselines, classification, abstention, retrieval, answer quality,
hallucination, policy, ablations, per-category/course, significance, and CIs.

Usage:
    python -m eval.report
    python -m eval.report --predictions results/predictions_all.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from eval.answer_quality import compute_answer_quality
from eval.baselines import ABLATION_SYSTEMS, BASELINE_SYSTEMS, RETRIEVAL_PROFILES
from eval.metrics_extended import (
    ACTIONS,
    compute_by_seed,
    compute_extended_metrics,
    compute_hallucination_by_category,
    compute_per_category,
    compute_per_course,
)
from eval.retrieval_metrics import (
    compute_retrieval_by_category,
    compute_retrieval_by_course,
    compute_retrieval_by_mode,
)
from eval.significance import (
    binary_detection_correct,
    bootstrap_ci,
    decision_correct,
    interpret_significance,
    mcnemar_test,
    paired_binary_labels,
)

SCALAR_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "hallucination_rate",
    "unsupported_answer_rate",
    "correct_abstention_rate",
    "abstention_rate",
    "abstention_precision",
    "abstention_recall",
    "clarification_rate",
    "coverage",
    "false_rejection_rate",
    "false_acceptance_rate",
    "answered_correct_rate",
    "policy_compliance_rate",
    "correct_answer_rate",
    "correct_clarify_rate",
    "correct_abstain_rate",
    "clarification_quality",
]
LOWER_IS_BETTER = {
    "hallucination_rate",
    "unsupported_answer_rate",
    "false_rejection_rate",
    "false_acceptance_rate",
}

FULL = "full_system"
BASELINES = [b for b in BASELINE_SYSTEMS if b != FULL]
ABLATIONS = list(ABLATION_SYSTEMS)


def _save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    try:
        df.to_csv(path, **kwargs)
        print(f"[report] Saved {path}")
    except PermissionError:
        print(f"[report] WARNING: {path} locked; skipped.")


def load_predictions(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summary_table(by_seed: Dict[str, Dict[int, Dict]]) -> pd.DataFrame:
    records = []
    for mode, seed_metrics in by_seed.items():
        rec = {"system": mode, "n_runs": len(seed_metrics)}
        for metric in SCALAR_METRICS:
            vals = [m[metric] for m in seed_metrics.values() if metric in m and m[metric] == m[metric]]
            rec[f"{metric}_mean"] = round(float(np.mean(vals)), 4) if vals else float("nan")
            rec[f"{metric}_std"] = round(float(np.std(vals)), 4) if vals else float("nan")
        records.append(rec)
    return pd.DataFrame(records).set_index("system")


def improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
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
            if base != base or full != full or base == 0:
                rec[metric] = float("nan")
            else:
                delta = (full - base) / abs(base) * 100.0
                if metric in LOWER_IS_BETTER:
                    delta = -delta
                rec[metric] = round(float(delta), 2)
        records.append(rec)
    return pd.DataFrame(records).set_index("baseline")


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    if FULL not in summary.index:
        return pd.DataFrame()
    for name in [FULL] + [a for a in ABLATIONS if a in summary.index]:
        rec = {"configuration": name}
        for metric in SCALAR_METRICS:
            val = summary.loc[name, f"{metric}_mean"] if name in summary.index else float("nan")
            full = summary.loc[FULL, f"{metric}_mean"]
            rec[metric] = val
            if val == val and full == full:
                rec[f"{metric}_delta"] = round(float(val - full), 4)
            else:
                rec[f"{metric}_delta"] = float("nan")
        records.append(rec)
    return pd.DataFrame(records).set_index("configuration")


def system_config_table(rows: List[Dict]) -> pd.DataFrame:
    records = []
    modes = sorted({r.get("mode") for r in rows})
    for m in modes:
        sample = next(r for r in rows if r.get("mode") == m)
        mode, rerank = RETRIEVAL_PROFILES.get(m, ("?", False))
        records.append(
            {
                "system": m,
                "llm_model": sample.get("llm_model"),
                "embed_model": sample.get("embed_model"),
                "retrieval_mode": mode,
                "reranker": rerank,
                "top_k_passages": sample.get("n_passages"),
                "rag_threshold": sample.get("rag_threshold"),
                "reasoning_mode": sample.get("reasoning_mode"),
                "seed_example": sample.get("seed"),
            }
        )
    return pd.DataFrame(records)


def significance_tables(rows: List[Dict], reference: str = FULL) -> pd.DataFrame:
    ref_rows = [r for r in rows if r.get("mode") == reference]
    records = []
    modes = sorted({r.get("mode") for r in rows if r.get("mode") != reference})
    for m in modes:
        other = [r for r in rows if r.get("mode") == m]
        for label_name, fn in (
            ("binary_detection_correct", binary_detection_correct),
            ("expected_action_correct", decision_correct),
        ):
            ya, yb = paired_binary_labels(ref_rows, other, fn)
            # McNemar: compare reference vs other; effect = other - reference
            # We pass y_a=ref, y_b=other so effect_size = mean(other)-mean(ref)
            res = mcnemar_test(ya, yb)
            records.append(
                {
                    "reference": reference,
                    "system": m,
                    "label": label_name,
                    "test": res.get("test"),
                    "n": res.get("n"),
                    "statistic": res.get("statistic"),
                    "p_value": res.get("p_value"),
                    "effect_size": res.get("effect_size"),
                    "effect_size_name": res.get("effect_size_name"),
                    "interpretation": interpret_significance(res.get("p_value", float("nan"))),
                }
            )
    return pd.DataFrame(records)


def confidence_interval_table(rows: List[Dict], metrics: Optional[List[str]] = None) -> pd.DataFrame:
    metrics = metrics or ["accuracy", "f1", "coverage", "hallucination_rate"]
    records = []
    modes = sorted({r.get("mode") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        # Per-question indicator vectors for bootstrap.
        # accuracy: binary detection correct
        indicators = {
            "accuracy": [binary_detection_correct(r) for r in mode_rows],
            "coverage": [1 if r.get("action") == "ANSWER" else 0 for r in mode_rows],
            "hallucination_rate": [
                1
                if (
                    not r.get("gold_answerable", True)
                    and r.get("action") == "ANSWER"
                    and (r.get("final_answer") or "").strip()
                )
                else 0
                for r in mode_rows
                if not r.get("gold_answerable", True)
            ],
        }
        # F1 is not a per-row mean; bootstrap over question resample via extended metrics
        # For CI table we report bootstrap on detection-correct as accuracy proxy and
        # leave f1 as metric-level via seed aggregation when available.
        for metric, vals in indicators.items():
            ci = bootstrap_ci(vals)
            records.append(
                {
                    "system": m,
                    "metric": metric,
                    "mean": ci["mean"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n": ci["n"],
                    "method": ci["method"],
                }
            )
        # F1 via cluster bootstrap on question ids (one seed pooled).
        f1_boot = _bootstrap_f1(mode_rows)
        records.append(
            {
                "system": m,
                "metric": "f1",
                "mean": f1_boot["mean"],
                "ci_low": f1_boot["ci_low"],
                "ci_high": f1_boot["ci_high"],
                "n": f1_boot["n"],
                "method": "cluster_bootstrap_questions",
            }
        )
    return pd.DataFrame(records)


def _bootstrap_f1(rows: List[Dict], n_boot: int = 500, seed: int = 42) -> Dict:
    from sklearn.metrics import precision_recall_fscore_support

    # Pool unique (id) within a single seed if multiple seeds present — use all rows.
    ids = sorted({r.get("id") for r in rows})
    by_id = {}
    for r in rows:
        by_id.setdefault(r.get("id"), []).append(r)
    if not ids:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(n_boot):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        sample = []
        for i in sample_ids:
            sample.extend(by_id[i])
        y_true = [0 if r.get("gold_answerable", True) else 1 for r in sample]
        y_pred = [0 if r.get("action") == "ANSWER" else 1 for r in sample]
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[1], average="binary", zero_division=0
        )
        f1s.append(float(f1))
    arr = np.asarray(f1s)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(float(np.quantile(arr, 0.025)), 4),
        "ci_high": round(float(np.quantile(arr, 0.975)), 4),
        "n": len(ids),
    }


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    ablation: pd.DataFrame,
    per_cat: pd.DataFrame,
    per_course: pd.DataFrame,
    retrieval: pd.DataFrame,
    answer_q: pd.DataFrame,
    halluc: pd.DataFrame,
    configs: pd.DataFrame,
    sig: pd.DataFrame,
    cis: pd.DataFrame,
    n_rows: int,
    seeds: List,
    reasoning_mode: str,
) -> None:
    lines = []
    lines.append("# Evaluation Report — Policy-Aware Multi-Agent RAG\n")
    lines.append(
        "Generated by `python -m eval.report`. FULL system = Multi-Agent + Hybrid "
        "Retrieval + Reranker + Answerability + Policy/Governance.\n"
    )
    lines.append(f"- Prediction rows: **{n_rows}**")
    lines.append(f"- Seeds: `{seeds}`")
    lines.append(f"- Reasoning backend: **{reasoning_mode}**\n")

    lines.append("## Metric definitions\n")
    lines.append(
        "- **accuracy / precision / recall / f1** — binary unanswerable detection "
        "(positive = gold unanswerable; predicted positive = action ≠ ANSWER).\n"
        "- **roc_auc / pr_auc** — from continuous `unanswerable_score = "
        "1 - answerability_confidence` (never from hard labels).\n"
        "- **coverage** — fraction of questions with action=ANSWER.\n"
        "- **abstention_precision** — P(gold unanswerable | ABSTAIN).\n"
        "- **abstention_recall** — P(ABSTAIN | gold unanswerable).\n"
        "- **hallucination_rate / unsupported_answer_rate** — gold-unanswerable "
        "questions that received a substantive ANSWER.\n"
        "- **policy_compliance_rate** — decision matches re-applied governance policy "
        "on recorded signals (not answer correctness).\n"
        "- **retrieval metrics** — binary relevance vs `evidence_ids` (nDCG binary).\n"
        "- **Agentic RAG** — honest single-agent proxy (reasoner self-governs); "
        "not an iterative tool-calling loop.\n"
    )

    def _add_table(title: str, df: pd.DataFrame):
        lines.append(f"## {title}\n")
        if df is None or df.empty:
            lines.append("_No data._\n")
        else:
            lines.append("```")
            lines.append(df.to_string())
            lines.append("```\n")

    _add_table("1. Dataset", pd.DataFrame({
        "field": ["questions_file", "n_prediction_rows", "seeds"],
        "value": ["data/eval/educational_v2_questions.jsonl", n_rows, str(seeds)],
    }))
    _add_table("2. Experimental configuration", configs)
    _add_table("3. Baseline comparison (mean over seeds)", summary.loc[
        [i for i in summary.index if i in BASELINE_SYSTEMS]
    ] if len(summary) else summary)
    _add_table("4. Answerability classification", summary[[
        c for c in summary.columns
        if any(m in c for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"))
    ]] if len(summary) else summary)
    _add_table("5. Abstention and coverage", summary[[
        c for c in summary.columns
        if any(m in c for m in ("abstention", "coverage", "clarification"))
    ]] if len(summary) else summary)
    _add_table("6. Retrieval quality", retrieval)
    _add_table("7. Answer quality (gold-answerable)", answer_q)
    _add_table("8. Hallucination (per category, pooled)", halluc)
    _add_table("9. Policy evaluation", summary[[
        c for c in summary.columns
        if any(m in c for m in ("policy", "correct_answer", "correct_clarify", "correct_abstain",
                                "false_acceptance", "false_rejection", "clarification_quality"))
    ]] if len(summary) else summary)
    _add_table("10. Ablation study (delta vs FULL)", ablation)
    _add_table("11. Per-category analysis", per_cat)
    _add_table("12. Per-course analysis", per_course)
    _add_table("13. Statistical significance (McNemar vs FULL)", sig)
    _add_table("14. Confidence intervals (bootstrap 95%)", cis)

    lines.append("## 15. Limitations\n")
    lines.append(
        "- Corpus for educational_v2 is reconstructed from `evidence_passage` texts "
        "already in the JSONL (held-out passages excluded). Full Wikipedia article "
        "context beyond those evidence spans is not regenerated.\n"
        "- nDCG uses binary relevance (no graded labels).\n"
        "- Agentic RAG is a single-agent self-governance proxy, not iterative planning.\n"
        "- Lexical groundedness is a proxy, not NLI entailment; LLM-judge metrics are "
        "not treated as objective ground truth.\n"
        "- ROC/PR-AUC require continuous `answerability_confidence`; systems without "
        "meaningful variation may yield unstable AUC.\n"
        "- Statistical significance does not imply practical importance; effect sizes "
        "are reported alongside p-values.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] Saved {path}")


def run(predictions_path: Path, out_dir: Path) -> None:
    rows = load_predictions(predictions_path)
    if not rows:
        raise SystemExit(f"No predictions in {predictions_path}")

    by_seed = compute_by_seed(rows)
    summary = summary_table(by_seed)
    improvement = improvement_table(summary)
    ablation = ablation_table(summary)

    # Per category / course
    per_cat_raw = compute_per_category(rows)
    per_cat_records = []
    for mode, cats in per_cat_raw.items():
        for cat, m in cats.items():
            per_cat_records.append({"system": mode, "category": cat, **m})
    per_cat = pd.DataFrame(per_cat_records)

    per_course_raw = compute_per_course(rows)
    per_course_records = []
    for mode, courses in per_course_raw.items():
        for course, m in courses.items():
            per_course_records.append({"system": mode, "course": course, **m})
    per_course = pd.DataFrame(per_course_records)

    # Retrieval (use first seed's rows per mode to avoid triple-counting, or all — mean)
    # Pool all seeds: metrics average over repeated retrievals with same pipeline.
    retrieval_by_mode = compute_retrieval_by_mode(rows, k=5)
    retrieval = pd.DataFrame(
        [{"system": m, **v} for m, v in retrieval_by_mode.items()]
    ).set_index("system") if retrieval_by_mode else pd.DataFrame()

    # Answer quality per system
    aq_records = []
    for mode in sorted({r.get("mode") for r in rows}):
        aq = compute_answer_quality([r for r in rows if r.get("mode") == mode])
        aq_records.append({"system": mode, **aq})
    answer_q = pd.DataFrame(aq_records).set_index("system") if aq_records else pd.DataFrame()

    # Hallucination by category for FULL (and all systems flattened)
    halluc_records = []
    for mode in sorted({r.get("mode") for r in rows}):
        h = compute_hallucination_by_category([r for r in rows if r.get("mode") == mode])
        for cat, m in h.items():
            halluc_records.append({"system": mode, "category": cat, **m})
    halluc = pd.DataFrame(halluc_records)

    configs = system_config_table(rows)
    sig = significance_tables(rows, reference=FULL)
    cis = confidence_interval_table(rows)

    # Confusion matrices
    for mode in sorted({r.get("mode") for r in rows}):
        metrics = compute_extended_metrics([r for r in rows if r.get("mode") == mode])
        cm_b = pd.DataFrame(
            metrics.get("confusion_matrix_binary", []),
            index=["gold_answerable", "gold_unanswerable"],
            columns=["pred_answered", "pred_not_answered"],
        )
        cm_d = pd.DataFrame(
            metrics.get("confusion_matrix_decision", []),
            index=[f"exp_{a}" for a in ACTIONS],
            columns=[f"pred_{a}" for a in ACTIONS],
        )
        _save_csv(cm_b, out_dir / f"confusion_binary_{mode}.csv")
        _save_csv(cm_d, out_dir / f"confusion_decision_{mode}.csv")

    _save_csv(summary, out_dir / "metrics_summary.csv")
    _save_csv(improvement, out_dir / "baseline_improvement.csv")
    _save_csv(ablation, out_dir / "ablation_comparison.csv")
    _save_csv(per_cat, out_dir / "per_category.csv", index=False)
    _save_csv(per_course, out_dir / "per_course.csv", index=False)
    _save_csv(retrieval, out_dir / "retrieval_metrics.csv")
    _save_csv(answer_q, out_dir / "answer_quality.csv")
    _save_csv(halluc, out_dir / "hallucination_by_category.csv", index=False)
    _save_csv(configs, out_dir / "system_configs.csv", index=False)
    _save_csv(sig, out_dir / "significance_tests.csv", index=False)
    _save_csv(cis, out_dir / "confidence_intervals.csv", index=False)

    seeds = sorted({r.get("seed") for r in rows})
    reasoning_mode = rows[0].get("reasoning_mode", "unknown")
    write_markdown(
        out_dir / "EVALUATION_REPORT.md",
        summary,
        improvement,
        ablation,
        per_cat,
        per_course,
        retrieval,
        answer_q,
        halluc,
        configs,
        sig,
        cis,
        len(rows),
        seeds,
        reasoning_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="results/predictions_all.jsonl")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    run(Path(args.predictions), Path(args.out_dir))


if __name__ == "__main__":
    main()
