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
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from eval.answer_quality import compute_answer_quality
from eval.baselines import (
    ABLATION_SYSTEMS,
    BASELINE_SYSTEMS,
    RETRIEVAL_PROFILES,
)
from eval.metrics_extended import (
    ACTIONS,
    compute_by_seed,
    compute_extended_metrics,
    compute_hallucination_by_category,
    compute_per_category,
    compute_per_course,
)
from eval.retrieval_metrics import compute_retrieval_by_mode
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


def _sorted_modes(rows: List[Dict[str, Any]]) -> List[str]:
    """Return non-null mode names as a sorted list."""
    modes: set[str] = set()

    for row in rows:
        mode = row.get("mode")
        if isinstance(mode, str):
            modes.add(mode)

    return sorted(modes)


def _sorted_ids(rows: List[Dict[str, Any]]) -> List[str]:
    """Return non-null prediction IDs as a sorted list."""
    ids: set[str] = set()

    for row in rows:
        item_id = row.get("id")
        if isinstance(item_id, str):
            ids.add(item_id)

    return sorted(ids)


def _sorted_seeds(rows: List[Dict[str, Any]]) -> List[int]:
    """Return integer seeds as a sorted list."""
    seeds: set[int] = set()

    for row in rows:
        seed = row.get("seed")
        if isinstance(seed, int):
            seeds.add(seed)

    return sorted(seeds)


def _save_csv(
    df: pd.DataFrame,
    path: Path,
    **kwargs: Any,
) -> None:
    """Save a DataFrame while gracefully handling locked files."""
    try:
        df.to_csv(path, **kwargs)
        print(f"[report] Saved {path}")
    except PermissionError:
        print(f"[report] WARNING: {path} locked; skipped.")


def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL prediction records."""
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                parsed = json.loads(line)

                if isinstance(parsed, dict):
                    rows.append(parsed)

    return rows


def summary_table(
    by_seed: Dict[str, Dict[int, Dict[str, Any]]],
) -> pd.DataFrame:
    """Build mean/std summary metrics across seeds."""
    records: List[Dict[str, Any]] = []

    for mode, seed_metrics in by_seed.items():
        rec: Dict[str, Any] = {
            "system": mode,
            "n_runs": len(seed_metrics),
        }

        for metric in SCALAR_METRICS:
            vals: List[float] = []

            for metrics in seed_metrics.values():
                value = metrics.get(metric)

                if isinstance(value, (int, float)) and value == value:
                    vals.append(float(value))

            rec[f"{metric}_mean"] = (
                round(float(np.mean(vals)), 4)
                if vals
                else float("nan")
            )

            rec[f"{metric}_std"] = (
                round(float(np.std(vals)), 4)
                if vals
                else float("nan")
            )

        records.append(rec)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("system")


def improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate full-vs-baseline percentage-point differences.

    Values are expressed as:
        (FULL - baseline) * 100

    For metrics where lower is better, the sign is inverted so that
    positive numbers consistently indicate an improvement for FULL.
    """
    records: List[Dict[str, Any]] = []

    if FULL not in summary.index:
        return pd.DataFrame()

    for baseline in BASELINES:
        if baseline not in summary.index:
            continue

        rec: Dict[str, Any] = {
            "baseline": baseline,
        }

        for metric in SCALAR_METRICS:
            base_value = summary.loc[
                baseline,
                f"{metric}_mean",
            ]
            full_value = summary.loc[
                FULL,
                f"{metric}_mean",
            ]

            if not isinstance(base_value, (int, float)):
                rec[metric] = float("nan")
                continue

            if not isinstance(full_value, (int, float)):
                rec[metric] = float("nan")
                continue

            if np.isnan(float(base_value)) or np.isnan(float(full_value)):
                rec[metric] = float("nan")
                continue

            delta_pp = (
                float(full_value) - float(base_value)
            ) * 100.0

            if metric in LOWER_IS_BETTER:
                delta_pp = -delta_pp

            rec[metric] = round(delta_pp, 2)

        records.append(rec)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("baseline")


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Build ablation results and deltas relative to FULL."""
    records: List[Dict[str, Any]] = []

    if FULL not in summary.index:
        return pd.DataFrame()

    configurations = [
        FULL,
        *[
            ablation
            for ablation in ABLATIONS
            if ablation in summary.index
        ],
    ]

    for name in configurations:
        rec: Dict[str, Any] = {
            "configuration": name,
        }

        for metric in SCALAR_METRICS:
            if name in summary.index:
                value = summary.loc[
                    name,
                    f"{metric}_mean",
                ]
            else:
                value = float("nan")

            full_value = summary.loc[
                FULL,
                f"{metric}_mean",
            ]

            rec[metric] = value

            if (
                isinstance(value, (int, float))
                and isinstance(full_value, (int, float))
                and not np.isnan(float(value))
                and not np.isnan(float(full_value))
            ):
                rec[f"{metric}_delta"] = round(
                    float(value) - float(full_value),
                    4,
                )
            else:
                rec[f"{metric}_delta"] = float("nan")

        records.append(rec)

    return pd.DataFrame(records).set_index("configuration")


def system_config_table(
    rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Build one configuration record per evaluated system."""
    records: List[Dict[str, Any]] = []
    modes = _sorted_modes(rows)

    for mode_name in modes:
        sample = next(
            row
            for row in rows
            if row.get("mode") == mode_name
        )

        retrieval_mode, reranker = RETRIEVAL_PROFILES.get(
            mode_name,
            ("?", False),
        )

        records.append(
            {
                "system": mode_name,
                "llm_model": sample.get("llm_model"),
                "embed_model": sample.get("embed_model"),
                "retrieval_mode": retrieval_mode,
                "reranker": reranker,
                "top_k_passages": sample.get("n_passages"),
                "rag_threshold": sample.get("rag_threshold"),
                "reasoning_mode": sample.get("reasoning_mode"),
                "seed_example": sample.get("seed"),
            }
        )

    return pd.DataFrame(records)


def significance_tables(
    rows: List[Dict[str, Any]],
    reference: str = FULL,
) -> pd.DataFrame:
    """Run paired McNemar tests against the reference system."""
    ref_rows = [
        row
        for row in rows
        if row.get("mode") == reference
    ]

    records: List[Dict[str, Any]] = []

    modes = [
        mode_name
        for mode_name in _sorted_modes(rows)
        if mode_name != reference
    ]

    for mode_name in modes:
        other_rows = [
            row
            for row in rows
            if row.get("mode") == mode_name
        ]

        for label_name, metric_fn in (
            (
                "binary_detection_correct",
                binary_detection_correct,
            ),
            (
                "expected_action_correct",
                decision_correct,
            ),
        ):
            y_a, y_b = paired_binary_labels(
                ref_rows,
                other_rows,
                metric_fn,
            )

            result = mcnemar_test(y_a, y_b)

            p_value = result.get("p_value")

            if isinstance(p_value, (int, float)):
                interpretation = interpret_significance(
                    float(p_value)
                )
            else:
                interpretation = "unknown"

            records.append(
                {
                    "reference": reference,
                    "system": mode_name,
                    "label": label_name,
                    "test": result.get("test"),
                    "n": result.get("n"),
                    "statistic": result.get("statistic"),
                    "p_value": p_value,
                    "effect_size": result.get("effect_size"),
                    "effect_size_name": result.get(
                        "effect_size_name"
                    ),
                    "interpretation": interpretation,
                }
            )

    return pd.DataFrame(records)


def confidence_interval_table(
    rows: List[Dict[str, Any]],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate bootstrap confidence intervals."""
    requested_metrics = metrics or [
        "accuracy",
        "f1",
        "coverage",
        "hallucination_rate",
    ]

    # Kept for API compatibility; the CI implementation currently
    # generates the standard four metrics above.
    _ = requested_metrics

    records: List[Dict[str, Any]] = []
    modes = _sorted_modes(rows)

    for mode_name in modes:
        mode_rows = [
            row
            for row in rows
            if row.get("mode") == mode_name
        ]

        indicators: Dict[str, List[int]] = {
            "accuracy": [
                int(binary_detection_correct(row))
                for row in mode_rows
            ],
            "coverage": [
                1 if row.get("action") == "ANSWER" else 0
                for row in mode_rows
            ],
            "hallucination_rate": [
                1
                if (
                    not row.get("gold_answerable", True)
                    and row.get("action") == "ANSWER"
                    and bool(
                        str(
                            row.get(
                                "final_answer",
                                "",
                            )
                        ).strip()
                    )
                )
                else 0
                for row in mode_rows
                if not row.get("gold_answerable", True)
            ],
        }

        for metric_name, values in indicators.items():
            ci = bootstrap_ci(values)

            records.append(
                {
                    "system": mode_name,
                    "metric": metric_name,
                    "mean": ci["mean"],
                    "ci_low": ci["ci_low"],
                    "ci_high": ci["ci_high"],
                    "n": ci["n"],
                    "method": ci["method"],
                }
            )

        f1_boot = _bootstrap_f1(mode_rows)

        records.append(
            {
                "system": mode_name,
                "metric": "f1",
                "mean": f1_boot["mean"],
                "ci_low": f1_boot["ci_low"],
                "ci_high": f1_boot["ci_high"],
                "n": f1_boot["n"],
                "method": "cluster_bootstrap_questions",
            }
        )

    return pd.DataFrame(records)


def _bootstrap_f1(
    rows: List[Dict[str, Any]],
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap F1 by question ID."""
    from sklearn.metrics import precision_recall_fscore_support

    ids = _sorted_ids(rows)

    by_id: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        item_id = row.get("id")

        if not isinstance(item_id, str):
            continue

        by_id.setdefault(item_id, []).append(row)

    if not ids:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }

    rng = np.random.default_rng(seed)
    f1_scores: List[float] = []

    for _ in range(n_boot):
        sampled_ids = rng.choice(
            ids,
            size=len(ids),
            replace=True,
        )

        sample: List[Dict[str, Any]] = []

        for item_id in sampled_ids:
            sample.extend(
                by_id[str(item_id)]
            )

        y_true = [
            0
            if row.get("gold_answerable", True)
            else 1
            for row in sample
        ]

        y_pred = [
            0
            if row.get("action") == "ANSWER"
            else 1
            for row in sample
        ]

        _, _, f1_value, _ = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                average="binary",
                zero_division=cast(Any, 0),
            )
        )

        f1_scores.append(float(f1_value))

    values = np.asarray(f1_scores)

    return {
        "mean": round(float(values.mean()), 4),
        "ci_low": round(
            float(np.quantile(values, 0.025)),
            4,
        ),
        "ci_high": round(
            float(np.quantile(values, 0.975)),
            4,
        ),
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
    seeds: List[int],
    reasoning_mode: str,
) -> None:
    """Write the consolidated Markdown report."""
    lines: List[str] = []

    lines.append(
        "# Evaluation Report — Policy-Aware Multi-Agent RAG\n"
    )

    lines.append(
        "Generated by `python -m eval.report`. "
        "FULL system = Multi-Agent + Hybrid Retrieval + "
        "Reranker + Answerability + Policy/Governance.\n"
    )

    lines.append(
        f"- Prediction rows: **{n_rows}**"
    )
    lines.append(
        f"- Seeds: `{seeds}`"
    )
    lines.append(
        f"- Reasoning backend: **{reasoning_mode}**\n"
    )

    lines.append("## Metric definitions\n")

    lines.append(
        "- **accuracy / precision / recall / f1** — "
        "binary unanswerable detection "
        "(positive = gold unanswerable; "
        "predicted positive = action ≠ ANSWER).\n"
    )

    lines.append(
        "- **roc_auc / pr_auc** — from continuous "
        "`unanswerable_score = 1 - answerability_confidence` "
        "(never from hard labels).\n"
    )

    lines.append(
        "- **coverage** — fraction of questions with "
        "action=ANSWER.\n"
    )

    lines.append(
        "- **abstention_precision** — "
        "P(gold unanswerable | ABSTAIN).\n"
    )

    lines.append(
        "- **abstention_recall** — "
        "P(ABSTAIN | gold unanswerable).\n"
    )

    lines.append(
        "- **hallucination_rate / unsupported_answer_rate** — "
        "gold-unanswerable questions that received a "
        "substantive ANSWER.\n"
    )

    lines.append(
        "- **policy_compliance_rate** — "
        "decision matches re-applied governance policy "
        "on recorded signals (not answer correctness).\n"
    )

    lines.append(
        "- **retrieval metrics** — binary relevance vs "
        "`evidence_ids` (nDCG binary).\n"
    )

    lines.append(
        "- **Agentic RAG** — honest single-agent proxy "
        "(reasoner self-governs); not an iterative "
        "tool-calling loop.\n"
    )

    def _add_table(
        title: str,
        df: pd.DataFrame,
    ) -> None:
        lines.append(f"## {title}\n")

        if df.empty:
            lines.append("_No data._\n")
            return

        lines.append("```")
        lines.append(df.to_string())
        lines.append("```\n")

    # ------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------
    dataset_df = pd.DataFrame(
        {
            "field": [
                "questions_file",
                "n_prediction_rows",
                "seeds",
            ],
            "value": [
                "data/eval/educational_v2_questions.jsonl",
                n_rows,
                str(seeds),
            ],
        }
    )

    _add_table(
        "1. Dataset",
        dataset_df,
    )

    # ------------------------------------------------------------
    # 2. Experimental configuration
    # ------------------------------------------------------------
    _add_table(
        "2. Experimental configuration",
        configs,
    )

    # ------------------------------------------------------------
    # 3. Baseline comparison
    # ------------------------------------------------------------
    baseline_indices = [
        index
        for index in summary.index
        if index in BASELINE_SYSTEMS
    ]

    baseline_view = cast(
        pd.DataFrame,
        summary.loc[
            baseline_indices,
            :,
        ],
    )

    _add_table(
        "3. Baseline comparison (mean over seeds)",
        baseline_view,
    )

    # ------------------------------------------------------------
    # 4. Answerability classification
    # ------------------------------------------------------------
    classification_columns = [
        column
        for column in summary.columns
        if any(
            metric in column
            for metric in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            )
        )
    ]

    classification_view = cast(
        pd.DataFrame,
        summary.loc[
            :,
            classification_columns,
        ],
    )

    _add_table(
        "4. Answerability classification",
        classification_view,
    )

    # ------------------------------------------------------------
    # 5. Abstention and coverage
    # ------------------------------------------------------------
    abstention_columns = [
        column
        for column in summary.columns
        if any(
            metric in column
            for metric in (
                "abstention",
                "coverage",
                "clarification",
            )
        )
    ]

    abstention_view = cast(
        pd.DataFrame,
        summary.loc[
            :,
            abstention_columns,
        ],
    )

    _add_table(
        "5. Abstention and coverage",
        abstention_view,
    )

    # ------------------------------------------------------------
    # 6. Retrieval quality
    # ------------------------------------------------------------
    _add_table(
        "6. Retrieval quality",
        retrieval,
    )

    # ------------------------------------------------------------
    # 7. Answer quality
    # ------------------------------------------------------------
    _add_table(
        "7. Answer quality (gold-answerable)",
        answer_q,
    )

    # ------------------------------------------------------------
    # 8. Hallucination
    # ------------------------------------------------------------
    _add_table(
        "8. Hallucination (per category, pooled)",
        halluc,
    )

    # ------------------------------------------------------------
    # 9. Policy evaluation
    # ------------------------------------------------------------
    policy_columns = [
        column
        for column in summary.columns
        if any(
            metric in column
            for metric in (
                "policy",
                "correct_answer",
                "correct_clarify",
                "correct_abstain",
                "false_acceptance",
                "false_rejection",
                "clarification_quality",
            )
        )
    ]

    policy_view = cast(
        pd.DataFrame,
        summary.loc[
            :,
            policy_columns,
        ],
    )

    _add_table(
        "9. Policy evaluation",
        policy_view,
    )

    # ------------------------------------------------------------
    # 10. Ablation
    # ------------------------------------------------------------
    _add_table(
        "10. Ablation study (delta vs FULL)",
        ablation,
    )

    # ------------------------------------------------------------
    # 11. Per-category
    # ------------------------------------------------------------
    _add_table(
        "11. Per-category analysis",
        per_cat,
    )

    # ------------------------------------------------------------
    # 12. Per-course
    # ------------------------------------------------------------
    _add_table(
        "12. Per-course analysis",
        per_course,
    )

    # ------------------------------------------------------------
    # 13. Statistical significance
    # ------------------------------------------------------------
    _add_table(
        "13. Statistical significance (McNemar vs FULL)",
        sig,
    )

    # ------------------------------------------------------------
    # 14. Confidence intervals
    # ------------------------------------------------------------
    _add_table(
        "14. Confidence intervals (bootstrap 95%)",
        cis,
    )

    # ------------------------------------------------------------
    # 15. Limitations
    # ------------------------------------------------------------
    lines.append("## 15. Limitations\n")

    lines.append(
        "- Corpus for educational_v2 is reconstructed from "
        "`evidence_passage` texts already in the JSONL "
        "(held-out passages excluded). Full Wikipedia article "
        "context beyond those evidence spans is not regenerated.\n"
    )

    lines.append(
        "- nDCG uses binary relevance "
        "(no graded labels).\n"
    )

    lines.append(
        "- Agentic RAG is a single-agent self-governance proxy, "
        "not iterative planning.\n"
    )

    lines.append(
        "- Lexical groundedness is a proxy, not NLI entailment; "
        "LLM-judge metrics are not treated as objective "
        "ground truth.\n"
    )

    lines.append(
        "- ROC/PR-AUC require continuous "
        "`answerability_confidence`; systems without meaningful "
        "variation may yield unstable AUC.\n"
    )

    lines.append(
        "- Statistical significance does not imply practical "
        "importance; effect sizes are reported alongside "
        "p-values.\n"
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"[report] Saved {path}")


def run(
    predictions_path: Path,
    out_dir: Path,
) -> None:
    """Run the complete reporting pipeline."""
    rows = load_predictions(predictions_path)

    if not rows:
        raise SystemExit(
            f"No predictions in {predictions_path}"
        )

    by_seed = compute_by_seed(rows)

    summary = summary_table(by_seed)
    improvement = improvement_table(summary)
    ablation = ablation_table(summary)

    # ------------------------------------------------------------
    # Per-category
    # ------------------------------------------------------------
    per_cat_raw = compute_per_category(rows)
    per_cat_records: List[Dict[str, Any]] = []

    for mode_name, categories in per_cat_raw.items():
        for category, metrics in categories.items():
            per_cat_records.append(
                {
                    "system": mode_name,
                    "category": category,
                    **metrics,
                }
            )

    per_cat = pd.DataFrame(per_cat_records)

    # ------------------------------------------------------------
    # Per-course
    # ------------------------------------------------------------
    per_course_raw = compute_per_course(rows)
    per_course_records: List[Dict[str, Any]] = []

    for mode_name, courses in per_course_raw.items():
        for course, metrics in courses.items():
            per_course_records.append(
                {
                    "system": mode_name,
                    "course": course,
                    **metrics,
                }
            )

    per_course = pd.DataFrame(per_course_records)

    # ------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------
    retrieval_by_mode = compute_retrieval_by_mode(
        rows,
        k=5,
    )

    if retrieval_by_mode:
        retrieval = pd.DataFrame(
            [
                {
                    "system": mode_name,
                    **metrics,
                }
                for mode_name, metrics
                in retrieval_by_mode.items()
            ]
        ).set_index("system")
    else:
        retrieval = pd.DataFrame()

    # ------------------------------------------------------------
    # Answer quality
    # ------------------------------------------------------------
    answer_quality_records: List[Dict[str, Any]] = []

    for mode_name in _sorted_modes(rows):
        mode_rows = [
            row
            for row in rows
            if row.get("mode") == mode_name
        ]

        answer_quality = compute_answer_quality(mode_rows)

        answer_quality_records.append(
            {
                "system": mode_name,
                **answer_quality,
            }
        )

    if answer_quality_records:
        answer_q = pd.DataFrame(
            answer_quality_records
        ).set_index("system")
    else:
        answer_q = pd.DataFrame()

    # ------------------------------------------------------------
    # Hallucination by category
    # ------------------------------------------------------------
    hallucination_records: List[Dict[str, Any]] = []

    for mode_name in _sorted_modes(rows):
        mode_rows = [
            row
            for row in rows
            if row.get("mode") == mode_name
        ]

        hallucination = compute_hallucination_by_category(
            mode_rows
        )

        for category, metrics in hallucination.items():
            hallucination_records.append(
                {
                    "system": mode_name,
                    "category": category,
                    **metrics,
                }
            )

    halluc = pd.DataFrame(
        hallucination_records
    )

    # ------------------------------------------------------------
    # System configs
    # ------------------------------------------------------------
    configs = system_config_table(rows)

    # ------------------------------------------------------------
    # Statistical significance
    # ------------------------------------------------------------
    sig = significance_tables(
        rows,
        reference=FULL,
    )

    # ------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------
    cis = confidence_interval_table(rows)

    # ------------------------------------------------------------
    # Confusion matrices
    # ------------------------------------------------------------
    for mode_name in _sorted_modes(rows):
        mode_rows = [
            row
            for row in rows
            if row.get("mode") == mode_name
        ]

        metrics = compute_extended_metrics(mode_rows)

        binary_cm = pd.DataFrame(
            metrics.get(
                "confusion_matrix_binary",
                [],
            ),
            index=[
                "gold_answerable",
                "gold_unanswerable",
            ],
            columns=[
                "pred_answered",
                "pred_not_answered",
            ],
        )

        decision_cm = pd.DataFrame(
            metrics.get(
                "confusion_matrix_decision",
                [],
            ),
            index=[
                f"exp_{action}"
                for action in ACTIONS
            ],
            columns=[
                f"pred_{action}"
                for action in ACTIONS
            ],
        )

        _save_csv(
            binary_cm,
            out_dir / (
                f"confusion_binary_{mode_name}.csv"
            ),
        )

        _save_csv(
            decision_cm,
            out_dir / (
                f"confusion_decision_{mode_name}.csv"
            ),
        )

    # ------------------------------------------------------------
    # Main CSV outputs
    # ------------------------------------------------------------
    _save_csv(
        summary,
        out_dir / "metrics_summary.csv",
    )

    _save_csv(
        improvement,
        out_dir / "baseline_improvement.csv",
    )

    _save_csv(
        ablation,
        out_dir / "ablation_comparison.csv",
    )

    _save_csv(
        per_cat,
        out_dir / "per_category.csv",
        index=False,
    )

    _save_csv(
        per_course,
        out_dir / "per_course.csv",
        index=False,
    )

    _save_csv(
        retrieval,
        out_dir / "retrieval_metrics.csv",
    )

    _save_csv(
        answer_q,
        out_dir / "answer_quality.csv",
    )

    _save_csv(
        halluc,
        out_dir / "hallucination_by_category.csv",
        index=False,
    )

    _save_csv(
        configs,
        out_dir / "system_configs.csv",
        index=False,
    )

    _save_csv(
        sig,
        out_dir / "significance_tests.csv",
        index=False,
    )

    _save_csv(
        cis,
        out_dir / "confidence_intervals.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Report metadata
    # ------------------------------------------------------------
    seeds = _sorted_seeds(rows)

    first_reasoning_mode = rows[0].get(
        "reasoning_mode",
        "unknown",
    )

    reasoning_mode = (
        first_reasoning_mode
        if isinstance(first_reasoning_mode, str)
        else "unknown"
    )

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
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--predictions",
        default="results/predictions_all.jsonl",
    )

    parser.add_argument(
        "--out-dir",
        default="results",
    )

    args = parser.parse_args()

    run(
        Path(args.predictions),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()