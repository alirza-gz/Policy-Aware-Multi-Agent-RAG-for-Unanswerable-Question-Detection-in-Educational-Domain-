"""Statistical significance utilities for paired system comparisons.

- McNemar's test for paired binary classification decisions
- Bootstrap 95% confidence intervals for scalar metrics
- Effect size: difference in proportions / metric means

Does NOT default to an independent-samples t-test.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


def _question_key(row: Dict) -> str:
    return str(row.get("id", "")) + "::" + str(row.get("seed", ""))


def paired_binary_labels(
    rows_a: List[Dict],
    rows_b: List[Dict],
    label_fn: Callable[[Dict], int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two systems on (id, seed) and return paired binary arrays."""
    map_b = {_question_key(r): r for r in rows_b}
    ya, yb = [], []
    for r in rows_a:
        key = _question_key(r)
        if key not in map_b:
            continue
        ya.append(label_fn(r))
        yb.append(label_fn(map_b[key]))
    return np.asarray(ya, dtype=int), np.asarray(yb, dtype=int)


def mcnemar_test(y_a: np.ndarray, y_b: np.ndarray, continuity_correction: bool = True) -> Dict:
    """McNemar mid-p / chi-square with continuity correction on disagreement cells.

    y_* are binary correctness indicators (1 = correct decision).
    """
    if len(y_a) != len(y_b) or len(y_a) == 0:
        return {"test": "mcnemar", "n": 0, "p_value": float("nan"), "statistic": float("nan")}

    # Contingency on correctness: b = A wrong B right, c = A right B wrong.
    a_right = y_a.astype(bool)
    b_right = y_b.astype(bool)
    b = int((~a_right & b_right).sum())  # A0 B1
    c = int((a_right & ~b_right).sum())  # A1 B0
    n_discordant = b + c
    if n_discordant == 0:
        return {
            "test": "mcnemar",
            "n": int(len(y_a)),
            "b": b,
            "c": c,
            "statistic": 0.0,
            "p_value": 1.0,
            "effect_size": 0.0,
            "effect_size_name": "proportion_difference",
            "note": "no discordant pairs",
        }

    if continuity_correction:
        stat = (abs(b - c) - 1) ** 2 / n_discordant
    else:
        stat = (b - c) ** 2 / n_discordant
    # Chi-square with 1 df -> survival function
    # P(X^2 >= stat) = 2 * (1 - Phi(sqrt(stat))) for 1 df? Better use chi2 sf.
    from scipy.stats import chi2

    p = float(chi2.sf(stat, df=1))
    # Effect size: (correct_B - correct_A) / n
    effect = float(b_right.mean() - a_right.mean())
    return {
        "test": "mcnemar",
        "n": int(len(y_a)),
        "b": b,
        "c": c,
        "statistic": round(float(stat), 4),
        "p_value": float(p),
        "effect_size": round(effect, 4),
        "effect_size_name": "proportion_difference",
        "continuity_correction": continuity_correction,
    }


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Percentile bootstrap CI for the mean of ``values``."""
    arr = np.asarray([v for v in values if v == v], dtype=float)  # drop NaN
    if len(arr) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    means = np.asarray(means)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "n": int(len(arr)),
        "n_boot": n_boot,
        "alpha": alpha,
        "method": "percentile_bootstrap",
    }


def interpret_significance(p_value: float, alpha: float = 0.05) -> str:
    if p_value != p_value:
        return "not_applicable"
    if p_value < alpha:
        return "statistically_significant"
    return "not_statistically_significant"


def decision_correct(row: Dict) -> int:
    """1 if predicted action matches gold expected_action."""
    return int(row.get("action") == row.get("expected_action"))


def binary_detection_correct(row: Dict) -> int:
    """1 if binary unanswerable detection matches gold_answerable."""
    gold_unans = 0 if row.get("gold_answerable", True) else 1
    pred_unans = 0 if row.get("action") == "ANSWER" else 1
    return int(gold_unans == pred_unans)


# ===========================================================================
# Question-level (cluster-correct) inference -- added in the evaluation refactor
# ===========================================================================
# The original pipeline paired on (id, seed) and pooled seeds, giving n = seeds * N.
# Seeds only re-order questions (the LLM seed was fixed at 42), so the same question
# re-evaluated three times is NOT three independent observations; pooling inflates n
# and understates p-values / CI widths. The functions below treat the QUESTION as
# the unit of analysis.

import math as _math
from collections import defaultdict as _dd


def per_question_correct(rows: List[Dict], label_fn: Callable[[Dict], int]) -> Dict[str, float]:
    """question id -> mean correctness over seeds (in [0, 1])."""
    acc: Dict[str, List[int]] = _dd(list)
    for r in rows:
        acc[str(r.get("id"))].append(int(label_fn(r)))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value on discordant counts."""
    from scipy.stats import binomtest
    n = b + c
    return 1.0 if n == 0 else float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def cohens_h(p1: float, p2: float) -> float:
    return 2 * _math.asin(_math.sqrt(p1)) - 2 * _math.asin(_math.sqrt(p2))


def practical_label(h: float) -> str:
    a = abs(h)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def compare_systems_by_question(rows_a: List[Dict], rows_b: List[Dict],
                                label_fn: Callable[[Dict], int] = decision_correct,
                                n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Dict:
    """Paired comparison of two systems with the QUESTION as the unit.

    * per-question correctness = mean over seeds; binarised by majority (>=0.5) for McNemar
    * exact McNemar p on discordant questions
    * effect: difference in accuracy (B - A), Cohen's h, odds ratio b/c
    * 95% CI: paired percentile bootstrap resampling questions
    """
    pa, pb = per_question_correct(rows_a, label_fn), per_question_correct(rows_b, label_fn)
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return {"n_questions": 0, "p_value": float("nan")}
    va, vb = np.array([pa[i] for i in ids]), np.array([pb[i] for i in ids])
    ba, bb = va >= 0.5, vb >= 0.5
    b_cnt, c_cnt = int((~ba & bb).sum()), int((ba & ~bb).sum())
    p = exact_mcnemar(b_cnt, c_cnt)
    diff = float(vb.mean() - va.mean())
    h = cohens_h(float(vb.mean()), float(va.mean()))
    rng = np.random.default_rng(seed)
    d = vb - va
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
    sig = p < alpha
    prac = practical_label(h)
    if sig and prac == "negligible":
        note = "The difference is statistically significant, but its practical effect is limited."
    elif sig:
        note = f"Statistically significant with a {prac} practical effect."
    else:
        note = "Not statistically significant."
    return {
        "unit": "question", "n_questions": len(ids), "acc_a": round(float(va.mean()), 4),
        "acc_b": round(float(vb.mean()), 4), "diff_b_minus_a": round(diff, 4),
        "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "b_A_wrong_B_right": b_cnt,
        "c_A_right_B_wrong": c_cnt, "odds_ratio": round(b_cnt / c_cnt, 4) if c_cnt else None,
        "p_value": p, "test": "exact_mcnemar", "cohens_h": round(h, 4),
        "practical_effect": prac, "statistically_significant": bool(sig), "interpretation": note,
    }


def question_level_mean_std_ci(rows: List[Dict], metric_fn: Callable[[Dict], float],
                               n_boot: int = 2000, seed: int = 42) -> Dict:
    """Mean, std (across questions), and bootstrap 95% CI resampling questions."""
    pq = per_question_correct(rows, metric_fn)
    arr = np.array(list(pq.values()), float)
    if not len(arr):
        return {"n_questions": 0}
    rng = np.random.default_rng(seed)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)])
    return {"unit": "question", "n_questions": int(len(arr)), "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0,
            "ci95_low": round(float(np.quantile(means, 0.025)), 4),
            "ci95_high": round(float(np.quantile(means, 0.975)), 4)}
