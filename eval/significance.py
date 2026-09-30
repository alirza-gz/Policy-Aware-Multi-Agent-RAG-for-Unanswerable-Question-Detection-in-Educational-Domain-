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
