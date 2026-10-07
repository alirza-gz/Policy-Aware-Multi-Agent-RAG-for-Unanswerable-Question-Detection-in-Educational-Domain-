"""Calibration + validation-only threshold selection.

Leakage rules enforced here:
  * ``grouped_split`` assigns whole groups (source document) to validation or test,
    so near-duplicate questions about the same passage never straddle the split.
  * Calibrators / thresholds are fit ONLY on validation rows; test rows are only
    scored. ``select_thresholds`` takes validation rows exclusively.
  * Seeds are NOT independent samples here (see eval.stats); callers must pass one
    row per question id.

Positive class convention: y = 1 means "fully answerable" for answerability
score calibration (score = P(fully answerable)).
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score


def grouped_split(rows: Sequence[Dict], val_fraction: float = 0.3, seed: int = 42,
                  group_key: str = "source_document") -> Tuple[List[int], List[int]]:
    """Deterministic group-wise split -> (val_idx, test_idx). Groups never straddle."""
    val, test = [], []
    for i, r in enumerate(rows):
        g = str(r.get(group_key) or r.get("id"))
        h = int(hashlib.sha1(f"{seed}:{g}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (val if h < val_fraction else test).append(i)
    return val, test


def expected_calibration_error(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_curve(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> List[Dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        out.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                    "mean_confidence": float(p[m].mean()) if m.any() else None,
                    "empirical_accuracy": float(y[m].mean()) if m.any() else None})
    return out


def calibration_report(y: Sequence[int], p: Sequence[float], n_bins: int = 10) -> Dict:
    y = list(map(int, y))
    p = [float(min(1.0, max(0.0, v))) for v in p]
    both = len(set(y)) == 2
    return {
        "n": len(y),
        "ece": round(expected_calibration_error(y, p, n_bins), 4),
        "brier": round(float(brier_score_loss(y, p)), 4) if y else float("nan"),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if both else float("nan"),
        "pr_auc": round(float(average_precision_score(y, p)), 4) if both else float("nan"),
        "reliability": reliability_curve(y, p, n_bins),
    }


# ---- calibrators (fit on validation only) ---------------------------------
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattScaler:
    def fit(self, scores, y):
        self.m = LogisticRegression(C=1e6, solver="lbfgs").fit(np.asarray(scores, float).reshape(-1, 1), y)
        return self

    def predict(self, scores):
        return self.m.predict_proba(np.asarray(scores, float).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, scores, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
        return self

    def predict(self, scores):
        return self.m.predict(np.asarray(scores, float))


class TemperatureScaler:
    """p' = sigmoid(logit(p)/T); T fit by minimising NLL on validation."""

    def fit(self, scores, y):
        z, y = _logit(scores), np.asarray(y, float)
        best_t, best = 1.0, float("inf")
        for t in np.exp(np.linspace(math.log(0.05), math.log(20), 200)):
            q = np.clip(1 / (1 + np.exp(-z / t)), 1e-9, 1 - 1e-9)
            nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
            if nll < best:
                best, best_t = nll, float(t)
        self.T = best_t
        return self

    def predict(self, scores):
        return 1 / (1 + np.exp(-_logit(scores) / self.T))


CALIBRATORS = {"platt": PlattScaler, "isotonic": IsotonicCalibrator, "temperature": TemperatureScaler}


def compare_calibrators(val_scores, val_y, test_scores, test_y) -> Dict[str, Dict]:
    """Fit each calibrator on VALIDATION, report on TEST. 'raw' = uncalibrated."""
    out = {"raw": calibration_report(test_y, test_scores)}
    for name, cls in CALIBRATORS.items():
        try:
            cal = cls().fit(val_scores, val_y)
            out[name] = calibration_report(test_y, cal.predict(test_scores))
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
    return out


# ---- score model + thresholds (validation only) ---------------------------
def fit_score_model(val_rows: Sequence[Dict], feature_names: Sequence[str]):
    """Fit logistic ScoreModel on validation features -> P(fully answerable)."""
    from app.agents.answerability_agent import ScoreModel

    X = np.asarray([[float(r["features"][f]) for f in feature_names] for r in val_rows], float)
    y = np.asarray([int(r["gold_fully_answerable"]) for r in val_rows])
    if len(set(y)) < 2:
        raise ValueError("validation set needs both classes to fit the score model")
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    m = LogisticRegression(C=1.0, max_iter=1000).fit((X - mean) / scale, y)
    return ScoreModel(list(feature_names), m.coef_[0].tolist(), float(m.intercept_[0]),
                      mean.tolist(), scale.tolist())


def select_thresholds(val_scores: Sequence[float], val_y: Sequence[int],
                      max_false_accept: Optional[float] = 0.10) -> Dict:
    """Pick the answer threshold on VALIDATION rows.

    Objective: maximise F1 of detecting NOT-fully-answerable (y=0) questions
    subject to a cap on false-accept rate (unanswerable answered). If the cap is
    infeasible the F1-optimal threshold is returned and ``cap_met`` is False.
    Never accepts test rows: callers must pass validation rows only.
    """
    s, y = np.asarray(val_scores, float), np.asarray(val_y, int)
    cand = np.unique(np.concatenate([s, [0.0, 1.0]]))
    best = None
    for t in cand:
        pred_answer = s >= t
        neg = y == 0
        fa = (pred_answer & neg).sum() / max(1, neg.sum())
        tp = ((~pred_answer) & neg).sum()
        fp = ((~pred_answer) & ~neg).sum()
        fn = (pred_answer & neg).sum()
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        ok = (max_false_accept is None) or fa <= max_false_accept
        key = (ok, f1)
        if best is None or key > best[0]:
            best = (key, float(t), float(fa), float(f1))
    return {"threshold": best[1], "val_false_accept_rate": round(best[2], 4),
            "val_f1_not_answerable": round(best[3], 4), "cap_met": bool(best[0][0]),
            "max_false_accept": max_false_accept, "fit_on": "validation_only"}
