"""Retrieval IR metrics against gold evidence_ids (binary relevance).

Uses passage IDs stored on prediction rows (``retrieved_ids``) and gold
``evidence_ids`` from the educational dataset. Graded relevance is unavailable,
so nDCG@K uses binary gains.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set


def _gold_set(row: Dict) -> Set[str]:
    ids = row.get("evidence_ids") or []
    if not ids and row.get("evidence_id"):
        ids = [row["evidence_id"]]
    return {str(x) for x in ids if x}


def _pred_list(row: Dict, k: Optional[int] = None) -> List[str]:
    ids = [str(x) for x in (row.get("retrieved_ids") or []) if x]
    return ids[:k] if k is not None else ids


def recall_at_k(gold: Set[str], pred: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    if not pred:
        return 0.0
    return len(gold & set(pred)) / len(gold)


def precision_at_k(gold: Set[str], pred: Sequence[str]) -> float:
    if not pred:
        return 0.0 if gold else float("nan")
    return len(gold & set(pred)) / len(pred)


def hit_at_k(gold: Set[str], pred: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    return 1.0 if gold & set(pred) else 0.0


def mrr(gold: Set[str], pred: Sequence[str]) -> float:
    if not gold:
        return float("nan")
    for i, pid in enumerate(pred, start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold: Set[str], pred: Sequence[str]) -> float:
    """Binary-relevance nDCG@K."""
    if not gold:
        return float("nan")
    if not pred:
        return 0.0
    dcg = 0.0
    for i, pid in enumerate(pred, start=1):
        rel = 1.0 if pid in gold else 0.0
        if rel:
            dcg += rel / math.log2(i + 1)
    ideal_hits = min(len(gold), len(pred))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def _mean_ignore_nan(vals: List[float]) -> float:
    clean = [v for v in vals if v == v]  # NaN != NaN
    return float(sum(clean) / len(clean)) if clean else float("nan")


def compute_retrieval_metrics(rows: List[Dict], k: int = 5) -> Dict:
    """Aggregate retrieval metrics over rows that have gold evidence_ids.

    Rows without gold evidence (OOD, unanswerable held-out) are excluded from
    the denominator — they are not retrieval failures of a labeled relevant set.
    """
    metrics = {
        "n_with_gold": 0,
        f"recall@{k}": [],
        f"precision@{k}": [],
        f"hit@{k}": [],
        "mrr": [],
        f"ndcg@{k}": [],
    }
    for r in rows:
        gold = _gold_set(r)
        if not gold:
            continue
        pred = _pred_list(r, k=k)
        metrics["n_with_gold"] += 1
        metrics[f"recall@{k}"].append(recall_at_k(gold, pred))
        metrics[f"precision@{k}"].append(precision_at_k(gold, pred))
        metrics[f"hit@{k}"].append(hit_at_k(gold, pred))
        metrics["mrr"].append(mrr(gold, pred))
        metrics[f"ndcg@{k}"].append(ndcg_at_k(gold, pred))

    out = {"n_with_gold": metrics["n_with_gold"], "k": k}
    for key in (f"recall@{k}", f"precision@{k}", f"hit@{k}", "mrr", f"ndcg@{k}"):
        out[key] = round(_mean_ignore_nan(metrics[key]), 4)
    return out


def compute_retrieval_by_mode(rows: List[Dict], k: int = 5) -> Dict[str, Dict]:
    modes = sorted({r.get("mode", "unknown") for r in rows})
    return {m: compute_retrieval_metrics([r for r in rows if r.get("mode") == m], k=k) for m in modes}


def compute_retrieval_by_course(rows: List[Dict], k: int = 5) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    modes = sorted({r.get("mode", "unknown") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        courses = sorted({str(r.get("course") or "unknown") for r in mode_rows})
        result[m] = {
            c: compute_retrieval_metrics([r for r in mode_rows if str(r.get("course") or "unknown") == c], k=k)
            for c in courses
        }
    return result


def compute_retrieval_by_category(rows: List[Dict], k: int = 5) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    modes = sorted({r.get("mode", "unknown") for r in rows})
    for m in modes:
        mode_rows = [r for r in rows if r.get("mode") == m]
        cats = sorted({str(r.get("category") or "unknown") for r in mode_rows})
        result[m] = {
            c: compute_retrieval_metrics([r for r in mode_rows if str(r.get("category") or "unknown") == c], k=k)
            for c in cats
        }
    return result
