"""Independent retrieval evaluation: Dense | BM25 | Hybrid(RRF) | Hybrid + Reranker.

Input: ``retrieval_trace.jsonl`` -- one row per question with ranked passage-id lists at
depth >= 10 for each retrieval configuration and the gold ``evidence_ids``.

Relevance ground truth (binary; graded relevance is NOT available in this dataset):
  A passage is relevant iff its id is in the question's ``evidence_ids`` -- the passage
  from which the question was generated (eval/link_evidence_ids.py links by SHA-1 of the
  normalised passage text). Consequences, stated so they are not over-interpreted:
    * other passages that also contain the answer count as NON-relevant (recall is a
      lower bound on true retrieval quality);
    * questions without evidence_ids (unanswerable held-out, out-of-domain) have no
      relevant passage and are excluded from every retrieval metric;
    * nDCG uses binary gains (rel in {0,1}).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Set

import numpy as np

CONFIGS = ("dense", "bm25", "hybrid", "hybrid_rerank")


def _gold(row: Dict) -> Set[str]:
    return {str(x) for x in (row.get("gold_ids") or []) if x}


def recall_at_k(g: Set[str], pred: Sequence[str], k: int) -> float:
    return len(g & set(pred[:k])) / len(g)


def precision_at_k(g: Set[str], pred: Sequence[str], k: int) -> float:
    return len(g & set(pred[:k])) / k            # divide by K, not by len(pred)


def hit_at_k(g: Set[str], pred: Sequence[str], k: int) -> float:
    return 1.0 if g & set(pred[:k]) else 0.0


def reciprocal_rank(g: Set[str], pred: Sequence[str]) -> float:
    for i, p in enumerate(pred, 1):
        if p in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(g: Set[str], pred: Sequence[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, p in enumerate(pred[:k], 1) if p in g)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(g), k) + 1))
    return dcg / idcg if idcg else 0.0


def per_question_metrics(row: Dict, config: str) -> Dict[str, float]:
    g, pred = _gold(row), [str(x) for x in (row.get(f"{config}_ids") or [])]
    return {"recall@1": recall_at_k(g, pred, 1), "recall@5": recall_at_k(g, pred, 5),
            "recall@10": recall_at_k(g, pred, 10), "precision@5": precision_at_k(g, pred, 5),
            "hit@5": hit_at_k(g, pred, 5), "mrr": reciprocal_rank(g, pred),
            "ndcg@10": ndcg_at_k(g, pred, 10), "hit@1": hit_at_k(g, pred, 1)}


def evaluate_retrieval(trace: List[Dict]) -> Dict:
    labelled = [r for r in trace if _gold(r)]
    out: Dict = {"n_questions_total": len(trace), "n_with_gold": len(labelled),
                 "relevance": "binary; gold = evidence_ids (source passage); see module docstring",
                 "configs": {}}
    for cfg in CONFIGS:
        if not labelled or not any(r.get(f"{cfg}_ids") is not None for r in labelled):
            continue
        pq = [per_question_metrics(r, cfg) for r in labelled]
        out["configs"][cfg] = {k: round(float(np.mean([m[k] for m in pq])), 4) for k in pq[0]}
    # Reranker analysis (hybrid -> hybrid_rerank), paired per question.
    if "hybrid" in out["configs"] and "hybrid_rerank" in out["configs"]:
        h = [per_question_metrics(r, "hybrid") for r in labelled]
        rr = [per_question_metrics(r, "hybrid_rerank") for r in labelled]
        changed_top1 = sum(1 for r in labelled
                           if (r["hybrid_ids"][:1] != r["hybrid_rerank_ids"][:1]))
        changed_top5 = sum(1 for r in labelled
                           if (r["hybrid_ids"][:5] != r["hybrid_rerank_ids"][:5]))
        d = {k: round(float(np.mean([b[k] - a[k] for a, b in zip(h, rr)])), 4) for k in h[0]}
        out["reranker_effect"] = {
            "frac_top1_changed": round(changed_top1 / len(labelled), 4),
            "frac_top5_order_changed": round(changed_top5 / len(labelled), 4),
            "top1_improved": sum(1 for a, b in zip(h, rr) if b["hit@1"] > a["hit@1"]),
            "top1_degraded": sum(1 for a, b in zip(h, rr) if b["hit@1"] < a["hit@1"]),
            "mean_delta_rerank_minus_hybrid": d,
            "note": "Recall@10 can only change if the reranker keeps fewer/more of the "
                    "candidate pool inside the top-10; check recall@10 delta before claiming a recall gain.",
        }
    return out
