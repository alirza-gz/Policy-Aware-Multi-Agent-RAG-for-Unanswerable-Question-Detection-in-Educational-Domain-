"""Hybrid retrieval: dense FAISS + sparse BM25 with reciprocal-rank fusion.

Extends the existing dense RetrieverAgent without changing its public
``retrieve()`` contract. New method ``retrieve_hybrid()`` exposes dense,
sparse, fused, and (optionally) reranked results for evaluation logging.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from app.agents.bm25_index import BM25Index
from app.agents.retriever_agent import RetrieverAgent
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger("hybrid_retriever", "logs/retriever.log")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Dict]],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict]:
    """RRF fusion over multiple ranked lists (by passage id)."""
    scores: Dict[str, float] = {}
    payload: Dict[str, Dict] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            pid = str(doc.get("id", ""))
            if not pid:
                continue
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            # Prefer keeping the richest metadata seen so far.
            if pid not in payload:
                payload[pid] = dict(doc)
            else:
                # Preserve original modality scores under side keys.
                prev = payload[pid]
                if "dense_score" not in prev and doc.get("retriever") == "dense":
                    prev["dense_score"] = doc.get("score")
                if "bm25_score" not in prev and doc.get("retriever") == "bm25":
                    prev["bm25_score"] = doc.get("score")
    fused = []
    for pid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        row = dict(payload[pid])
        row["score"] = float(sc)
        row["retriever"] = "hybrid_rrf"
        row["rrf_score"] = float(sc)
        fused.append(row)
    return fused


class HybridRetrieverAgent(RetrieverAgent):
    """Dense + BM25 hybrid retriever with optional cross-encoder reranking."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        retrieval_mode: str = "hybrid",
        rrf_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        reranker=None,
    ):
        super().__init__(model_name=model_name or Config.EMBED_MODEL)
        cfg = getattr(Config, "RETRIEVAL", None) or {}
        self.retrieval_mode = (retrieval_mode or cfg.get("mode", "hybrid")).lower()
        self.rrf_k = int(rrf_k if rrf_k is not None else cfg.get("rrf_k", 60))
        self.candidate_k = int(
            candidate_k if candidate_k is not None else cfg.get("candidate_k", 20)
        )
        self.reranker = reranker
        self._bm25: Optional[BM25Index] = None
        if self.meta:
            self._bm25 = BM25Index(self.meta)

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is None:
            if not self.meta:
                raise RuntimeError("No passage meta loaded; cannot build BM25 index.")
            self._bm25 = BM25Index(self.meta)
        return self._bm25

    def retrieve_dense(self, query: str, top_k: int = 5) -> List[Dict]:
        results = super().retrieve(query, top_k=top_k)
        for r in results:
            r["retriever"] = "dense"
            r["dense_score"] = r.get("score")
        return results

    def retrieve_sparse(self, query: str, top_k: int = 5) -> List[Dict]:
        return self._ensure_bm25().search(query, top_k=top_k)

    def retrieve_fused(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """Return dense, sparse, and RRF-fused lists (pre-rerank)."""
        cand = max(self.candidate_k, top_k)
        dense = self.retrieve_dense(query, top_k=cand)
        sparse = self.retrieve_sparse(query, top_k=cand)
        fused = reciprocal_rank_fusion([dense, sparse], k=self.rrf_k, top_k=cand)
        return {"dense": dense, "sparse": sparse, "fused": fused}

    def retrieve_pipeline(
        self,
        query: str,
        top_k: int = 5,
        mode: Optional[str] = None,
        use_reranker: Optional[bool] = None,
    ) -> Dict:
        """Configurable retrieval pipeline used by the evaluation harness.

        Returns a dict with passages (final), dense/sparse/fused lists, and flags.
        Each final passage retains ``dense_score`` when available so downstream
        governance can use a calibrated confidence independent of RRF/CE scales.
        """
        mode = (mode or self.retrieval_mode or "hybrid").lower()
        do_rerank = (
            bool(use_reranker)
            if use_reranker is not None
            else (self.reranker is not None and getattr(self.reranker, "enabled", False))
        )

        dense: List[Dict] = []
        sparse: List[Dict] = []
        fused: List[Dict] = []

        if mode == "none":
            final: List[Dict] = []
        elif mode == "dense":
            dense = self.retrieve_dense(query, top_k=max(self.candidate_k, top_k))
            final = [dict(p) for p in dense[: max(self.candidate_k, top_k)]]
        elif mode == "bm25":
            sparse = self.retrieve_sparse(query, top_k=max(self.candidate_k, top_k))
            final = [dict(p) for p in sparse[: max(self.candidate_k, top_k)]]
        else:  # hybrid
            parts = self.retrieve_fused(query, top_k=top_k)
            dense, sparse, fused = parts["dense"], parts["sparse"], parts["fused"]
            # Annotate fused docs with dense_score from the dense list when known.
            dense_map = {str(p.get("id")): p for p in dense}
            annotated = []
            for p in fused:
                row = dict(p)
                d = dense_map.get(str(row.get("id")))
                if d is not None and d.get("dense_score") is not None:
                    row["dense_score"] = d["dense_score"]
                annotated.append(row)
            final = annotated

        reranked = False
        pre_rerank = [dict(p) for p in final]
        if do_rerank and final and self.reranker is not None:
            # Preserve dense_score across rerank.
            dense_map = {str(p.get("id")): p.get("dense_score") for p in final}
            final = self.reranker.rerank(query, final, top_k=max(top_k, 10))
            for p in final:
                ds = dense_map.get(str(p.get("id")))
                if ds is not None:
                    p["dense_score"] = ds
            reranked = True
        else:
            final = final[:max(top_k, 10)]
            for p in final:
                p.setdefault("reranker", "skipped")
        ranked_deep = [dict(p) for p in final]     # depth >= 10 for retrieval metrics
        final = final[:top_k]                      # what the reasoner actually sees

        # NOTE (eval fix): candidate lists are returned at FULL candidate depth so
        # Recall@10 / nDCG@10 can be computed independently of the final top_k.
        # Previously they were truncated to top_k, making Recall@10 undefinable.
        return {
            "passages": final,
            "dense": dense,
            "sparse": sparse,
            "fused": (fused if fused else final),
            "pre_rerank": pre_rerank,
            "rerank_changed_top1": (bool(reranked and pre_rerank and final
                                         and str(pre_rerank[0].get("id")) != str(final[0].get("id")))),
            "rerank_changed_order": (bool(reranked and
                                          [str(p.get("id")) for p in pre_rerank[:top_k]] !=
                                          [str(p.get("id")) for p in final])),
            "retrieval_mode": mode,
            "reranked": reranked,
            "ranked_deep": ranked_deep,
            "retrieved_ids": [str(p.get("id", "")) for p in final],
        }

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Back-compat: use configured hybrid/dense pipeline without forcing rerank."""
        # Prefer hybrid when this class is used, unless mode is dense.
        mode = self.retrieval_mode if self.retrieval_mode != "none" else "dense"
        out = self.retrieve_pipeline(query, top_k=top_k, mode=mode, use_reranker=False)
        return out["passages"]
