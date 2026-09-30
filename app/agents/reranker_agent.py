"""Lightweight cross-encoder reranker for retrieved passages.

Additive, configurable, and disable-able. Default model is a small MS MARCO
cross-encoder for reproducibility. When disabled, passages are returned
unchanged (preserving upstream hybrid/dense order).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from app.config import Config
from app.utils.logger import get_logger

logger = get_logger("reranker", "logs/reranker.log")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerAgent:
    """Cross-encoder reranker. Safe to construct with enabled=False (no model load)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        enabled: bool = True,
        top_k: Optional[int] = None,
    ):
        cfg = getattr(Config, "RERANKER", None) or {}
        self.enabled = bool(enabled if enabled is not None else cfg.get("enabled", True))
        self.model_name = model_name or cfg.get("model", DEFAULT_RERANKER_MODEL)
        self.default_top_k = int(top_k if top_k is not None else cfg.get("top_k", 5))
        self._model = None
        if self.enabled:
            self._load()

    def _load(self) -> None:
        # Lazy import keeps eval startup light when reranker is disabled.
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder reranker: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        passages: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """Return top_k passages re-ordered by cross-encoder score.

        When disabled or passages empty, returns the input list truncated to top_k
        with ``reranker`` metadata left unset / ``skipped``.
        """
        k = int(top_k if top_k is not None else self.default_top_k)
        if not passages:
            return []
        if not self.enabled or self._model is None:
            out = []
            for p in passages[:k]:
                row = dict(p)
                row["reranker"] = "skipped"
                out.append(row)
            return out

        pairs = [[query, str(p.get("text", "") or "")] for p in passages]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(passages, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        results = []
        for p, sc in ranked[:k]:
            row = dict(p)
            row["score"] = float(sc)
            row["reranker"] = self.model_name
            row["rerank_score"] = float(sc)
            results.append(row)
        logger.info("Reranked %d -> %d passages", len(passages), len(results))
        return results
