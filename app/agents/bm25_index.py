"""Sparse BM25 retrieval over the same passage collection as FAISS.

Additive component: does not modify dense retrieval. Used by the hybrid
retriever for reciprocal-rank fusion.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Okapi BM25 over an in-memory passage list (same meta as FAISS)."""

    def __init__(self, passages: Sequence[Dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.passages = list(passages)
        self._docs_tokens: List[List[str]] = [tokenize(p.get("text", "")) for p in self.passages]
        self._doc_len = [len(t) for t in self._docs_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        self._df: Dict[str, int] = {}
        for toks in self._docs_tokens:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1
        self._n = len(self._docs_tokens)

    def _idf(self, term: str) -> float:
        # Robertson–Walker IDF with +0.5 smoothing.
        df = self._df.get(term, 0)
        return max(0.0, ((self._n - df + 0.5) / (df + 0.5)))

    def score(self, query: str, doc_idx: int) -> float:
        q_terms = tokenize(query)
        if not q_terms or self._avgdl <= 0:
            return 0.0
        toks = self._docs_tokens[doc_idx]
        if not toks:
            return 0.0
        from collections import Counter

        tf = Counter(toks)
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = self._idf(term)
            freq = tf[term]
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
            score += idf * (freq * (self.k1 + 1.0)) / denom
        return float(score)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self._n == 0 or top_k <= 0:
            return []
        scored = [(i, self.score(query, i)) for i in range(self._n)]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, sc in scored[:top_k]:
            if sc <= 0:
                continue
            meta = self.passages[idx]
            results.append(
                {
                    "id": meta.get("id", idx),
                    "text": meta.get("text"),
                    "source": meta.get("source", ""),
                    "score": float(sc),
                    "retriever": "bm25",
                }
            )
        return results
