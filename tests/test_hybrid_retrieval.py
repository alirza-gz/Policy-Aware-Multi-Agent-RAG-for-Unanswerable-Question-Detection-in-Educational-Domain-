"""Unit tests for hybrid retrieval, RRF fusion, BM25, and reranker disable path."""

from __future__ import annotations

import pytest

from app.agents.bm25_index import BM25Index, tokenize
from app.agents.hybrid_retriever import reciprocal_rank_fusion


PASSAGES = [
    {"id": "a#p0", "text": "Gradient descent minimizes a loss function iteratively.", "source": "a"},
    {"id": "a#p1", "text": "Random forests are ensembles of decision trees.", "source": "a"},
    {"id": "b#p0", "text": "Bayes theorem relates conditional probabilities.", "source": "b"},
    {"id": "b#p1", "text": "The capital of France is Paris.", "source": "b"},
]


class TestBM25:
    def test_tokenize(self):
        assert "gradient" in tokenize("Gradient Descent!")

    def test_relevant_ranks_higher(self):
        idx = BM25Index(PASSAGES)
        hits = idx.search("gradient descent loss", top_k=2)
        assert hits
        assert hits[0]["id"] == "a#p0"


class TestRRF:
    def test_fusion_prefers_consensus(self):
        dense = [
            {"id": "a#p0", "score": 0.9, "retriever": "dense", "text": "x"},
            {"id": "b#p0", "score": 0.8, "retriever": "dense", "text": "y"},
        ]
        sparse = [
            {"id": "b#p0", "score": 5.0, "retriever": "bm25", "text": "y"},
            {"id": "a#p1", "score": 4.0, "retriever": "bm25", "text": "z"},
        ]
        fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)
        ids = [d["id"] for d in fused]
        assert "b#p0" in ids
        assert fused[0]["retriever"] == "hybrid_rrf"


class TestRerankerDisabled:
    def test_skipped_preserves_order(self):
        from app.agents.reranker_agent import RerankerAgent

        rr = RerankerAgent(enabled=False)
        passages = [{"id": "1", "text": "aaa", "score": 0.5}, {"id": "2", "text": "bbb", "score": 0.4}]
        out = rr.rerank("q", passages, top_k=2)
        assert [p["id"] for p in out] == ["1", "2"]
        assert out[0]["reranker"] == "skipped"
