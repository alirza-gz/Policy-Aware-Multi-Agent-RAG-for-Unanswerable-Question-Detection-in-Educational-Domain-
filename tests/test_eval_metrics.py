"""Unit tests for extended metrics, retrieval IR, significance, answer quality."""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.answer_quality import exact_match, token_f1, groundedness
from eval.metrics_extended import compute_extended_metrics, policy_expected_action
from eval.retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from eval.significance import bootstrap_ci, interpret_significance, mcnemar_test


def _row(**kwargs):
    base = {
        "gold_answerable": True,
        "expected_action": "ANSWER",
        "action": "ANSWER",
        "final_answer": "Rayleigh distribution",
        "gold_answers": ["Rayleigh distribution"],
        "raw_answer": "Rayleigh distribution",
        "answerability_confidence": 0.9,
        "reasoner_confidence": 0.9,
        "unanswerable_score": 0.1,
        "retriever_confidence": 0.8,
        "model_is_answerable": True,
        "model_needs_clarification": False,
        "category": "answerable",
    }
    base.update(kwargs)
    return base


class TestClassificationAndAUC:
    def test_perfect_separation_auc(self):
        rows = [
            _row(gold_answerable=True, unanswerable_score=0.1, action="ANSWER"),
            _row(gold_answerable=True, unanswerable_score=0.2, action="ANSWER"),
            _row(
                gold_answerable=False,
                expected_action="ABSTAIN",
                unanswerable_score=0.9,
                action="ABSTAIN",
                final_answer="",
                category="unanswerable",
            ),
            _row(
                gold_answerable=False,
                expected_action="ABSTAIN",
                unanswerable_score=0.8,
                action="ABSTAIN",
                final_answer="",
                category="unanswerable",
            ),
        ]
        m = compute_extended_metrics(rows)
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0
        assert m["coverage"] == 0.5

    def test_abstention_metrics(self):
        rows = [
            _row(gold_answerable=True, action="ANSWER"),
            _row(
                gold_answerable=False,
                expected_action="ABSTAIN",
                action="ABSTAIN",
                final_answer="",
                category="unanswerable",
            ),
            _row(
                gold_answerable=False,
                expected_action="ABSTAIN",
                action="ANSWER",
                final_answer="guess",
                category="unanswerable",
            ),
        ]
        m = compute_extended_metrics(rows)
        assert m["abstention_precision"] == 1.0
        assert m["hallucination_rate"] == 0.5
        assert m["correct_abstention_rate"] == 0.5


class TestPolicyPrecedence:
    def test_clarify_before_unanswerable(self):
        row = _row(
            model_needs_clarification=True,
            model_is_answerable=False,
            answerability_confidence=0.1,
        )
        assert policy_expected_action(row) == "CLARIFY"


class TestRetrievalMetrics:
    def test_recall_hit_mrr_ndcg(self):
        gold = {"edu2_x.txt#p0"}
        pred = ["edu2_x.txt#p1", "edu2_x.txt#p0", "edu2_x.txt#p2"]
        assert recall_at_k(gold, pred[:2]) == 1.0
        assert hit_at_k(gold, pred[:1]) == 0.0
        assert mrr(gold, pred) == pytest.approx(0.5)
        assert ndcg_at_k(gold, pred) > 0


class TestAnswerQuality:
    def test_em_and_f1(self):
        assert exact_match("Paris", ["Paris"]) == 1.0
        assert token_f1("the Rayleigh distribution", ["Rayleigh distribution"]) > 0.5
        assert groundedness("gradient descent", ["Gradient descent minimizes loss"]) > 0


class TestSignificance:
    def test_mcnemar_identical(self):
        y = np.array([1, 0, 1, 1, 0])
        res = mcnemar_test(y, y.copy())
        assert res["p_value"] == 1.0
        assert interpret_significance(res["p_value"]) == "not_statistically_significant"

    def test_bootstrap_ci_contains_mean(self):
        vals = [0.0, 1.0, 1.0, 0.0, 1.0]
        ci = bootstrap_ci(vals, n_boot=200, seed=0)
        assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
