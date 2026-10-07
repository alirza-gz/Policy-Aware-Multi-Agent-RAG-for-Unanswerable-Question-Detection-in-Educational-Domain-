"""Unit tests: calibration/thresholds, hallucination, groundedness, clarification, retrieval, statistics."""
import math

import numpy as np
import pytest

from eval.calibration import (TemperatureScaler, calibration_report, compare_calibrators,
                              expected_calibration_error, grouped_split, select_thresholds)
from eval.clarification_metrics import compute_clarification_metrics, compute_partial_answerable_eval
from eval.hallucination import (compute_hallucination_metrics, lexical_groundedness, semantic_groundedness)
from eval.retrieval_eval import evaluate_retrieval, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from eval.significance import compare_systems_by_question, decision_correct, exact_mcnemar
from app.agents.hybrid_retriever import reciprocal_rank_fusion


# ---------------- calibration / thresholds ----------------
def test_ece_zero_for_perfectly_calibrated_and_positive_when_overconfident():
    y = [1, 1, 0, 0] * 25
    assert expected_calibration_error(y, [0.5] * 100) == pytest.approx(0.0)
    assert expected_calibration_error(y, [0.99] * 100) > 0.4


def test_brier_and_report_keys():
    r = calibration_report([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert r["brier"] == pytest.approx(0.025) and len(r["reliability"]) == 10 and r["roc_auc"] == 1.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 2000); y = (rng.uniform(size=2000) < 0.5 + 0.2 * (p - 0.5)).astype(int)
    t = TemperatureScaler().fit(p, y)
    assert t.T > 1 and expected_calibration_error(y, t.predict(p)) < expected_calibration_error(y, p)


def test_compare_calibrators_fit_on_val_report_on_test():
    rng = np.random.default_rng(1)
    s = rng.uniform(size=400); y = (rng.uniform(size=400) < s).astype(int)
    out = compare_calibrators(s[:200], y[:200], s[200:], y[200:])
    assert set(out) == {"raw", "platt", "isotonic", "temperature"} and out["raw"]["n"] == 200


def test_threshold_uses_only_rows_passed_and_respects_cap():
    val_s = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]; val_y = [0, 0, 0, 1, 1, 1]
    t = select_thresholds(val_s, val_y, max_false_accept=0.0)
    assert 0.3 < t["threshold"] <= 0.8 and t["cap_met"] and t["fit_on"] == "validation_only"
    # infeasible cap -> flagged, not silently ignored
    t2 = select_thresholds([0.5, 0.5], [0, 1], max_false_accept=0.0)
    assert t2["cap_met"] and t2["threshold"] > 0.5      # only "answer nothing" meets a 0% cap


def test_grouped_split_has_no_group_leakage_and_is_deterministic():
    rows = [{"id": i, "source_document": f"doc{i % 17}"} for i in range(300)]
    v, t = grouped_split(rows, 0.3, seed=7)
    assert not ({rows[i]["source_document"] for i in v} & {rows[i]["source_document"] for i in t})
    assert (v, t) == grouped_split(rows, 0.3, seed=7) and len(v) + len(t) == 300


# ---------------- hallucination / groundedness ----------------
def R(action, ga, gold=("x",), ans="x", passages=("x is y",), cat="answerable"):
    return {"action": action, "gold_answerable": ga, "gold_answers": list(gold) if ga else [],
            "final_answer": ans, "passage_texts": list(passages), "category": cat}


def test_hallucination_metrics_are_separate_quantities():
    rows = [R("ANSWER", False, ans="made up thing", passages=("zzz",)),          # unanswerable answered (+unsupported)
            R("ANSWER", True, gold=("mean",), ans="the median value", passages=("the median value",)),  # wrong
            R("ANSWER", True, gold=("mean",), ans="the mean", passages=("the mean is",)),               # fine
            R("ABSTAIN", False, ans="")]
    m = compute_hallucination_metrics(rows)
    assert m["unanswerable_answer_rate"] == 0.5 and m["wrong_answer_rate"] == 0.5
    assert m["unsupported_answer_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert m["false_acceptance_rate"] == m["unanswerable_answer_rate"]
    assert m["contradiction_rate"] is None and "not_computed" in m["contradiction_method"]
    assert m["hallucination_rate_any"] == pytest.approx(2 / 3, abs=1e-3)


def test_contradiction_and_nli_support_with_injected_scorers():
    rows = [R("ANSWER", True, gold=("mean",), ans="the mean", passages=("the mean is",))]
    m = compute_hallucination_metrics(rows, entail_prob=lambda p, h: 0.9, contradict_prob=lambda p, h: 0.8)
    assert m["contradiction_rate"] == 1.0 and m["support_method"] == "nli" and m["unsupported_answer_rate"] == 0.0


def test_lexical_groundedness_is_overlap_not_entailment():
    assert lexical_groundedness("the sky is green", ["the sky is blue"]) == 0.75     # contradicts yet scores high
    assert lexical_groundedness("", ["a"]) == 0.0 and lexical_groundedness("a", []) == 0.0
    assert semantic_groundedness("claim", ["p1", "p2"], lambda p, h: 0.2 if p == "p1" else 0.7) == 0.7


# ---------------- clarification ----------------
def test_clarification_metrics_and_partial_eval():
    rows = [{"action": "CLARIFY", "expected_action": "CLARIFY", "category": "partially_answerable",
             "clarification_question": "The materials support the first part but not 'speech recognition history'. Could you rephrase it?",
             "missing_information": "speech recognition history", "out_of_scope_part_question": "history of speech recognition",
             "answerability_label": "partially_answerable"},
            {"action": "ANSWER", "expected_action": "CLARIFY", "category": "partially_answerable"},
            {"action": "CLARIFY", "expected_action": "ANSWER", "category": "answerable",
             "clarification_question": "Could you clarify?"},
            {"action": "ANSWER", "expected_action": "ANSWER", "category": "answerable"}]
    m = compute_clarification_metrics(rows)
    assert m["clarification_precision"] == 0.5 and m["clarification_recall"] == 0.5 and m["clarification_f1"] == 0.5
    assert m["clarification_rate"] == 0.5 and m["correct_clarification_rate"] == 0.25
    p = compute_partial_answerable_eval(rows)
    assert p["n"] == 2 and p["partially_answerable_action_accuracy"] == 0.5 and p["answer_rate_on_partial"] == 0.5


def test_generic_clarification_scores_lower_than_specific():
    from eval.clarification_metrics import clarification_quality_row as q
    g = q({"action": "CLARIFY", "clarification_question": "Could you clarify?"})
    s = q({"action": "CLARIFY", "missing_information": "x",
           "clarification_question": "The course materials do not cover deep learning history; could you provide the source?"})
    assert s > g and math.isnan(q({"action": "ANSWER"}))


# ---------------- retrieval + fusion ----------------
def test_ranking_metrics_known_values():
    g = {"a"}
    assert recall_at_k(g, ["b", "a"], 1) == 0 and recall_at_k(g, ["b", "a"], 5) == 1
    assert precision_at_k(g, ["a", "b"], 5) == 0.2 and reciprocal_rank(g, ["b", "c", "a"]) == pytest.approx(1 / 3)
    assert ndcg_at_k({"a", "b"}, ["a", "b"], 10) == pytest.approx(1.0)
    assert ndcg_at_k(g, ["b", "a"], 10) == pytest.approx(1 / math.log2(3))


def test_evaluate_retrieval_excludes_rows_without_gold_and_reports_reranker_effect():
    trace = [{"id": 1, "gold_ids": ["p1"], "dense_ids": ["p2", "p1"], "bm25_ids": ["p1"], "hybrid_ids": ["p2", "p1"], "hybrid_rerank_ids": ["p1", "p2"]},
             {"id": 2, "gold_ids": [], "dense_ids": ["p9"], "bm25_ids": [], "hybrid_ids": [], "hybrid_rerank_ids": []}]
    r = evaluate_retrieval(trace)
    assert r["n_with_gold"] == 1 and r["configs"]["dense"]["mrr"] == 0.5 and r["configs"]["hybrid_rerank"]["hit@1"] == 1.0
    assert r["reranker_effect"]["frac_top1_changed"] == 1.0 and r["reranker_effect"]["top1_improved"] == 1


def test_rrf_fusion_prefers_docs_in_both_lists_and_uses_rrf_formula():
    dense = [{"id": "a", "score": .9, "retriever": "dense"}, {"id": "b", "score": .8, "retriever": "dense"}]
    sparse = [{"id": "c", "score": 9, "retriever": "bm25"}, {"id": "b", "score": 8, "retriever": "bm25"}]
    f = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)
    assert f[0]["id"] == "b" and f[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 62 + 0)
    assert f[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 62)  # rank 2 in both lists
    assert {x["id"] for x in f} == {"a", "b", "c"}


# ---------------- statistics ----------------
def _rows(mode, correct_pattern, seeds=(1,)):
    return [{"id": f"q{i}", "seed": s, "mode": mode, "action": "ANSWER" if ok else "ABSTAIN", "expected_action": "ANSWER"}
            for s in seeds for i, ok in enumerate(correct_pattern)]


def test_question_level_inference_is_not_inflated_by_seed_pooling():
    pat_a = [1] * 60 + [0] * 40; pat_b = [1] * 70 + [0] * 30
    one = compare_systems_by_question(_rows("a", pat_a), _rows("b", pat_b), decision_correct)
    three = compare_systems_by_question(_rows("a", pat_a, (1, 2, 3)), _rows("b", pat_b, (1, 2, 3)), decision_correct)
    assert one["n_questions"] == three["n_questions"] == 100
    assert one["p_value"] == pytest.approx(three["p_value"]) and one["unit"] == "question"


def test_exact_mcnemar_and_practical_vs_statistical_significance():
    assert exact_mcnemar(0, 0) == 1.0 and exact_mcnemar(10, 10) == 1.0 and exact_mcnemar(0, 20) < 1e-4
    n = 20000
    a = [1] * (n - 40) + [0] * 40
    b = [1] * (n - 10) + [0] * 10                # +0.15 pp: significant, negligible
    r = compare_systems_by_question(_rows("a", a), _rows("b", b), decision_correct, n_boot=200)
    assert r["statistically_significant"] and r["practical_effect"] == "negligible"
    assert "statistically significant, but its practical effect is limited" in r["interpretation"]


def test_bm25_idf_is_logarithmic_and_ranks_rare_term_match_first():
    import math
    from app.agents.bm25_index import BM25Index
    docs = [{"id": "a", "text": "entropy of a distribution"}, {"id": "b", "text": "the distribution of data"},
            {"id": "c", "text": "a distribution of values"}]
    ix = BM25Index(docs)
    assert ix._idf("entropy") == pytest.approx(math.log(1 + (3 - 1 + .5) / (1 + .5)))
    assert 0 < ix._idf("distribution") < ix._idf("entropy") < 2          # bounded: not an unbounded odds ratio
    assert ix.search("entropy distribution", top_k=3)[0]["id"] == "a"
