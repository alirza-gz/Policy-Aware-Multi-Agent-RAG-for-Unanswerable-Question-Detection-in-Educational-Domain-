"""Integration + regression tests for the decision pipeline and the single-source evaluation."""
import json
from pathlib import Path

import pytest

from app.agents.answerability_agent import AnswerabilityAgent
from app.agents.policy_agent import ALL_RULES, DecisionController, PolicyAgent
from eval.aggregate import aggregate_run
from eval.pipeline_v2 import ALL_SYSTEMS, RETRIEVAL_PROFILES, build_systems, fit_artifacts
from run_evaluation import retrieval_trace, signals_to_rows

EV = [{"id": "p1", "text": "Ensemble learning combines several models. An intrusion detection system monitors a "
                           "network; ensemble learning reduces the total error of such monitoring systems.", "score": .7}]
PARTIAL_Q = ("How does ensemble learning improve intrusion detection systems? And when did industrial "
             "applications of deep learning to speech recognition begin?")
OK_RR = {"answer": "It reduces total error.", "is_answerable": True, "answerability_confidence": .9,
         "needs_clarification": False, "confidence": .9}


def controller(rules=ALL_RULES, **kw):
    return DecisionController(AnswerabilityAgent(), PolicyAgent(rules, 0.3), **kw)


def test_answer_path():
    d = controller().decide("How does ensemble learning reduce error in intrusion detection systems?", EV, .7, OK_RR)
    assert d["action"] == "ANSWER" and d["answerability_label"] == "fully_answerable" and d["final_answer"] == OK_RR["answer"]


def test_clarify_path_partially_answerable_even_when_llm_claims_answerable():
    d = controller().decide(PARTIAL_Q, EV, .7, OK_RR)
    assert d["action"] == "CLARIFY" and d["clarification_trigger"] == "partial_coverage"
    assert "speech recognition" in d["final_answer"] and d["clarification_question"] == d["final_answer"]


def test_abstain_paths_ood_unanswerable_false_premise_service_error():
    c = controller()
    assert c.decide("What is the StarKist tuna mascot?", EV, .05, OK_RR)["action"] == "ABSTAIN"
    assert c.decide("Who invented the quantum gradient teleporter?", EV, .6, OK_RR)["action"] == "ABSTAIN"
    fp = c.decide("How does ensemble learning increase error in intrusion detection systems?", EV, .7,
                  {**OK_RR, "answerability_label": "false_premise"})
    # FINDING (documented): in the full system the answerability stage already proposes ABSTAIN for
    # false_premise/out_of_domain, so Policy rule R5 is redundant there (no override, not triggered).
    assert fp["action"] == "ABSTAIN" and fp["original_action"] == "ABSTAIN" and not fp["policy_override"]
    assert "R5" not in fp["policy_rule_triggered"]
    # R5 is only effective when the answerability stage is bypassed (defence in depth):
    fp2 = controller(use_answerability=False).decide(
        "How does ensemble learning increase error in intrusion detection systems?", EV, .7,
        {**OK_RR, "answerability_label": "false_premise"})
    assert fp2["original_action"] == "ANSWER" and fp2["final_action"] == "ABSTAIN" and "R5" in fp2["policy_rule_triggered"]
    assert c.decide("anything", EV, .7, {**OK_RR, "service_error": True})["action"] == "ABSTAIN"


def test_instrumentation_logged_on_every_decision():
    d = controller().decide(PARTIAL_Q, EV, .7, OK_RR)
    for k in ("predicted_action", "answerability_score", "answerability_label", "retrieval_confidence",
              "policy_decision", "clarification_trigger", "clarification_reason", "policy_checked",
              "policy_rule_ids", "policy_rule_triggered", "policy_override", "original_action",
              "final_action", "policy_reason"):
        assert k in d


def test_policy_changes_final_decision_when_it_matters():
    """Evidence floor: label says FULL (LLM/feature) but retrieval is below the evidence floor."""
    q = "How does ensemble learning reduce error in intrusion detection systems?"
    with_pol = controller().decide(q, EV, 0.05, OK_RR)             # retrieval 0.05 < floor, coverage fine
    no_pol = DecisionController(AnswerabilityAgent(), None).decide(q, EV, 0.05, OK_RR)
    assert no_pol["action"] == "ANSWER"
    assert with_pol["action"] == "ABSTAIN" and with_pol["policy_override"] and "R4" in with_pol["policy_rule_triggered"]


def test_no_answerability_still_lets_guard_rules_block_unsupported_answers():
    d = controller(use_answerability=False).decide(PARTIAL_Q, EV, .7, OK_RR)
    assert d["original_action"] == "ANSWER" and d["final_action"] != "ANSWER"       # R4 blocks (evidence insufficient)


def test_policy_bypass_vs_empty_rules_are_equivalent_by_construction():
    bypass = DecisionController(AnswerabilityAgent(), None)
    empty = DecisionController(AnswerabilityAgent(), PolicyAgent([], .3))
    for q, rc in [(PARTIAL_Q, .7), ("What is the StarKist tuna mascot?", .05)]:
        assert bypass.decide(q, EV, rc, OK_RR)["action"] == empty.decide(q, EV, rc, OK_RR)["action"]


# ----------------------- synthetic end-to-end run -----------------------
def _signal(i, cat, q, expected, passages, rc, rr, seed=42, mode=("hybrid", True)):
    return {"seed": seed, "retrieval_mode": mode[0], "reranked": mode[1], "retriever_confidence": rc,
            "passages": passages, "retrieved_ids": [p["id"] for p in passages], "dense": passages, "sparse": passages,
            "fused": passages, "pre_rerank": passages, "ranked_deep": passages, "reasoning_result": rr,
            "question": {"id": f"q{i}", "category": cat, "question": q, "expected_action": expected,
                         "gold_answerable": expected == "ANSWER", "gold_answers": ["total error"] if expected == "ANSWER" else [],
                         "evidence_ids": ["p1"] if cat != "out_of_domain" else [], "course": "ml",
                         "source_document": f"doc{i % 9}"}}


@pytest.fixture()
def synthetic_signals():
    sig = []
    for i in range(60):
        k = i % 3
        if k == 0:
            sig.append(_signal(i, "answerable", "How does ensemble learning reduce error in intrusion detection systems?", "ANSWER", EV, .7, OK_RR))
        elif k == 1:
            sig.append(_signal(i, "partially_answerable", PARTIAL_Q, "CLARIFY", EV, .7, OK_RR))
        else:
            sig.append(_signal(i, "out_of_domain", f"What is the name of mascot number {i}?", "ABSTAIN", [{"id": "p9", "text": "unrelated", "score": .1}], .05,
                               {**OK_RR, "is_answerable": False, "answer": ""}))
    extra = []
    for s in sig:                                   # other retrieval profiles needed by ablations
        for mode in (("none", False), ("hybrid", False), ("dense", True)):
            extra.append({**s, "retrieval_mode": mode[0], "reranked": mode[1],
                          "passages": [] if mode[0] == "none" else s["passages"]})
    return sig + extra


def test_full_run_single_source_of_truth(tmp_path: Path, synthetic_signals):
    art = fit_artifacts(synthetic_signals, val_fraction=0.4)
    assert set(art["val_ids"]).isdisjoint(art["test_ids"]) and art["threshold_selection"]["fit_on"] == "validation_only"
    rows = signals_to_rows(synthetic_signals, ALL_SYSTEMS, art, 0.2, {"reasoning_mode": "unit"})
    assert {r["mode"] for r in rows} == set(ALL_SYSTEMS)
    run = tmp_path
    with open(run / "predictions.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    with open(run / "retrieval_trace.jsonl", "w") as f:
        for r in retrieval_trace(synthetic_signals):
            f.write(json.dumps(r) + "\n")
    (run / "calibration_artifacts.json").write_text(json.dumps({k: v for k, v in art.items() if k != "score_model"}))
    out = aggregate_run(run, banned=["i diagnose you"])
    shas = set()
    for name in ("metrics", "category_metrics", "course_metrics", "retrieval_metrics", "calibration",
                 "statistical_tests", "confidence_intervals", "partial_answerable_eval",
                 "policy_instrumentation", "ablation_equivalence"):
        shas.add(json.loads((run / f"{name}.json").read_text())["_meta"]["predictions_sha256"])
    assert len(shas) == 1 and json.loads((run / "consistency.json").read_text())["predictions_sha256"] in shas
    m = out["metrics"]
    # test split only: no validation question leaks into reported metrics
    n_rows = json.loads((run / "metrics.json").read_text())["_meta"]["n_prediction_rows_test_split"]
    assert n_rows < len(rows)
    # CLARIFY is a usable decision on this synthetic set and the baseline never clarifies
    assert m["full_system"]["clarification_rate"] > 0 and m["standard_rag"]["clarification_rate"] == 0
    eq = {(p["system_a"], p["system_b"]) for p in json.loads((run / "ablation_equivalence.json").read_text())["identical_pairs"]}
    assert ("no_governance", "standard_rag") in eq or ("standard_rag", "no_governance") in eq
    assert RETRIEVAL_PROFILES["no_hybrid"] == ("dense", True)


# ----------------------- regression vs. previous results -----------------------
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results/predictions.jsonl").exists(), reason="legacy results absent")
def test_regression_legacy_sciq_metrics_match_predictions_but_metrics_json_does_not():
    """Intentional/known: sciq_metrics.json was produced from results/predictions.jsonl;
    metrics.json belongs to an earlier run whose predictions are not in the repo."""
    from eval.metrics import compute_by_mode, load_predictions
    got = compute_by_mode(load_predictions(ROOT / "results/predictions.jsonl"))
    sciq = json.loads((ROOT / "results/sciq_metrics.json").read_text())
    old = json.loads((ROOT / "results/metrics.json").read_text())
    assert got["policy_aware"]["confusion_matrix"] == sciq["policy_aware"]["confusion_matrix"]
    assert got["policy_aware"]["confusion_matrix"] != old["policy_aware"]["confusion_matrix"]


@pytest.mark.skipif(not (ROOT / "results/confusion_decision_full_system.csv").exists(), reason="legacy results absent")
def test_regression_trace_report_vs_confusion_matrix_discrepancy_is_definitional():
    """2318 (trace report) = 2131 (ABSTAIN where expected ABSTAIN) + 187 (expected CLARIFY -> ABSTAIN):
    the trace report used the binary gold label, the confusion matrix the 3-class expected action."""
    import csv
    rows = list(csv.reader(open(ROOT / "results/confusion_decision_full_system.csv")))
    exp_abs_abs = int(rows[3][3]); exp_clar_abs = int(rows[2][3])
    assert (exp_abs_abs, exp_clar_abs, exp_abs_abs + exp_clar_abs) == (2131, 187, 2318)
