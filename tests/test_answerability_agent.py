"""Unit tests: Answerability Agent (labels, evidence sufficiency, clarification)."""
import inspect

import app.agents.answerability_agent as mod
import app.agents.policy_agent as pol_mod
from app.agents.answerability_agent import (LABEL_FALSE_PREMISE, LABEL_FULL, LABEL_OOD, LABEL_PARTIAL,
                                            LABEL_UNANSWERABLE, AnswerabilityAgent, ScoreModel,
                                            is_generic_clarification, split_question_parts)

EV = [{"text": "Ensemble learning combines several models. An intrusion detection system monitors "
               "a network; ensemble learning reduces the total error of such monitoring systems."}]
PARTIAL_Q = ("How does ensemble learning improve intrusion detection systems? And when did industrial "
             "applications of deep learning to speech recognition begin?")


def test_fully_answerable():
    a = AnswerabilityAgent().assess("How does ensemble learning reduce error in intrusion detection systems?", EV, 0.7)
    assert a.answerability_label == LABEL_FULL and a.evidence_sufficient


def test_partially_answerable_two_parts():
    a = AnswerabilityAgent().assess(PARTIAL_Q, EV, 0.7)
    assert a.answerability_label == LABEL_PARTIAL
    assert len(a.answerable_parts) == 1 and len(a.missing_parts) == 1
    assert "speech recognition" in a.missing_parts[0] and not a.evidence_sufficient


def test_out_of_domain_low_retrieval_and_overlap():
    a = AnswerabilityAgent().assess("What is the name of the StarKist tuna mascot?", EV, 0.05)
    assert a.answerability_label == LABEL_OOD


def test_no_passages_is_ood():
    assert AnswerabilityAgent().assess("What is entropy?", [], 0.0).answerability_label == LABEL_OOD


def test_single_part_uncovered_is_unanswerable_not_ood_when_retrieval_ok():
    a = AnswerabilityAgent().assess("Who invented the quantum gradient teleporter?", EV, 0.6)
    assert a.answerability_label == LABEL_UNANSWERABLE


def test_false_premise_needs_nli_or_reasoner_label():
    q = "How does ensemble learning increase error in intrusion detection systems?"
    assert AnswerabilityAgent().assess(q, EV, 0.7).answerability_label != LABEL_FALSE_PREMISE   # lexical alone cannot
    a = AnswerabilityAgent().assess(q, EV, 0.7, {"answerability_label": LABEL_FALSE_PREMISE})
    assert a.answerability_label == LABEL_FALSE_PREMISE and a.label_source == "reasoner"
    b = AnswerabilityAgent(nli_contradiction=lambda q, p: 0.9).assess(q, EV, 0.7)
    assert b.answerability_label == LABEL_FALSE_PREMISE and b.label_source == "nli"


def test_ambiguous_question_is_partial_with_ambiguity_clarification():
    ag = AnswerabilityAgent()
    a = ag.assess("Entropy?", EV, 0.6)
    assert a.ambiguous and a.answerability_label == LABEL_PARTIAL
    assert ag.build_clarification(a)["clarification_trigger"] == "ambiguity"


def test_reasoner_veto_on_covered_question():
    a = AnswerabilityAgent().assess("How does ensemble learning reduce error in intrusion detection systems?",
                                    EV, 0.7, {"is_answerable": False})
    assert a.answerability_label == LABEL_UNANSWERABLE


def test_clarification_names_missing_and_supported_parts():
    ag = AnswerabilityAgent()
    a = ag.assess(PARTIAL_Q, EV, 0.7)
    c = ag.build_clarification(a)
    assert c["clarification_trigger"] == "partial_coverage"
    assert "speech recognition" in c["clarification_question"] and "intrusion detection" in c["clarification_question"]
    assert not is_generic_clarification(c["clarification_question"])


def test_generic_clarification_detected():
    assert is_generic_clarification("Could you please clarify or add more detail to your question?")
    assert is_generic_clarification("")


def test_split_parts_edge_cases():
    assert len(split_question_parts("What is entropy?")) == 1
    assert len(split_question_parts("What is bias, and how does variance relate to it?")) == 2
    assert split_question_parts("") == []


def test_score_model_is_probability_and_monotone():
    m = ScoreModel(["retrieval_confidence"], [3.0], -1.0)
    lo, hi = m.predict({"retrieval_confidence": 0.1}), m.predict({"retrieval_confidence": 0.9})
    assert 0 < lo < hi < 1
    assert ScoreModel.from_json(m.to_json()).predict({"retrieval_confidence": 0.9}) == hi


def test_uncalibrated_score_is_flagged():
    assert AnswerabilityAgent().assess("What is entropy in data?", EV, 0.5).score_kind == "heuristic_uncalibrated"


def test_no_gold_label_fields_used_by_system_code():
    """Leakage guard: system code must never read gold/annotation fields."""
    banned = ["out_of_scope_part_question", "answerable_part_question", "gold_answerable",
              "expected_action", "held_out_passage", "evidence_passage", "gold_answers"]
    for m in (mod, pol_mod):
        src = inspect.getsource(m)
        for b in banned:
            assert b not in src, f"{m.__name__} references {b}"
