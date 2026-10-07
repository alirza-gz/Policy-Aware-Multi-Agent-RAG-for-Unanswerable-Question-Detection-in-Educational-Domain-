"""Per-rule unit tests (positive, negative, edge) + instrumentation contract."""
import pytest

from app.agents.answerability_agent import (LABEL_FALSE_PREMISE, LABEL_FULL, LABEL_OOD, LABEL_PARTIAL,
                                            AnswerabilityAssessment)
from app.agents.policy_agent import ALL_RULES, PolicyAgent

A, C, B = "ANSWER", "CLARIFY", "ABSTAIN"


def assess(label=LABEL_FULL, score=0.9, suff=True):
    return AnswerabilityAssessment(label, score, "t", suff, 0.8, [1.0], ["q"], ["q"], [], [], False, "", {}, "features")


def ev(rules=None, **kw):
    ag = PolicyAgent(enabled_rules=rules, abstain_below=0.3)
    return ag.evaluate(kw.pop("a", assess()), kw.pop("orig", A), kw.pop("text", "Entropy measures uncertainty."), **kw)


def test_contract_fields_present():
    r = ev()
    for k in ("policy_checked", "policy_rule_ids", "policy_rule_triggered", "policy_decision",
              "policy_override", "original_action", "final_action", "policy_reason"):
        assert k in r
    assert r["policy_checked"] and r["policy_rule_ids"] == ALL_RULES and not r["policy_override"]


# R1
def test_R1_positive(): r = ev(service_error=True); assert r["final_action"] == B and "R1" in r["policy_rule_triggered"]
def test_R1_negative(): assert "R1" not in ev()["policy_rule_triggered"]
def test_R1_edge_disabled(): assert ev(rules=[], service_error=True)["final_action"] == A
# R2
def test_R2_positive(): r = ev(text="I diagnose you with flu"); assert r["final_action"] == B and "R2" in r["policy_rule_triggered"]
def test_R2_negative(): assert "R2" not in ev(text="Bayes theorem relates probabilities")["policy_rule_triggered"]
def test_R2_edge_case_insensitive_and_not_applied_when_already_abstain():
    assert ev(text="I PRESCRIBE rest")["final_action"] == B
    assert "R2" not in ev(orig=B, text="i prescribe")["policy_rule_triggered"]
# R3
def test_R3_positive(): r = ev(text="Mail me at a@b.com"); assert "[REDACTED_EMAIL]" in r["final_answer_text"] and r["final_action"] == A and "R3" in r["policy_rule_triggered"]
def test_R3_negative(): assert ev()["final_answer_text"] == "Entropy measures uncertainty."
def test_R3_edge_only_on_answer_path(): assert "R3" not in ev(orig=B, text="a@b.com")["policy_rule_triggered"]
# R4
def test_R4_positive(): r = ev(a=assess(suff=False)); assert r["final_action"] == B and r["policy_override"] and "R4" in r["policy_rule_triggered"]
def test_R4_negative(): assert "R4" not in ev(a=assess(suff=True))["policy_rule_triggered"]
def test_R4_edge_not_applied_to_clarify(): assert ev(a=assess(suff=False), orig=C)["final_action"] == C
# R5
@pytest.mark.parametrize("lab", [LABEL_OOD, LABEL_FALSE_PREMISE])
def test_R5_positive(lab): r = ev(a=assess(label=lab), orig=A); assert r["final_action"] == B and "R5" in r["policy_rule_triggered"]
def test_R5_negative(): assert "R5" not in ev()["policy_rule_triggered"]
def test_R5_edge_overrides_clarify(): assert ev(a=assess(label=LABEL_OOD), orig=C)["final_action"] == B
# R6
def test_R6_positive(): r = ev(a=assess(score=0.1)); assert r["final_action"] == B and "R6" in r["policy_rule_triggered"]
def test_R6_negative(): assert "R6" not in ev(a=assess(score=0.9))["policy_rule_triggered"]
def test_R6_edge_boundary_not_below(): assert "R6" not in ev(a=assess(score=0.3))["policy_rule_triggered"]
# R7
def test_R7_positive(): r = ev(a=assess(label=LABEL_PARTIAL), orig=A); assert r["final_action"] == C and "R7" in r["policy_rule_triggered"]
def test_R7_negative(): assert "R7" not in ev()["policy_rule_triggered"]
def test_R7_edge_not_applied_when_other_rule_already_abstained():
    r = ev(a=assess(label=LABEL_PARTIAL, suff=False), orig=A)
    assert r["final_action"] == B           # R4 fires first; R7 must not resurrect it as CLARIFY


def test_bypass_marks_policy_not_invoked():
    r = PolicyAgent.bypass(A)
    assert r["policy_checked"] is False and r["final_action"] == A and not r["policy_override"]


def test_empty_rule_set_is_checked_but_never_overrides():
    r = ev(rules=[], a=assess(suff=False, label=LABEL_OOD, score=0.0))
    assert r["policy_checked"] and r["policy_rule_ids"] == [] and not r["policy_override"]
