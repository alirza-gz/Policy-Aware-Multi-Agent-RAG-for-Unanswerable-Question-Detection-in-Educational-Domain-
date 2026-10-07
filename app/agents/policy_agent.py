"""Policy/Governance Agent + Decision Controller.

The Answerability Agent proposes an ``original_action``. The Policy Agent then
evaluates declarative rules (each with a stable ID) and may override it. Every
request returns full instrumentation so the effect of the policy layer is
observable and testable:

    policy_checked, policy_rule_ids, policy_rule_triggered, policy_decision,
    policy_override, original_action, final_action, policy_reason

Rules (R-ids are stable; ablations select groups, never rewrite outcomes):

  group "safety"
    R1 service_error_abstain       backend failure                     -> ABSTAIN
    R2 banned_phrase_block         banned phrase in generated answer   -> ABSTAIN
    R3 pii_redaction               redact PII in the final answer      (text only)
  group "answerability_guard"
    R4 evidence_floor              ANSWER proposed but evidence insufficient     -> ABSTAIN
    R5 ood_false_premise_abstain   label is out_of_domain/false_premise          -> ABSTAIN
    R6 low_confidence_abstain      score below validated abstain threshold       -> ABSTAIN
    R7 partial_requires_clarify    label partial but ANSWER proposed             -> CLARIFY
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from app.agents.answerability_agent import (
    AnswerabilityAssessment,
    LABEL_FALSE_PREMISE,
    LABEL_FULL,
    LABEL_OOD,
    LABEL_PARTIAL,
    LABEL_UNANSWERABLE,
)
from app.agents.governance_agent import (
    ABSTAIN_MESSAGE,
    ACTION_ABSTAIN,
    ACTION_ANSWER,
    ACTION_CLARIFY,
    GovernanceAgent,
    SERVICE_ERROR_MESSAGE,
)

RULE_GROUPS = {
    "safety": ["R1", "R2", "R3"],
    "answerability_guard": ["R4", "R5", "R6", "R7"],
}
ALL_RULES = RULE_GROUPS["safety"] + RULE_GROUPS["answerability_guard"]
RULE_NAMES = {
    "R1": "service_error_abstain", "R2": "banned_phrase_block", "R3": "pii_redaction",
    "R4": "evidence_floor", "R5": "ood_false_premise_abstain",
    "R6": "low_confidence_abstain", "R7": "partial_requires_clarify",
}


def proposed_action_from_label(a: AnswerabilityAssessment) -> str:
    """Answerability stage -> proposed action (before any policy)."""
    if a.answerability_label == LABEL_FULL:
        return ACTION_ANSWER
    if a.answerability_label == LABEL_PARTIAL:
        return ACTION_CLARIFY
    return ACTION_ABSTAIN


class PolicyAgent:
    def __init__(self, enabled_rules: Optional[Iterable[str]] = None,
                 abstain_below: float = 0.3, governor: Optional[GovernanceAgent] = None):
        self.enabled = set(ALL_RULES if enabled_rules is None else enabled_rules)
        self.abstain_below = float(abstain_below)
        self.gov = governor or GovernanceAgent()

    def evaluate(self, assessment: AnswerabilityAssessment, original_action: str,
                 answer_text: str = "", service_error: bool = False) -> Dict:
        """Apply rules in fixed precedence; return decision + instrumentation."""
        checked = sorted(self.enabled, key=ALL_RULES.index)
        triggered: List[str] = []
        reasons: List[str] = []
        action = original_action
        final_answer = answer_text

        def fire(rid: str, new_action: Optional[str], why: str):
            nonlocal action
            triggered.append(rid)
            reasons.append(f"{rid}:{RULE_NAMES[rid]}({why})")
            if new_action is not None:
                action = new_action

        if "R1" in self.enabled and service_error:
            fire("R1", ACTION_ABSTAIN, "backend unavailable")
        if "R2" in self.enabled and action != ACTION_ABSTAIN:
            hits = self.gov._check_banned_phrases(answer_text)
            if hits:
                fire("R2", ACTION_ABSTAIN, str(hits))
        if "R5" in self.enabled and action != ACTION_ABSTAIN \
                and assessment.answerability_label in (LABEL_OOD, LABEL_FALSE_PREMISE):
            fire("R5", ACTION_ABSTAIN, assessment.answerability_label)
        if "R4" in self.enabled and action == ACTION_ANSWER and not assessment.evidence_sufficient:
            fire("R4", ACTION_ABSTAIN, "evidence insufficient")
        if "R6" in self.enabled and action == ACTION_ANSWER \
                and assessment.answerability_score < self.abstain_below:
            fire("R6", ACTION_ABSTAIN,
                 f"score {assessment.answerability_score:.2f} < {self.abstain_below:.2f}")
        if "R7" in self.enabled and action == ACTION_ANSWER \
                and assessment.answerability_label == LABEL_PARTIAL:
            fire("R7", ACTION_CLARIFY, "partial coverage")

        if action == ACTION_ANSWER and "R3" in self.enabled:
            red = self.gov._redact_pii(answer_text)
            if red != answer_text:
                fire("R3", None, "pii redacted")
                final_answer = red

        return {
            "policy_checked": True,
            "policy_rule_ids": checked,
            "policy_rule_triggered": triggered,
            "policy_decision": action,
            "policy_override": action != original_action,
            "original_action": original_action,
            "final_action": action,
            "policy_reason": "; ".join(reasons) if reasons else "no_rule_triggered",
            "final_answer_text": final_answer,
        }

    @staticmethod
    def bypass(original_action: str) -> Dict:
        """Used by the no_policy_agent ablation: Policy is not invoked at all."""
        return {
            "policy_checked": False, "policy_rule_ids": [], "policy_rule_triggered": [],
            "policy_decision": original_action, "policy_override": False,
            "original_action": original_action, "final_action": original_action,
            "policy_reason": "policy_agent_not_invoked", "final_answer_text": None,
        }


class DecisionController:
    """Combines Answerability Agent + Policy Agent into the final decision.

    ablation switches:
      use_answerability=False -> proposed action is ANSWER (no answerability reasoning);
                                 policy rules still run on the evidence features.
      policy=None             -> Policy Agent bypassed.
    """

    def __init__(self, answerability, policy: Optional[PolicyAgent], use_answerability: bool = True):
        self.answerability = answerability
        self.policy = policy
        self.use_answerability = use_answerability

    def decide(self, question: str, passages, retrieval_confidence: float,
               reasoner: Dict) -> Dict:
        a = self.answerability.assess(question, passages, retrieval_confidence, reasoner)
        proposed = proposed_action_from_label(a) if self.use_answerability else ACTION_ANSWER
        answer_text = str(reasoner.get("answer", "") or "")
        svc = bool(reasoner.get("service_error"))
        pol = (self.policy.evaluate(a, proposed, answer_text, svc) if self.policy is not None
               else PolicyAgent.bypass(proposed))
        action = pol["final_action"]

        clar = {"clarification_trigger": "none", "clarification_reason": "",
                "missing_information": "", "clarification_question": ""}
        if action == ACTION_CLARIFY:
            clar = self.answerability.build_clarification(a)
            if clar["clarification_trigger"] == "none":      # CLARIFY reached via a rule/ablation
                clar["clarification_trigger"] = "policy_rule"
                clar["clarification_reason"] = pol["policy_reason"]
                clar["clarification_question"] = (
                    "I can only partly answer this from the course materials. Which specific "
                    "aspect of the question should I focus on?")
        if action == ACTION_ANSWER:
            final_answer = pol["final_answer_text"] if pol["final_answer_text"] is not None else answer_text
        elif action == ACTION_CLARIFY:
            final_answer = clar["clarification_question"]
        else:
            final_answer = SERVICE_ERROR_MESSAGE if svc else ABSTAIN_MESSAGE

        return {
            "action": action,
            "predicted_action": action,
            "final_answer": final_answer,
            "answerability_label": a.answerability_label,
            "answerability_score": a.answerability_score,
            "answerability_score_kind": a.score_kind,
            "retrieval_confidence": a.retrieval_confidence,
            "evidence_sufficient": a.evidence_sufficient,
            "answerable_parts": a.answerable_parts,
            "missing_parts": a.missing_parts,
            "answerability_label_source": a.label_source,
            **{k: v for k, v in pol.items() if k != "final_answer_text"},
            **clar,
            "reason": pol["policy_reason"],
        }
