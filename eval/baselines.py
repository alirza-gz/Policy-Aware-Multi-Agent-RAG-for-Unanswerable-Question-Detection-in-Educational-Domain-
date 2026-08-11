"""Baseline and ablation system definitions for the thesis evaluation.

Every system consumes the SAME cached per-question signals (retrieved passages,
retriever confidence, and the reasoning agent's output) so the comparison is
strictly fair: identical retriever, embedding model, LLM, and prompts. The
systems differ ONLY in the decision layer applied on top of those signals.

Systems
-------
vanilla_rag       Retriever + LLM only. The raw generated answer is always
                  returned (action is always ANSWER). No answerability
                  detection of any kind.
single_agent_rag  One agent (the reasoner) self-governs: its own
                  ``is_answerable`` / ``needs_clarification`` flags are
                  respected, but there is NO Governance Agent (no policy
                  thresholds, no safety rules, no confidence bands).
policy_aware      The current, unmodified Policy-Aware Multi-Agent RAG:
                  GovernanceAgent.decide() with the configured policy.

Ablations (variants of policy_aware, built ONLY through the constructor
parameters GovernanceAgent already exposes -- the agent code is untouched)
--------------------------------------------------------------------------
no_governance           GOVERNANCE_ENABLED=false equivalent (= single agent).
no_clarification        CLARIFY path disabled: clarify band collapsed and the
                        reasoner's clarification flag ignored.
no_retriever_confidence Retrieval-confidence abstention rule disabled.
no_policy_rules         Safety policy (banned phrases) disabled.
"""

from typing import Dict, List, Optional

from app.agents.governance_agent import (
    GovernanceAgent,
    ACTION_ANSWER,
    ACTION_CLARIFY,
    ACTION_ABSTAIN,
    ABSTAIN_MESSAGE,
)

# The three baselines required by the thesis evaluation.
BASELINE_SYSTEMS = ["vanilla_rag", "single_agent_rag", "policy_aware"]

# Ablation configurations: which GovernanceAgent constructor overrides to use.
# ``None`` for a governance key means "keep the configured value".
ABLATION_SYSTEMS = {
    "full_system": {},  # identical to policy_aware; kept for the ablation table
    "no_governance": {"disable_governance": True},
    "no_clarification": {
        "governance": {"clarify_band": [0.0, 0.0]},
        "ignore_model_clarification": True,
    },
    "no_retriever_confidence": {"governance": {"retriever_abstain_below": -1.0}},
    # GovernanceAgent treats an empty list as "use config", so pass a sentinel
    # phrase that can never occur in an answer -> safety rules effectively off.
    "no_policy_rules": {"banned_phrases": ["@@never-matches-sentinel@@"]},
}

ALL_SYSTEMS = BASELINE_SYSTEMS + [s for s in ABLATION_SYSTEMS if s != "full_system"] + ["full_system"]


def _row(action: str, final_answer: str, reason: str) -> Dict:
    return {"action": action, "final_answer": final_answer, "reason": reason}


def decide_vanilla_rag(reasoning_result: Dict, retriever_confidence: float) -> Dict:
    """Vanilla RAG: retriever + LLM, the generated answer is always returned."""
    return _row(
        ACTION_ANSWER,
        str(reasoning_result.get("answer", "") or ""),
        "vanilla_rag_always_answers",
    )


def decide_single_agent_rag(reasoning_result: Dict, retriever_confidence: float) -> Dict:
    """Single-Agent RAG: the reasoner's own flags, no Governance Agent."""
    if not bool(reasoning_result.get("is_answerable", False)):
        return _row(ACTION_ABSTAIN, ABSTAIN_MESSAGE, "reasoner_self_reported_unanswerable")
    if bool(reasoning_result.get("needs_clarification", False)):
        cq = str(reasoning_result.get("clarification_question", "") or "").strip() or (
            "Could you please clarify or add more detail to your question?"
        )
        return _row(ACTION_CLARIFY, cq, "reasoner_self_requested_clarification")
    return _row(
        ACTION_ANSWER,
        str(reasoning_result.get("answer", "") or ""),
        "reasoner_self_reported_answerable",
    )


class GovernedSystem:
    """A policy-aware system (full or ablated) built on the unmodified agent."""

    def __init__(
        self,
        governance: Optional[Dict] = None,
        banned_phrases: Optional[List[str]] = None,
        ignore_model_clarification: bool = False,
    ):
        kwargs = {}
        if governance:
            kwargs["governance"] = governance
        if banned_phrases is not None:
            kwargs["banned_phrases"] = banned_phrases
        self.governor = GovernanceAgent(**kwargs)
        self.ignore_model_clarification = ignore_model_clarification

    def decide(self, reasoning_result: Dict, retriever_confidence: float) -> Dict:
        rr = dict(reasoning_result)
        if self.ignore_model_clarification:
            rr["needs_clarification"] = False
            rr["clarification_question"] = ""
        decision = self.governor.decide(rr, retriever_confidence=retriever_confidence)
        return _row(decision["action"], decision["final_answer"], decision["reason"])


def build_systems(seed_governance: Optional[Dict] = None) -> Dict[str, callable]:
    """Instantiate every system as ``name -> decide(reasoning_result, retr_conf)``.

    ``seed_governance`` optionally overrides the base governance config for all
    governed systems (used by the experiment config file).
    """
    base_gov = dict(seed_governance or {})

    systems: Dict[str, callable] = {
        "vanilla_rag": decide_vanilla_rag,
        "single_agent_rag": decide_single_agent_rag,
        "policy_aware": GovernedSystem(governance=base_gov or None).decide,
    }

    for name, cfg in ABLATION_SYSTEMS.items():
        if cfg.get("disable_governance"):
            systems[name] = decide_single_agent_rag
            continue
        gov = dict(base_gov)
        gov.update(cfg.get("governance", {}))
        systems[name] = GovernedSystem(
            governance=gov or None,
            banned_phrases=cfg.get("banned_phrases"),
            ignore_model_clarification=cfg.get("ignore_model_clarification", False),
        ).decide

    return systems
