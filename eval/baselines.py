"""Baseline and ablation system definitions for the thesis evaluation.

FULL system (advisor Option B)
------------------------------
    Multi-Agent RAG + Hybrid Retrieval + Reranker + Answerability + Policy

Baselines
---------
    llm_no_rag       LLM only (no retrieval). Unavoidable difference: no corpus.
    standard_rag     Retriever (+rerank) -> LLM; always ANSWER; no governance.
    rag_threshold    Standard RAG + retrieval-confidence threshold abstention.
                     Configurable ``rag_threshold``; does NOT reuse GovernanceAgent.
    agentic_rag      Honest proxy for Agentic RAG: reasoner self-governs via its
                     own is_answerable / needs_clarification flags. No separate
                     Policy/Governance Agent. Not an iterative tool-calling loop
                     (the codebase has none); documented as such.
    full_system      Policy-Aware Multi-Agent RAG (hybrid + rerank + governance).

Ablations (delta vs FULL; only the named component changes)
-----------------------------------------------------------
    no_policy_agent       FULL - Policy Agent   (= agentic_rag decide on FULL retrieval)
    no_answerability      FULL - Answerability  (governance with answerability_enabled=false;
                                                 safety/PII remain; typically ANSWER)
    no_reranker           FULL - Reranker       (same hybrid retrieval, skip rerank)
    no_multi_agent        FULL - Multi-Agent    (single-agent decide on FULL retrieval)
    no_policy_rules       FULL - Policy Rules   (banned-phrase safety disabled)
    no_hybrid             FULL - Hybrid         (dense-only + rerank + governance;
                                                 optional ablation kept for Phase 1)

Fairness
--------
Systems that share the same (retrieval_mode, use_reranker) tuple share one
retriever+reasoner pass. Decision layers then diverge on cached signals.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.agents.governance_agent import (
    GovernanceAgent,
    ACTION_ANSWER,
    ACTION_CLARIFY,
    ACTION_ABSTAIN,
    ABSTAIN_MESSAGE,
)

# Advisor-facing names (also used as ``mode`` in prediction rows).
BASELINE_SYSTEMS = [
    "llm_no_rag",
    "standard_rag",
    "rag_threshold",
    "agentic_rag",
    "full_system",
]

ABLATION_SYSTEMS = [
    "no_policy_agent",
    "no_answerability",
    "no_reranker",
    "no_multi_agent",
    "no_policy_rules",
    "no_hybrid",
]

ALL_SYSTEMS = BASELINE_SYSTEMS + ABLATION_SYSTEMS

# Retrieval profile per system: (retrieval_mode, use_reranker)
# FULL uses hybrid + rerank. Ablations change exactly one axis where applicable.
RETRIEVAL_PROFILES: Dict[str, Tuple[str, bool]] = {
    "llm_no_rag": ("none", False),
    "standard_rag": ("hybrid", True),
    "rag_threshold": ("hybrid", True),
    "agentic_rag": ("hybrid", True),
    "full_system": ("hybrid", True),
    "no_policy_agent": ("hybrid", True),
    "no_answerability": ("hybrid", True),
    "no_reranker": ("hybrid", False),
    "no_multi_agent": ("hybrid", True),
    "no_policy_rules": ("hybrid", True),
    "no_hybrid": ("dense", True),
}


def _row(action: str, final_answer: str, reason: str) -> Dict:
    return {"action": action, "final_answer": final_answer, "reason": reason}


def decide_always_answer(reasoning_result: Dict, retriever_confidence: float) -> Dict:
    """Standard RAG / LLM-no-RAG: always return the generated answer."""
    return _row(
        ACTION_ANSWER,
        str(reasoning_result.get("answer", "") or ""),
        "always_answer_no_governance",
    )


def decide_rag_threshold(
    reasoning_result: Dict,
    retriever_confidence: float,
    threshold: float = 0.2,
) -> Dict:
    """RAG + Threshold: abstain only when retrieval confidence is below tau.

    Intentionally independent of GovernanceAgent (no clarify band, no
    answerability gate, no banned-phrase policy).
    """
    if float(retriever_confidence) < float(threshold):
        return _row(
            ACTION_ABSTAIN,
            ABSTAIN_MESSAGE,
            f"rag_threshold_below:{threshold}",
        )
    return _row(
        ACTION_ANSWER,
        str(reasoning_result.get("answer", "") or ""),
        f"rag_threshold_pass:{threshold}",
    )


def decide_agentic_rag(reasoning_result: Dict, retriever_confidence: float) -> Dict:
    """Single-agent / Agentic RAG proxy: reasoner self-governs, no Policy Agent."""
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
    """Policy-aware system (full or ablated) on the unmodified GovernanceAgent."""

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


def build_systems(
    seed_governance: Optional[Dict] = None,
    rag_threshold: float = 0.2,
) -> Dict[str, callable]:
    """Instantiate every system as ``name -> decide(reasoning_result, retr_conf)``."""
    base_gov = dict(seed_governance or {})
    tau = float(rag_threshold)

    def _threshold_decide(rr, rc, _tau=tau):
        return decide_rag_threshold(rr, rc, threshold=_tau)

    full = GovernedSystem(governance=base_gov or None)
    no_ans_gov = dict(base_gov)
    no_ans_gov["answerability_enabled"] = False

    systems: Dict[str, callable] = {
        "llm_no_rag": decide_always_answer,
        "standard_rag": decide_always_answer,
        "rag_threshold": _threshold_decide,
        "agentic_rag": decide_agentic_rag,
        "full_system": full.decide,
        # Ablations
        "no_policy_agent": decide_agentic_rag,
        "no_answerability": GovernedSystem(governance=no_ans_gov).decide,
        "no_reranker": GovernedSystem(governance=base_gov or None).decide,
        "no_multi_agent": decide_agentic_rag,
        "no_policy_rules": GovernedSystem(
            governance=base_gov or None,
            banned_phrases=["@@never-matches-sentinel@@"],
        ).decide,
        "no_hybrid": GovernedSystem(governance=base_gov or None).decide,
    }
    return systems


def systems_for_profile(mode: str, use_reranker: bool) -> List[str]:
    """Return system names that share a retrieval profile."""
    return [
        name
        for name, (m, r) in RETRIEVAL_PROFILES.items()
        if m == mode and r == use_reranker and name in ALL_SYSTEMS
    ]


# Back-compat aliases used by older report code / docs.
decide_vanilla_rag = decide_always_answer
decide_single_agent_rag = decide_agentic_rag
