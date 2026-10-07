"""Decision systems (baselines + ablations) built on Answerability + Policy agents,
plus validation-only fitting of the score model / thresholds.

Everything here consumes *signals* (retrieval + reasoner outputs) and is a pure function
of them, so decisions can be replayed deterministically from ``signals.jsonl``.

Systems (decision layer only; retrieval profile in RETRIEVAL_PROFILES):
  llm_no_rag, standard_rag, rag_threshold, agentic_rag     baselines (unchanged semantics)
  full_system        Answerability Agent + Policy Agent (R1-R7)
  no_policy_agent    Policy Agent NOT invoked (proposed action is final)
  no_policy_rules    Policy Agent invoked with an EMPTY rule set
  no_safety_rules    Policy rules R1-R3 removed
  no_guard_rules     Policy rules R4-R7 removed
  no_answerability   Answerability stage bypassed (proposes ANSWER); Policy rules still run
  no_governance      neither Answerability nor Policy: always ANSWER
  no_reranker, no_hybrid   full decision layer, different retrieval profile
  no_multi_agent     single-agent self-governance (== agentic_rag decision function)

Some pairs are identical BY CONSTRUCTION (no_governance==standard_rag,
no_multi_agent==agentic_rag, and no_policy_agent==no_policy_rules because the Policy
Agent has no behaviour beyond executing its rules). aggregate.py reports measured
equivalence instead of assuming it.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from app.agents.answerability_agent import SCORE_FEATURES, AnswerabilityAgent, ScoreModel
from app.agents.policy_agent import ALL_RULES, RULE_GROUPS, DecisionController, PolicyAgent
from eval.baselines import (decide_agentic_rag, decide_always_answer, decide_rag_threshold)
from eval.calibration import fit_score_model, grouped_split, select_thresholds

BASELINES = ["llm_no_rag", "standard_rag", "rag_threshold", "agentic_rag", "full_system"]
ABLATIONS = ["no_policy_agent", "no_policy_rules", "no_safety_rules", "no_guard_rules",
             "no_answerability", "no_governance", "no_reranker", "no_hybrid", "no_multi_agent"]
ALL_SYSTEMS = BASELINES + ABLATIONS

RETRIEVAL_PROFILES: Dict[str, Tuple[str, bool]] = {s: ("hybrid", True) for s in ALL_SYSTEMS}
RETRIEVAL_PROFILES.update({"llm_no_rag": ("none", False), "no_reranker": ("hybrid", False),
                           "no_hybrid": ("dense", True)})

SystemFn = Callable[[str, list, Dict, float], Dict]


def _legacy(fn) -> SystemFn:
    def run(question, passages, rr, rc):
        d = fn(rr, rc)
        return {**d, "predicted_action": d["action"], "policy_checked": False,
                "policy_rule_ids": [], "policy_rule_triggered": [], "policy_override": False,
                "policy_decision": d["action"], "original_action": d["action"],
                "final_action": d["action"], "policy_reason": "no_policy_layer"}
    return run


def build_systems(artifacts: Optional[Dict] = None, rag_threshold: float = 0.2) -> Dict[str, SystemFn]:
    art = artifacts or {}
    ans = AnswerabilityAgent(thresholds=art.get("assessment_thresholds"),
                             score_model=art.get("score_model"))
    abstain_below = float((art.get("thresholds") or {}).get("abstain_below", 0.3))

    def ctrl(rules, use_ans=True, policy=True):
        pol = PolicyAgent(enabled_rules=rules, abstain_below=abstain_below) if policy else None
        c = DecisionController(ans, pol, use_answerability=use_ans)
        return lambda q, ps, rr, rc: c.decide(q, ps, rc, rr)

    full = ctrl(ALL_RULES)
    not_in = lambda g: [r for r in ALL_RULES if r not in RULE_GROUPS[g]]  # noqa: E731
    return {
        "llm_no_rag": _legacy(decide_always_answer),
        "standard_rag": _legacy(decide_always_answer),
        "rag_threshold": _legacy(lambda rr, rc: decide_rag_threshold(rr, rc, rag_threshold)),
        "agentic_rag": _legacy(decide_agentic_rag),
        "full_system": full,
        "no_policy_agent": ctrl(ALL_RULES, policy=False),
        "no_policy_rules": ctrl([]),
        "no_safety_rules": ctrl(not_in("safety")),
        "no_guard_rules": ctrl(not_in("answerability_guard")),
        "no_answerability": ctrl(ALL_RULES, use_ans=False),
        "no_governance": ctrl([], use_ans=False, policy=False),
        "no_reranker": full, "no_hybrid": full,
        "no_multi_agent": _legacy(decide_agentic_rag),
    }


def fit_artifacts(signals: List[Dict], val_fraction: float = 0.3, split_seed: int = 42,
                  max_false_accept: float = 0.10) -> Dict:
    """Fit score model + abstain threshold on the VALIDATION split of the full-profile signals.

    Returns artifacts plus the split (question ids) so reporting can use TEST ids only.
    """
    prof = [s for s in signals if (s["retrieval_mode"], s["reranked"]) == ("hybrid", True)]
    uniq: Dict[str, Dict] = {}
    for s in prof:
        uniq.setdefault(str(s["question"]["id"]), s)       # one row per question (seeds are not independent)
    qs = list(uniq.values())
    val_idx, test_idx = grouped_split([s["question"] for s in qs], val_fraction, split_seed)
    base = AnswerabilityAgent()
    rows = []
    for i in val_idx:
        s = qs[i]
        a = base.assess(s["question"]["question"], s["passages"], s["retriever_confidence"],
                        s["reasoning_result"])
        rows.append({"features": a.features, "gold_fully_answerable": s["question"]["expected_action"] == "ANSWER"})
    model = fit_score_model(rows, SCORE_FEATURES)
    scored = AnswerabilityAgent(score_model=model)
    v_scores = [scored.assess(qs[i]["question"]["question"], qs[i]["passages"],
                              qs[i]["retriever_confidence"], qs[i]["reasoning_result"]).answerability_score
                for i in val_idx]
    v_y = [int(qs[i]["question"]["expected_action"] == "ANSWER") for i in val_idx]
    th = select_thresholds(v_scores, v_y, max_false_accept=max_false_accept)
    return {"score_model": model, "thresholds": {"abstain_below": th["threshold"]},
            "threshold_selection": th,
            "val_ids": sorted(str(qs[i]["question"]["id"]) for i in val_idx),
            "test_ids": sorted(str(qs[i]["question"]["id"]) for i in test_idx),
            "n_val": len(val_idx), "n_test": len(test_idx), "split_seed": split_seed,
            "split_group_key": "source_document"}
