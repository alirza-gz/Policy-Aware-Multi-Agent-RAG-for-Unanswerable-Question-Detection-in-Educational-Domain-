"""Unit tests for the three-way governance decision engine (decide()).

Covers the core research behaviour: choosing between ANSWER, CLARIFY, and
ABSTAIN based on answerability, retrieval/reasoner confidence, ambiguity, and
the safety policy.
"""

import pytest

from app.agents.governance_agent import (
    GovernanceAgent,
    ACTION_ANSWER,
    ACTION_CLARIFY,
    ACTION_ABSTAIN,
)


def make_agent(**gov):
    base = {
        "answerability_enabled": True,
        "retriever_abstain_below": 0.2,
        "reasoner_abstain_below": 0.3,
        "clarify_band": [0.3, 0.5],
    }
    base.update(gov)
    return GovernanceAgent(governance=base)


class TestAnswer:
    def test_high_confidence_answerable_is_answered(self):
        agent = make_agent()
        result = agent.decide(
            {
                "answer": "Chlorophyll absorbs light.",
                "is_answerable": True,
                "confidence": 0.9,
                "needs_clarification": False,
            },
            retriever_confidence=0.8,
        )
        assert result["action"] == ACTION_ANSWER
        assert result["approved"] is True
        assert result["final_answer"] == "Chlorophyll absorbs light."


class TestAbstain:
    def test_unanswerable_abstains(self):
        agent = make_agent()
        result = agent.decide(
            {"answer": "", "is_answerable": False, "confidence": 0.0},
            retriever_confidence=0.7,
        )
        assert result["action"] == ACTION_ABSTAIN
        assert "unanswerable" in result["reason"]

    def test_low_retrieval_abstains(self):
        agent = make_agent()
        result = agent.decide(
            {"answer": "Something.", "is_answerable": True, "confidence": 0.9},
            retriever_confidence=0.05,
        )
        assert result["action"] == ACTION_ABSTAIN
        assert "retriever_below_abstain" in result["reason"]

    def test_low_reasoner_confidence_abstains(self):
        agent = make_agent()
        result = agent.decide(
            {"answer": "Maybe.", "is_answerable": True, "confidence": 0.1},
            retriever_confidence=0.8,
        )
        assert result["action"] == ACTION_ABSTAIN
        assert "reasoner_below_abstain" in result["reason"]

    def test_safety_violation_abstains(self):
        agent = GovernanceAgent(
            banned_phrases=["classified"],
            governance={"answerability_enabled": True},
        )
        result = agent.decide(
            {"answer": "This is classified information.", "is_answerable": True, "confidence": 0.9},
            retriever_confidence=0.9,
        )
        assert result["action"] == ACTION_ABSTAIN
        assert "safety_policy_violation" in result["reason"]


class TestClarify:
    def test_model_requested_clarification(self):
        agent = make_agent()
        result = agent.decide(
            {
                "answer": "",
                "is_answerable": True,
                "confidence": 0.9,
                "needs_clarification": True,
                "clarification_question": "Which subject do you mean?",
            },
            retriever_confidence=0.8,
        )
        assert result["action"] == ACTION_CLARIFY
        assert result["clarification_question"] == "Which subject do you mean?"

    def test_mid_band_confidence_clarifies(self):
        agent = make_agent()
        result = agent.decide(
            {"answer": "Possibly X.", "is_answerable": True, "confidence": 0.4},
            retriever_confidence=0.8,
        )
        assert result["action"] == ACTION_CLARIFY
        assert "clarify_band" in result["reason"]


class TestAblation:
    def test_answerability_disabled_answers_anyway(self):
        agent = make_agent(answerability_enabled=False)
        result = agent.decide(
            {"answer": "Best guess.", "is_answerable": False, "confidence": 0.0},
            retriever_confidence=0.0,
        )
        # With answerability policy off, only safety/PII apply -> it answers.
        assert result["action"] == ACTION_ANSWER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
