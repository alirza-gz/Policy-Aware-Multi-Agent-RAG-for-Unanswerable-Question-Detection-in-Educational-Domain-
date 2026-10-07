"""Answerability Agent: evidence sufficiency + five-way answerability label.

Pipeline position (see docs/ARCHITECTURE_DIAGRAMS.md):
    Retrieval -> Rerank -> [Evidence Sufficiency -> Answerability] -> Policy -> Decision

Labels
------
fully_answerable | partially_answerable | unanswerable | out_of_domain | false_premise

What this agent uses (inputs only -- NEVER gold labels):
  * the question text (multi-part split, ambiguity heuristics)
  * retrieved passage texts and retrieval confidence (dense score, optional rerank score)
  * optional signals from the reasoner (its own label / flags)
  * optional NLI callable for false-premise detection

Honest limitations (documented, not hidden):
  * Lexical coverage is a proxy for evidence sufficiency, not entailment.
  * ``false_premise`` is only emitted when an NLI callable or the reasoner supplies
    it; lexical features cannot detect a false presupposition.
  * Multi-part detection is a surface heuristic (several ``?`` clauses, or
    interrogative conjunctions). It happens to align with how the
    ``partially_answerable`` category is constructed; generalisation to other
    phrasing is NOT established.

``answerability_score`` is P(fully answerable) from a small logistic model over
the features below when a fitted ``ScoreModel`` is supplied (fit on a
validation split by ``eval.calibration``); otherwise a documented heuristic is
returned with ``score_kind == "heuristic_uncalibrated"``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

LABEL_FULL = "fully_answerable"
LABEL_PARTIAL = "partially_answerable"
LABEL_UNANSWERABLE = "unanswerable"
LABEL_OOD = "out_of_domain"
LABEL_FALSE_PREMISE = "false_premise"
LABELS = (LABEL_FULL, LABEL_PARTIAL, LABEL_UNANSWERABLE, LABEL_OOD, LABEL_FALSE_PREMISE)

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was",
    "were", "what", "which", "who", "whom", "when", "where", "why", "how", "does", "do",
    "did", "with", "that", "this", "these", "those", "by", "as", "at", "be", "it", "its",
    "from", "into", "about", "can", "could", "would", "should", "than", "then", "there",
    "their", "they", "you", "your", "i", "me", "my", "we", "our", "if", "so", "such",
    "also", "used", "use", "using", "between", "within", "over", "under", "has", "have",
    "had", "not", "no", "any", "some", "more", "most", "many", "much",
}
_TOKEN = re.compile(r"[a-z0-9]+")
_INTERROG = ("what", "how", "why", "when", "where", "which", "who", "whom", "does", "do",
             "is", "are", "can", "could", "should")
_PRONOUN_START = re.compile(r"^\s*(it|this|that|they|these|those|he|she)\b", re.I)


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "es", "s", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_terms(text: str) -> List[str]:
    return [_stem(w) for w in _TOKEN.findall((text or "").lower())
            if w not in _STOP and len(w) > 2]


def split_question_parts(question: str) -> List[str]:
    """Split a question into sub-questions (generic surface heuristic)."""
    q = (question or "").strip()
    if not q:
        return []
    chunks = [c.strip() for c in re.split(r"(?<=\?)\s+", q) if c.strip()]
    parts: List[str] = []
    for c in chunks:
        # "X, and how/what ...": interrogative conjunction opening a second question.
        m = re.split(r"[,;]?\s+(?:and|also|then)\s+(?=(?:%s)\b)" % "|".join(_INTERROG), c, flags=re.I)
        parts.extend(p.strip() for p in m if p.strip())
    # A fragment with no content words is not a sub-question; merge it back.
    merged: List[str] = []
    for p in parts:
        if merged and len(content_terms(p)) == 0:
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged or [q]


@dataclass
class ScoreModel:
    """Logistic model P(fully answerable | features). Fit by eval.calibration."""
    feature_names: List[str]
    weights: List[float]
    bias: float
    mean: List[float] = field(default_factory=list)
    scale: List[float] = field(default_factory=list)
    kind: str = "logistic_fitted_on_validation"

    def predict(self, feats: Dict[str, float]) -> float:
        z = self.bias
        for i, name in enumerate(self.feature_names):
            x = float(feats.get(name, 0.0))
            if self.mean:
                x = (x - self.mean[i]) / (self.scale[i] or 1.0)
            z += self.weights[i] * x
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "ScoreModel":
        return ScoreModel(**json.loads(s))


SCORE_FEATURES = ["retrieval_confidence", "min_part_coverage", "mean_part_coverage",
                  "best_passage_coverage", "n_parts_norm", "model_says_answerable"]


@dataclass
class AnswerabilityAssessment:
    answerability_label: str
    answerability_score: float          # P(fully answerable) or documented heuristic
    score_kind: str
    evidence_sufficient: bool
    retrieval_confidence: float
    part_coverages: List[float]
    parts: List[str]
    answerable_parts: List[str]
    missing_parts: List[str]
    uncovered_terms: List[str]
    ambiguous: bool
    ambiguity_reason: str
    features: Dict[str, float]
    label_source: str                   # "features" | "reasoner" | "nli"

    def as_log(self) -> Dict:
        d = asdict(self)
        return d


DEFAULT_THRESHOLDS = {
    # Selected on a VALIDATION split by eval.calibration.select_thresholds; these
    # defaults are placeholders used only when no validated values are supplied.
    "part_covered": 0.5,        # a sub-question is "covered" if >= this fraction of its terms appear
    "ood_retrieval": 0.2,       # max dense score below this AND coverage tiny => out_of_domain
    "ood_coverage": 0.2,
    "evidence_retrieval": 0.2,  # minimum retrieval confidence for sufficiency
}


class AnswerabilityAgent:
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        score_model: Optional[ScoreModel] = None,
        nli_contradiction: Optional[Callable[[str, str], float]] = None,
        nli_threshold: float = 0.5,
    ):
        self.t = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.t.update(thresholds)
        self.score_model = score_model
        self.nli = nli_contradiction
        self.nli_threshold = nli_threshold

    # ---- features ---------------------------------------------------------
    @staticmethod
    def _coverage(part: str, passage_tokens: List[set], union: set) -> float:
        terms = content_terms(part)
        if not terms:
            return 0.0
        return sum(1 for t in terms if t in union) / len(terms)

    def _ambiguity(self, question: str) -> (bool, str):
        n_terms = len(content_terms(question))
        if n_terms <= 1:
            return True, "question has too few content words to identify a topic"
        if _PRONOUN_START.match(question or "") and n_terms <= 3:
            return True, "question starts with an unresolved pronoun/demonstrative"
        return False, ""

    def assess(
        self,
        question: str,
        passages: Sequence[Dict],
        retrieval_confidence: float,
        reasoner: Optional[Dict] = None,
    ) -> AnswerabilityAssessment:
        reasoner = reasoner or {}
        texts = [str(p.get("text", "") or "") for p in passages]
        tokens = [set(content_terms(t)) for t in texts]
        union = set().union(*tokens) if tokens else set()
        parts = split_question_parts(question)
        covs = [self._coverage(p, tokens, union) for p in parts]
        best_single = max(
            (self._coverage(question, tokens, tk) for tk in tokens), default=0.0
        )
        tau = self.t["part_covered"]
        covered = [c >= tau for c in covs]
        answerable_parts = [p for p, ok in zip(parts, covered) if ok]
        missing_parts = [p for p, ok in zip(parts, covered) if not ok]
        q_terms = content_terms(question)
        uncovered_terms = sorted({t for t in q_terms if t not in union})
        ambiguous, amb_reason = self._ambiguity(question)
        ret = float(retrieval_confidence or 0.0)

        feats = {
            "retrieval_confidence": ret,
            "min_part_coverage": min(covs) if covs else 0.0,
            "mean_part_coverage": (sum(covs) / len(covs)) if covs else 0.0,
            "best_passage_coverage": best_single,
            "n_parts_norm": min(len(parts), 4) / 4.0,
            "model_says_answerable": 1.0 if reasoner.get("is_answerable") else 0.0,
        }
        evidence_sufficient = bool(passages) and ret >= self.t["evidence_retrieval"] and all(covered)

        label, source = self._label(question, parts, covered, covs, ret, texts, reasoner, ambiguous)
        if self.score_model is not None:
            score, kind = self.score_model.predict(feats), self.score_model.kind
        else:
            score = 0.5 * feats["min_part_coverage"] + 0.3 * min(ret, 1.0) + 0.2 * feats["model_says_answerable"]
            kind = "heuristic_uncalibrated"
        return AnswerabilityAssessment(
            answerability_label=label,
            answerability_score=round(float(score), 4),
            score_kind=kind,
            evidence_sufficient=evidence_sufficient,
            retrieval_confidence=round(ret, 4),
            part_coverages=[round(c, 3) for c in covs],
            parts=parts,
            answerable_parts=answerable_parts,
            missing_parts=missing_parts,
            uncovered_terms=uncovered_terms,
            ambiguous=ambiguous,
            ambiguity_reason=amb_reason,
            features=feats,
            label_source=source,
        )

    def _label(self, question, parts, covered, covs, ret, texts, reasoner, ambiguous):
        # False premise: only from NLI or an explicit reasoner label.
        if self.nli is not None and texts:
            try:
                if max(self.nli(question, t) for t in texts[:3]) >= self.nli_threshold:
                    return LABEL_FALSE_PREMISE, "nli"
            except Exception:  # noqa: BLE001
                pass
        if reasoner.get("answerability_label") == LABEL_FALSE_PREMISE:
            return LABEL_FALSE_PREMISE, "reasoner"
        mean_cov = sum(covs) / len(covs) if covs else 0.0
        if not texts or (ret < self.t["ood_retrieval"] and mean_cov < self.t["ood_coverage"]):
            return LABEL_OOD, "features"
        if ambiguous:
            return LABEL_PARTIAL, "features"
        if all(covered):
            if reasoner.get("needs_clarification"):
                return LABEL_PARTIAL, "reasoner"
            # Reasoner veto: model reports the passages do not contain the answer.
            if reasoner.get("is_answerable") is False and "is_answerable" in reasoner:
                return LABEL_UNANSWERABLE, "reasoner"
            return LABEL_FULL, "features"
        if any(covered) and len(parts) > 1:
            return LABEL_PARTIAL, "features"
        return LABEL_UNANSWERABLE, "features"

    # ---- clarification ----------------------------------------------------
    def build_clarification(self, a: AnswerabilityAssessment) -> Dict:
        """Specific clarification: what is missing / ambiguous / what to provide."""
        if a.ambiguous:
            q = ("Your question is too short or unclear for me to identify the topic "
                 f"({a.ambiguity_reason}). Which concept or course topic do you mean, "
                 "and what exactly do you want to know about it?")
            return {"clarification_trigger": "ambiguity",
                    "clarification_reason": a.ambiguity_reason,
                    "missing_information": "topic or concept being asked about",
                    "clarification_question": q}
        if a.missing_parts and a.answerable_parts:
            cov = "; ".join(f'"{p}"' for p in a.answerable_parts)
            miss = "; ".join(f'"{p}"' for p in a.missing_parts)
            q = (f"The course materials support an answer to: {cov}. They do not contain "
                 f"enough information for: {miss}. Would you like me to answer only the "
                 "supported part, or can you rephrase the unsupported part or point me to "
                 "the material it refers to?")
            return {"clarification_trigger": "partial_coverage",
                    "clarification_reason": f"{len(a.missing_parts)} of {len(a.parts)} sub-questions lack supporting evidence",
                    "missing_information": miss,
                    "clarification_question": q}
        return {"clarification_trigger": "none", "clarification_reason": "",
                "missing_information": "", "clarification_question": ""}


def is_generic_clarification(text: str) -> bool:
    """True for non-specific clarification such as 'Could you clarify?'."""
    t = (text or "").strip().lower()
    if len(t.split()) < 8:
        return True
    generic = ("could you please clarify or add more detail", "could you clarify",
               "please clarify", "can you clarify")
    return any(t.startswith(g) for g in generic) and len(t.split()) < 16
