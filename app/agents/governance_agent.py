import re
from typing import Dict, List, Optional

from app.utils.logger import get_logger
from app.config import Config

logger = get_logger("governance", "logs/governance.log")

BANNED_PHRASES = getattr(Config, "BANNED_PHRASES", [])
CONFIDENCE_THRESHOLD = getattr(Config, "CONFIDENCE_THRESHOLD", 0.5)
THRESHOLDS = getattr(Config, "THRESHOLDS", {})

# Three possible governance actions (the core of the policy-aware design).
ACTION_ANSWER = "ANSWER"
ACTION_CLARIFY = "CLARIFY"
ACTION_ABSTAIN = "ABSTAIN"

# Simple regexes for crude PII detection/redaction
RE_DATE = re.compile(r'\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b')
RE_ID = re.compile(r'\b(?:id|ssn|passport|card)[\s:]*[A-Za-z0-9-]{4,}\b', re.IGNORECASE)
# Names: this is heuristic; you can replace with NER later
RE_NAME = re.compile(r'\b([A-Z][a-z]{1,20}\s[A-Z][a-z]{1,20})\b')
# Email pattern: user@domain.com
RE_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
# Phone pattern: supports various formats
RE_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b")
# IP address pattern: IPv4 addresses
RE_IP = re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b')

# Default answerability policy (overridable via config GOVERNANCE section).
_DEFAULT_GOVERNANCE = {
    "answerability_enabled": True,
    "retriever_abstain_below": 0.2,
    "reasoner_abstain_below": 0.3,
    "clarify_band": [0.3, 0.5],
}

# Message templates surfaced to the end user for non-answer actions.
ABSTAIN_MESSAGE = (
    "I don't have enough reliable information in the course materials to answer this "
    "question, so I will refrain from answering rather than guess."
)

SERVICE_ERROR_MESSAGE = (
    "The reasoning service is temporarily unavailable. Please try again shortly."
)


class GovernanceAgent:
    def __init__(
        self,
        banned_phrases=None,
        threshold: Optional[float] = None,
        thresholds: Optional[Dict[str, float]] = None,
        governance: Optional[Dict] = None,
    ):
        self.banned_phrases = banned_phrases or BANNED_PHRASES
        self.thresholds = dict(THRESHOLDS or {})

        if thresholds:
            self.thresholds.update(thresholds)
        if threshold is not None:
            # Backwards compatibility for existing single-threshold usage
            self.thresholds["reasoner"] = threshold

        self.reasoner_threshold = self.thresholds.get("reasoner", CONFIDENCE_THRESHOLD)
        self.retriever_threshold = self.thresholds.get("retriever")

        # Answerability policy configuration.
        gov_cfg = dict(_DEFAULT_GOVERNANCE)
        cfg_from_config = getattr(Config, "GOVERNANCE", None)
        if isinstance(cfg_from_config, dict):
            gov_cfg.update(cfg_from_config)
        if governance:
            gov_cfg.update(governance)
        self.answerability_enabled = bool(gov_cfg.get("answerability_enabled", True))
        self.retriever_abstain_below = float(gov_cfg.get("retriever_abstain_below", 0.2))
        self.reasoner_abstain_below = float(gov_cfg.get("reasoner_abstain_below", 0.3))
        band = gov_cfg.get("clarify_band", [0.3, 0.5]) or [0.3, 0.5]
        self.clarify_low = float(band[0])
        self.clarify_high = float(band[1])

        logger.info(
            "GovernanceAgent initialized (answerability=%s, retriever_abstain<%.2f, "
            "reasoner_abstain<%.2f, clarify_band=[%.2f, %.2f])",
            self.answerability_enabled,
            self.retriever_abstain_below,
            self.reasoner_abstain_below,
            self.clarify_low,
            self.clarify_high,
        )

    def _check_banned_phrases(self, text: str) -> List[str]:
        matches = []
        source = text or ""
        for p in self.banned_phrases:
            try:
                pattern = r"\b" + p + r"\b" if " " not in p.strip("()") else p
                if re.search(pattern, source, flags=re.IGNORECASE):
                    matches.append(p)
            except re.error:
                if p.lower() in source.lower():
                    matches.append(p)
        return matches

    def _get_pii_filters(self) -> Dict[str, bool]:
        """Get current PII filter settings from config (respects runtime updates)."""
        filters = getattr(Config, "PII_FILTERS", None) or Config.get("PII_FILTERS")
        if not isinstance(filters, dict):
            return {
                "email": True, "phone": True, "ip": True,
                "date": True, "id": True, "name": True,
            }
        return {k: bool(v) for k, v in filters.items()}

    def _redact_pii(self, text: str) -> str:
        original_text = text
        filters = self._get_pii_filters()
        redactions = []

        if filters.get("email", True):
            email_matches = RE_EMAIL.findall(text)
            if email_matches:
                text = RE_EMAIL.sub("[REDACTED_EMAIL]", text)
                redactions.append(f"email(s): {len(email_matches)}")
                logger.info("Redacted %d email(s) from text", len(email_matches))

        if filters.get("phone", True):
            phone_matches = RE_PHONE.findall(text)
            if phone_matches:
                text = RE_PHONE.sub("[REDACTED_PHONE]", text)
                redactions.append(f"phone(s): {len(phone_matches)}")
                logger.info("Redacted %d phone number(s) from text", len(phone_matches))

        if filters.get("ip", True):
            ip_matches = RE_IP.findall(text)
            valid_ips = [ip for ip in ip_matches if self._is_valid_ip(ip)]
            if valid_ips:
                for ip in valid_ips:
                    text = text.replace(ip, "[REDACTED_IP]")
                redactions.append(f"IP address(es): {len(valid_ips)}")
                logger.info("Redacted %d IP address(es) from text", len(valid_ips))

        if filters.get("date", True):
            date_matches = RE_DATE.findall(text)
            if date_matches:
                text = RE_DATE.sub("[REDACTED_DATE]", text)
                redactions.append(f"date(s): {len(date_matches)}")

        if filters.get("id", True):
            id_matches = RE_ID.findall(text)
            if id_matches:
                text = RE_ID.sub("[REDACTED_ID]", text)
                redactions.append(f"ID(s): {len(id_matches)}")

        if filters.get("name", True):
            name_matches = RE_NAME.findall(text)
            if name_matches:
                text = RE_NAME.sub("[REDACTED_NAME]", text)
                redactions.append(f"name(s): {len(name_matches)}")

        if redactions and text != original_text:
            logger.info("PII redaction summary: %s", ", ".join(redactions))

        return text

    def _is_valid_ip(self, ip_str: str) -> bool:
        """Validate if a string is a valid IP address."""
        parts = ip_str.split('.')
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False

    def decide(
        self,
        reasoning_result: Dict,
        retriever_confidence: Optional[float] = None,
    ) -> Dict:
        """Core policy-aware decision: return ANSWER, CLARIFY, or ABSTAIN.

        Precedence:
          1. Safety policy (banned phrases) -> ABSTAIN (blocked).
          2. Answerability (no evidence / low retrieval / low reasoner conf) -> ABSTAIN.
          3. Ambiguity or mid-band confidence -> CLARIFY.
          4. Otherwise -> ANSWER (with PII redaction).
        """
        answer = str(reasoning_result.get("answer", "") or "")
        is_answerable = bool(reasoning_result.get("is_answerable", bool(answer.strip())))
        needs_clarification = bool(reasoning_result.get("needs_clarification", False))
        clarification_question = str(reasoning_result.get("clarification_question", "") or "").strip()
        try:
            confidence = float(reasoning_result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        reasons: List[str] = []

        # --- 0. Backend/service failure (distinct from a normal abstain) -------
        if reasoning_result.get("service_error"):
            reasons.append("reasoning_service_unavailable")
            logger.info("Governance decision=ABSTAIN reason=%s", reasons)
            return self._result(
                ACTION_ABSTAIN,
                final_answer=SERVICE_ERROR_MESSAGE,
                redacted_answer=SERVICE_ERROR_MESSAGE,
                reasons=reasons,
                clarification_question="",
            )

        # --- 1. Safety policy ---------------------------------------------------
        banned = self._check_banned_phrases(answer)
        if banned:
            reasons.append(f"safety_policy_violation: {banned}")
            logger.info("Governance decision=ABSTAIN reason=%s", reasons)
            return self._result(
                ACTION_ABSTAIN,
                final_answer=ABSTAIN_MESSAGE,
                redacted_answer=ABSTAIN_MESSAGE,
                reasons=reasons,
                clarification_question="",
            )

        # --- 2. Answerability policy -------------------------------------------
        if self.answerability_enabled:
            if not is_answerable:
                reasons.append("unanswerable_no_supporting_evidence")
            if (
                retriever_confidence is not None
                and retriever_confidence < self.retriever_abstain_below
            ):
                reasons.append(
                    f"retriever_below_abstain ({retriever_confidence:.2f} < {self.retriever_abstain_below})"
                )
            if is_answerable and confidence < self.reasoner_abstain_below:
                reasons.append(
                    f"reasoner_below_abstain ({confidence:.2f} < {self.reasoner_abstain_below})"
                )

            if reasons:
                logger.info("Governance decision=ABSTAIN reason=%s", reasons)
                return self._result(
                    ACTION_ABSTAIN,
                    final_answer=ABSTAIN_MESSAGE,
                    redacted_answer=ABSTAIN_MESSAGE,
                    reasons=reasons,
                    clarification_question="",
                )

            # --- 3. Clarification policy ---------------------------------------
            in_clarify_band = self.clarify_low <= confidence < self.clarify_high
            if needs_clarification or in_clarify_band:
                if needs_clarification:
                    reasons.append("model_requested_clarification")
                if in_clarify_band:
                    reasons.append(
                        f"confidence_in_clarify_band ({confidence:.2f} in "
                        f"[{self.clarify_low}, {self.clarify_high}))"
                    )
                if not clarification_question:
                    clarification_question = (
                        "Could you please clarify or add more detail to your question?"
                    )
                logger.info("Governance decision=CLARIFY reason=%s", reasons)
                return self._result(
                    ACTION_CLARIFY,
                    final_answer=clarification_question,
                    redacted_answer=clarification_question,
                    reasons=reasons,
                    clarification_question=clarification_question,
                )

        # --- 4. Answer path (with PII redaction) --------------------------------
        redacted_answer = self._redact_pii(answer)
        if redacted_answer != answer:
            reasons.append("pii_redacted")
        reasons.append("approved") if not reasons else None
        logger.info("Governance decision=ANSWER reason=%s", reasons or ["approved"])
        return self._result(
            ACTION_ANSWER,
            final_answer=redacted_answer,
            redacted_answer=redacted_answer,
            reasons=reasons or ["approved"],
            clarification_question="",
        )

    @staticmethod
    def _result(action, final_answer, redacted_answer, reasons, clarification_question) -> Dict:
        reason = "; ".join(reasons) if reasons else "approved"
        return {
            "action": action,
            "approved": action == ACTION_ANSWER,
            "reason": reason,
            "reasons": reasons,
            "final_answer": final_answer,
            "redacted_answer": redacted_answer,
            "clarification_question": clarification_question,
        }

    def evaluate(
        self,
        answer: str,
        trace: list,
        confidence: float,
        retriever_confidence: Optional[float] = None,
    ) -> Dict:
        """Backward-compatible wrapper around the older approve/redact behaviour.

        Retained so existing tests and callers using the two-way (approved) API
        keep working. New callers should use decide().
        """
        reasons = []
        approved = True
        if confidence is None:
            confidence = 0.0
        if confidence < self.reasoner_threshold:
            approved = False
            reasons.append(
                f"reasoner_low_confidence ({confidence:.2f} < {self.reasoner_threshold})"
            )

        if (
            retriever_confidence is not None
            and self.retriever_threshold is not None
            and retriever_confidence < self.retriever_threshold
        ):
            approved = False
            reasons.append(
                f"retriever_low_confidence ({retriever_confidence:.2f} < {self.retriever_threshold})"
            )

        banned = self._check_banned_phrases(answer)
        if banned:
            approved = False
            reasons.append(f"banned_phrases_present: {banned}")
        redacted_answer = self._redact_pii(answer)
        if redacted_answer != answer:
            reasons.append("pii_redacted")
        reason = "; ".join(reasons) if reasons else "approved"
        logger.info(
            "Governance evaluated: approved=%s reason=%s (thresholds=%s)",
            approved,
            reason,
            {
                "reasoner": self.reasoner_threshold,
                "retriever": self.retriever_threshold,
            },
        )
        return {"approved": approved, "reason": reason, "redacted_answer": redacted_answer}
