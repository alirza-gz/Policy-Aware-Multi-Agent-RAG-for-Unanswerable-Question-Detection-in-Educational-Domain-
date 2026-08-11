import asyncio
import os
import httpx
from typing import List, Dict
from app.utils.logger import get_logger
import json
import re
from app.config import Config

logger = get_logger("reasoning", "logs/reasoning.log")

OLLAMA_URL = os.getenv("OLLAMA_API_URL", Config.OLLAMA_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", Config.OLLAMA_MODEL)
OLLAMA_MAX_RETRIES = 3
OLLAMA_BACKOFF_BASE = 0.5  # seconds
OLLAMA_TIMEOUT = httpx.Timeout(30.0, read=300.0)

# When set (env REASONING_MODE=mock or config REASONING_MODE), the agent answers
# with a lightweight lexical heuristic instead of calling Ollama. This keeps the
# full pipeline and evaluation harness runnable on machines without a capable LLM.
REASONING_MODE = os.getenv("REASONING_MODE", getattr(Config, "REASONING_MODE", "ollama")).lower()

# Sentinel string the LLM is instructed to emit when the passages do not contain
# the answer. Detecting it lets us map to is_answerable=False deterministically.
UNANSWERABLE_TOKEN = "UNANSWERABLE"


class ReasoningAgent:
    def __init__(self, ollama_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, mode: str = REASONING_MODE):
        self.ollama_url = ollama_url
        self.model = model
        self.mode = (mode or "ollama").lower()
        self.user_instructions = self._load_user_instructions()

        logger.info(
            "ReasoningAgent initialized (mode=%s, model=%s, url=%s)",
            self.mode,
            model,
            ollama_url,
        )

    def _build_prompt(self, query: str, passages: List[Dict]) -> str:
        context = "\n\n".join([
            f"[PASSAGE {i}] (score={p['score']:.3f})\n{p['text']}"
            for i, p in enumerate(passages)
        ]) or "(no passages retrieved)"

        prompt = (
            "You are a careful educational assistant for a Retrieval-Augmented Generation system.\n"
            "Answer the student's question STRICTLY using the retrieved passages below.\n"
            "You must NEVER guess, hallucinate, or use outside knowledge.\n\n"
            f"Question: {query}\n\n"
            "Retrieved passages:\n"
            f"{context}\n\n"
            "Additional user instructions:\n"
            f"{self.user_instructions}\n\n"
            "Decision rules:\n"
            f"1) If the passages do NOT contain enough information to answer, set "
            f"\"is_answerable\": false and \"answer\": \"{UNANSWERABLE_TOKEN}\".\n"
            "2) If the question is ambiguous or underspecified (it could be interpreted in\n"
            "   multiple substantially different ways), set \"needs_clarification\": true and\n"
            "   provide a single focused \"clarification_question\".\n"
            "3) Otherwise provide a concise, grounded answer.\n\n"
            "Output requirements (return ONLY a JSON object, no markdown):\n"
            "{\n"
            '  "answer": string,\n'
            '  "is_answerable": boolean,\n'
            '  "answerability_confidence": float (0..1, how sure you are the passages support an answer),\n'
            '  "needs_clarification": boolean,\n'
            '  "clarification_question": string (empty if none),\n'
            '  "trace": [{"index": int, "note": string}],\n'
            '  "confidence": float (0..1, confidence in the answer itself)\n'
            "}\n"
        )

        logger.info("Built prompt (%d chars, %d passages)", len(prompt), len(passages))
        return prompt

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama with retry, exponential backoff, and streaming enabled."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.0,
            "stream": True,
        }

        last_exception = None

        for attempt in range(OLLAMA_MAX_RETRIES):
            try:
                logger.info(
                    "Calling Ollama (attempt %d/%d) at %s",
                    attempt + 1,
                    OLLAMA_MAX_RETRIES,
                    self.ollama_url,
                )

                response_text = ""

                async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                    async with client.stream("POST", self.ollama_url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    response_text += data["response"]
                                if data.get("done"):
                                    break
                            except Exception as e:
                                logger.warning("Failed to parse streaming chunk: %s", e)

                logger.info("Raw LLM output length: %d", len(response_text))
                return response_text

            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s",
                    attempt + 1,
                    OLLAMA_MAX_RETRIES,
                    e,
                )

                if attempt < OLLAMA_MAX_RETRIES - 1:
                    backoff = OLLAMA_BACKOFF_BASE * (2 ** attempt)
                    logger.info("Retrying after %.2f seconds", backoff)
                    await asyncio.sleep(backoff)

        logger.error("Ollama unavailable after retries")
        raise RuntimeError("Ollama unavailable") from last_exception

    def _normalize_result(self, parsed: Dict, passages: List[Dict]) -> Dict:
        """Coerce a raw parsed dict into the canonical answerability schema."""
        answer = str(parsed.get("answer", "") or "").strip()

        # Infer answerability from an explicit flag, the sentinel token, or an empty answer.
        is_answerable = parsed.get("is_answerable")
        token_present = UNANSWERABLE_TOKEN.lower() in answer.lower()
        if is_answerable is None:
            is_answerable = bool(answer) and not token_present
        else:
            is_answerable = bool(is_answerable) and not token_present

        if token_present:
            answer = ""

        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        answerability_confidence = parsed.get("answerability_confidence")
        try:
            answerability_confidence = float(answerability_confidence)
        except (TypeError, ValueError):
            # Fall back to the answer confidence when the model omitted the field.
            answerability_confidence = confidence if is_answerable else 1.0 - confidence
        answerability_confidence = max(0.0, min(1.0, answerability_confidence))

        needs_clarification = bool(parsed.get("needs_clarification", False))
        clarification_question = str(parsed.get("clarification_question", "") or "").strip()
        if needs_clarification and not clarification_question:
            clarification_question = "Could you please clarify or add more detail to your question?"

        trace = parsed.get("trace", [])
        if not isinstance(trace, list):
            trace = []

        if not is_answerable:
            confidence = 0.0

        return {
            "answer": answer,
            "is_answerable": is_answerable,
            "answerability_confidence": answerability_confidence,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "trace": trace,
            "confidence": confidence,
        }

    def _parse_llm_output(self, raw_text: str) -> Dict:
        """Parse JSON from model output. Fallback to wrapping raw text if parsing fails."""
        try:
            return json.loads(raw_text)
        except Exception:
            m = re.search(r"\{.*\}", raw_text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
            return {
                "answer": raw_text.strip(),
                "trace": [],
                "confidence": Config.CONFIDENCE_THRESHOLD,
            }

    def _mock_reason(self, query: str, passages: List[Dict]) -> Dict:
        """Deterministic lexical heuristic used when no LLM is available.

        Grounds "answerability" purely in retrieval overlap: if a passage shares
        enough content words with the question, the question is treated as
        answerable and the best passage is returned as the answer. Otherwise the
        question is flagged unanswerable. This mirrors the real decision signals
        (retrieval evidence) closely enough to exercise the full pipeline.
        """
        stop = {
            "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
            "are", "was", "were", "what", "which", "who", "whom", "when", "where",
            "why", "how", "does", "do", "did", "with", "that", "this", "these",
            "those", "by", "as", "at", "be", "it", "its", "from", "into", "about",
        }
        q_words = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop and len(w) > 2}

        best_score = 0.0
        best_idx = -1
        best_overlap = 0
        for i, p in enumerate(passages):
            p_words = set(re.findall(r"[a-z0-9]+", (p.get("text", "") or "").lower()))
            overlap = len(q_words & p_words)
            retrieval_score = float(p.get("score", 0.0))
            combined = overlap + retrieval_score
            if combined > best_score:
                best_score = combined
                best_idx = i
                best_overlap = overlap

        coverage = (best_overlap / len(q_words)) if q_words else 0.0
        retriever_top = max((float(p.get("score", 0.0)) for p in passages), default=0.0)

        # Ambiguous / underspecified: extremely short question with no clear focus.
        needs_clarification = len(q_words) <= 1 and retriever_top < 0.5

        if best_idx < 0 or coverage < 0.34 or retriever_top < 0.2:
            return self._normalize_result(
                {
                    "answer": UNANSWERABLE_TOKEN,
                    "is_answerable": False,
                    "answerability_confidence": round(1.0 - min(coverage, 1.0), 3),
                    "needs_clarification": needs_clarification,
                    "clarification_question": (
                        "Could you please clarify what specifically you are asking about?"
                        if needs_clarification else ""
                    ),
                    "trace": [],
                    "confidence": 0.0,
                },
                passages,
            )

        best_passage = passages[best_idx]
        answer_text = (best_passage.get("text", "") or "").strip()
        answer_text = answer_text[:400]
        return self._normalize_result(
            {
                "answer": answer_text,
                "is_answerable": True,
                "answerability_confidence": round(min(coverage, 1.0), 3),
                "needs_clarification": False,
                "clarification_question": "",
                "trace": [{"index": best_idx, "note": f"lexical overlap={best_overlap}"}],
                "confidence": round(min(0.4 + coverage / 2 + retriever_top / 4, 1.0), 3),
            },
            passages,
        )

    async def reason(self, query: str, passages: List[Dict]) -> Dict:
        if self.mode == "mock":
            result = self._mock_reason(query, passages)
            logger.info("Reasoning (mock) completed for query '%s'", query)
            return result

        prompt = self._build_prompt(query, passages)
        try:
            raw = await self._call_ollama(prompt)
            parsed = self._parse_llm_output(raw)
            result = self._normalize_result(parsed, passages)
            logger.info("Reasoning completed for query '%s'", query)
            return result
        except Exception as e:
            logger.exception("ReasoningAgent fallback activated: %s", e)
            return {
                "answer": "",
                "is_answerable": False,
                "answerability_confidence": 0.0,
                "needs_clarification": False,
                "clarification_question": "",
                "trace": [],
                "confidence": 0.0,
            }

    def _load_user_instructions(self) -> str:
        """Load user-defined instructions from data/instructions.txt. Empty on failure."""
        instructions_path = os.path.join("data", "instructions.txt")
        try:
            with open(instructions_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                logger.info("Loaded user instructions from %s", instructions_path)
                return content
        except Exception as e:
            logger.error("Failed to load instructions.txt: %s", e)
            return ""
