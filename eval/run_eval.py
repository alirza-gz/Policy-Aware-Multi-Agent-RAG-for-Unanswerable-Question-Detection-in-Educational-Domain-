"""Run the multi-agent pipeline over the labelled question set.

For every question the retriever and reasoning agents run once (their output is
identical regardless of governance), then two governance conditions are recorded:

  * policy_aware : the full system, governor.decide() -> ANSWER / CLARIFY / ABSTAIN
  * baseline     : governance disabled, the raw reasoner answer is always returned

Predictions are written to a JSONL file (one row per question per mode) that
``metrics.py`` and ``analyze.py`` consume. The agents run in-process, so the
FastAPI server does not need to be running.

Usage:
    python -m eval.run_eval --questions data/eval/questions.jsonl --limit 50
    python -m eval.run_eval --reasoning-mode mock --out results/predictions.jsonl
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List

from app.agents.retriever_agent import RetrieverAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.governance_agent import (
    GovernanceAgent,
    ACTION_ANSWER,
)


def load_questions(path: Path, limit: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _retriever_confidence(passages: List[Dict]) -> float:
    if not passages:
        return 0.0
    return max((float(p.get("score", 0.0)) for p in passages), default=0.0)


async def evaluate_question(
    q: Dict,
    retriever: RetrieverAgent,
    reasoner: ReasoningAgent,
    governor: GovernanceAgent,
    top_k: int,
) -> List[Dict]:
    question = q["question"]
    passages = retriever.retrieve(question, top_k=top_k)
    retr_conf = _retriever_confidence(passages)

    reasoning_result = await reasoner.reason(question, passages)
    reasoner_conf = float(reasoning_result.get("confidence", 0.0))
    model_answerable = bool(reasoning_result.get("is_answerable", False))

    base = {
        "id": q.get("id"),
        "question": question,
        "gold_answerable": bool(q.get("answerable", True)),
        "gold_answers": q.get("gold_answers", []),
        "retriever_confidence": round(retr_conf, 4),
        "reasoner_confidence": round(reasoner_conf, 4),
        "model_is_answerable": model_answerable,
        "trace": reasoning_result.get("trace", []),
        "n_passages": len(passages),
    }

    # policy_aware: full governance decision.
    decision = governor.decide(reasoning_result, retriever_confidence=retr_conf)
    policy_row = dict(base)
    policy_row.update(
        {
            "mode": "policy_aware",
            "action": decision["action"],
            "final_answer": decision["final_answer"],
            "reason": decision["reason"],
        }
    )

    # baseline: governance disabled -> always answer with the raw model output.
    baseline_row = dict(base)
    baseline_row.update(
        {
            "mode": "baseline",
            "action": ACTION_ANSWER,
            "final_answer": reasoning_result.get("answer", ""),
            "reason": "governance_disabled_baseline",
        }
    )

    return [policy_row, baseline_row]


async def run(args) -> None:
    questions = load_questions(Path(args.questions), args.limit)
    print(f"[run_eval] Loaded {len(questions)} questions from {args.questions}")

    # Allow overriding the reasoning mode from the CLI (e.g. force 'mock').
    if args.reasoning_mode:
        os.environ["REASONING_MODE"] = args.reasoning_mode

    print("[run_eval] Initialising agents ...")
    retriever = RetrieverAgent()
    reasoner = ReasoningAgent(mode=args.reasoning_mode) if args.reasoning_mode else ReasoningAgent()
    governor = GovernanceAgent()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    with open(out_path, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            try:
                rows = await evaluate_question(q, retriever, reasoner, governor, args.top_k)
            except Exception as e:  # noqa: BLE001
                print(f"[run_eval] Question {q.get('id')} failed: {e}")
                continue
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                all_rows.append(row)
            if i % 10 == 0 or i == len(questions):
                print(f"[run_eval] Processed {i}/{len(questions)} questions")

    print(f"[run_eval] Wrote {len(all_rows)} prediction rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline over the question set.")
    parser.add_argument("--questions", default="data/eval/questions.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0 = all).")
    parser.add_argument(
        "--reasoning-mode",
        choices=["ollama", "mock"],
        default=None,
        help="Override REASONING_MODE for this run.",
    )
    parser.add_argument("--out", default="results/predictions.jsonl")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
