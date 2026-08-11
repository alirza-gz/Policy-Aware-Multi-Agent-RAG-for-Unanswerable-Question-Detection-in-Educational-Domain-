"""Run every baseline and ablation system over the labelled question set.

For each question the retriever and the reasoning agent run exactly ONCE per
run; the cached signals are then fed to every system's decision layer
(see ``eval/baselines.py``). This guarantees a strictly fair comparison:
identical retriever, embedding model, LLM, prompts, and dataset - the systems
differ only in how the final action is decided.

Reproducibility: all parameters come from ``eval/experiments.yml``; the config
is copied into the output directory, seeds are fixed per run, and one
predictions file is written per seed plus a combined file.

Usage:
    python -m eval.run_experiments
    python -m eval.run_experiments --config eval/experiments.yml --reasoning-mode mock
"""

import argparse
import asyncio
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

DEFAULT_CONFIG = "eval/experiments.yml"


def _ollama_base_url() -> str:
    """Base URL of the Ollama server, derived from config.yml's OLLAMA_URL."""
    from app.config import Config

    gen_url = os.getenv("OLLAMA_API_URL", getattr(Config, "OLLAMA_URL", "http://localhost:11434/api/generate"))
    # Strip the "/api/..." suffix to get the server root for /api/tags.
    return gen_url.split("/api/")[0] or "http://localhost:11434"


def preflight_ollama() -> None:
    """Verify the Ollama server is reachable and the configured model is pulled.

    Raises SystemExit with an actionable message instead of letting every
    question fail one-by-one inside the reasoning agent's fallback path.
    """
    import httpx
    from app.config import Config

    base = _ollama_base_url()
    model = os.getenv("OLLAMA_MODEL", getattr(Config, "OLLAMA_MODEL", "qwen2.5:3b-instruct"))
    tags_url = f"{base}/api/tags"

    try:
        resp = httpx.get(tags_url, timeout=10.0)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"[run_experiments] Ollama server not reachable at {base} ({e}).\n"
            f"  Start it with:  ollama serve\n"
            f"  Then verify:    curl {tags_url}\n"
            f"  Or switch to the mock reasoner: set reasoning_mode: mock in the config,\n"
            f"  or run: python -m eval.run_experiments --reasoning-mode mock"
        )

    installed = [m.get("name", "") for m in resp.json().get("models", [])]
    # Ollama tags are like "qwen2.5:3b-instruct"; accept an exact or prefix match.
    if not any(name == model or name.startswith(model.split(":")[0]) for name in installed):
        raise SystemExit(
            f"[run_experiments] Ollama is running but model '{model}' is not pulled.\n"
            f"  Installed models: {installed or '(none)'}\n"
            f"  Pull it with:  ollama pull {model}"
        )
    print(f"[run_experiments] Ollama preflight OK: server {base}, model '{model}' available.")


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_questions(path: Path, limit: int) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def question_category(q: Dict) -> str:
    """Category label for the per-category analysis.

    Prefers an explicit ``category`` field; otherwise answerable questions are
    labelled "answerable" and unanswerable ones "unlabeled".
    """
    cat = str(q.get("category", "") or "").strip().lower()
    if cat:
        return cat
    return "answerable" if q.get("answerable", True) else "unlabeled"


def expected_action(q: Dict) -> str:
    """Gold governance action used for the decision confusion matrix.

    answerable -> ANSWER; underspecified -> CLARIFY; other unanswerable -> ABSTAIN.
    """
    if q.get("answerable", True):
        return "ANSWER"
    if question_category(q) == "underspecified":
        return "CLARIFY"
    return "ABSTAIN"


def _retriever_confidence(passages: List[Dict]) -> float:
    return max((float(p.get("score", 0.0)) for p in passages), default=0.0)


async def collect_signals(questions: List[Dict], retriever, reasoner, top_k: int) -> List[Dict]:
    """Run retriever + reasoner once per question and cache all signals."""
    signals = []
    for i, q in enumerate(questions, 1):
        passages = retriever.retrieve(q["question"], top_k=top_k)
        reasoning_result = await reasoner.reason(q["question"], passages)
        signals.append(
            {
                "question": q,
                "passages": passages,
                "retriever_confidence": _retriever_confidence(passages),
                "reasoning_result": reasoning_result,
            }
        )
        if i % 10 == 0 or i == len(questions):
            print(f"[run_experiments] Signals collected for {i}/{len(questions)} questions")
    return signals


def apply_systems(signals: List[Dict], systems: Dict, seed: int, reasoning_mode: str = "mock") -> List[Dict]:
    """Apply every system's decision layer to the cached signals."""
    rows = []
    for s in signals:
        q = s["question"]
        rr = s["reasoning_result"]
        retr_conf = s["retriever_confidence"]
        base = {
            "id": q.get("id"),
            "seed": seed,
            "reasoning_mode": reasoning_mode,
            "question": q["question"],
            "gold_answerable": bool(q.get("answerable", True)),
            "gold_answers": q.get("gold_answers", []),
            "category": question_category(q),
            "expected_action": expected_action(q),
            "retriever_confidence": round(retr_conf, 4),
            "reasoner_confidence": round(float(rr.get("confidence", 0.0)), 4),
            "model_is_answerable": bool(rr.get("is_answerable", False)),
            "model_needs_clarification": bool(rr.get("needs_clarification", False)),
            "raw_answer": str(rr.get("answer", "") or ""),
            "n_passages": len(s["passages"]),
        }
        for name, decide in systems.items():
            row = dict(base)
            decision = decide(rr, retr_conf)
            row.update(
                {
                    "mode": name,
                    "action": decision["action"],
                    "final_answer": decision["final_answer"],
                    "reason": decision["reason"],
                }
            )
            rows.append(row)
    return rows


async def run(args) -> None:
    cfg = load_config(args.config)
    if args.reasoning_mode:
        cfg["reasoning_mode"] = args.reasoning_mode
    if args.questions:
        cfg["questions"] = args.questions

    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "experiments_config_used.yml")

    questions = load_questions(Path(cfg["questions"]), int(cfg.get("limit", 0)))
    print(f"[run_experiments] Loaded {len(questions)} questions from {cfg['questions']}")

    os.environ["REASONING_MODE"] = cfg.get("reasoning_mode", "mock")
    reasoning_mode = cfg.get("reasoning_mode", "mock")

    # Fail fast with an actionable message if Ollama is selected but unavailable,
    # rather than silently producing empty-answer fallback rows for every question.
    if reasoning_mode == "ollama":
        preflight_ollama()

    # Import after REASONING_MODE is set so the agents pick it up.
    from app.agents.retriever_agent import RetrieverAgent
    from app.agents.reasoning_agent import ReasoningAgent
    from eval.baselines import build_systems

    print("[run_experiments] Initialising agents ...")
    retriever = RetrieverAgent()
    reasoner = ReasoningAgent(mode=reasoning_mode)
    systems = build_systems(cfg.get("governance") or {})
    print(f"[run_experiments] Systems under evaluation: {list(systems)}")

    seeds = list(cfg.get("seeds", [42]))
    all_rows: List[Dict] = []
    for seed in seeds:
        print(f"[run_experiments] === Run with seed {seed} ===")
        random.seed(seed)
        np.random.seed(seed)
        ordered = list(questions)
        random.Random(seed).shuffle(ordered)

        signals = await collect_signals(ordered, retriever, reasoner, int(cfg.get("top_k", 5)))
        rows = apply_systems(signals, systems, seed, reasoning_mode)
        all_rows.extend(rows)

        run_path = out_dir / f"predictions_seed{seed}.jsonl"
        with open(run_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[run_experiments] Wrote {len(rows)} rows to {run_path}")

    combined = out_dir / cfg.get("predictions_file", "predictions_all.jsonl")
    with open(combined, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[run_experiments] Wrote combined predictions ({len(all_rows)} rows) to {combined}")


def main():
    parser = argparse.ArgumentParser(description="Run all baseline/ablation experiments.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--questions", default=None, help="Override the question file.")
    parser.add_argument("--reasoning-mode", choices=["ollama", "mock"], default=None)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
