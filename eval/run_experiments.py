"""Run every baseline and ablation over the labelled educational dataset.

Retrieval profiles
------------------
Systems that share (retrieval_mode, use_reranker) share one retriever+reasoner
pass so comparisons remain fair within each profile. Profiles that differ
(e.g. no_reranker, llm_no_rag, no_hybrid) collect separate signals.

Reproducibility: parameters come from ``eval/experiments.yml`` (copied to
results/); seeds fixed per run.

Usage:
    python -m eval.run_experiments
    python -m eval.run_experiments --reasoning-mode mock --limit 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

DEFAULT_CONFIG = "eval/experiments.yml"


def _ollama_base_url() -> str:
    from app.config import Config

    gen_url = os.getenv(
        "OLLAMA_API_URL",
        getattr(Config, "OLLAMA_URL", "http://localhost:11434/api/generate"),
    )
    return gen_url.split("/api/")[0] or "http://localhost:11434"


def preflight_ollama() -> None:
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
            f"  Or run with: --reasoning-mode mock"
        )
    installed = [m.get("name", "") for m in resp.json().get("models", [])]
    if not any(name == model or name.startswith(model.split(":")[0]) for name in installed):
        raise SystemExit(
            f"[run_experiments] Ollama model '{model}' not pulled. Installed: {installed}"
        )
    print(f"[run_experiments] Ollama preflight OK: {base}, model '{model}'")


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
    cat = str(q.get("category", "") or "").strip().lower()
    if cat:
        return cat
    return "answerable" if q.get("answerable", True) else "unlabeled"


def expected_action(q: Dict) -> str:
    explicit = str(q.get("expected_action", "") or "").strip().upper()
    if explicit in ("ANSWER", "CLARIFY", "ABSTAIN"):
        return explicit
    if q.get("gold_answerable", q.get("answerable", True)):
        return "ANSWER"
    if question_category(q) in ("underspecified", "partially_answerable"):
        return "CLARIFY"
    return "ABSTAIN"


def _retriever_confidence(passages: List[Dict]) -> float:
    """Governance-calibrated retrieval confidence in roughly [0, 1].

    Prefer per-passage ``dense_score`` (FAISS IP) so hybrid RRF / cross-encoder
    score scales do not silently invalidate ``retriever_abstain_below`` and the
    RAG+Threshold baseline. Falls back to ``score`` when dense_score is absent
    (e.g. dense-only retrieval before annotation).
    """
    if not passages:
        return 0.0
    dense_vals = [
        float(p["dense_score"])
        for p in passages
        if p.get("dense_score") is not None
    ]
    if dense_vals:
        return max(dense_vals)
    return max((float(p.get("score", 0.0)) for p in passages), default=0.0)


def unique_profiles(system_names: List[str], profiles: Dict) -> List[Tuple[str, bool]]:
    seen = []
    for name in system_names:
        prof = profiles[name]
        if prof not in seen:
            seen.append(prof)
    return seen


async def collect_signals_for_profile(
    questions: List[Dict],
    retriever,
    reasoner,
    top_k: int,
    retrieval_mode: str,
    use_reranker: bool,
) -> List[Dict]:
    """Run retrieval(+optional rerank)+reasoner once per question for one profile."""
    signals = []
    for i, q in enumerate(questions, 1):
        if retrieval_mode == "none":
            pipe = {
                "passages": [],
                "dense": [],
                "sparse": [],
                "fused": [],
                "retrieval_mode": "none",
                "reranked": False,
                "retrieved_ids": [],
            }
            # Reason with empty passages (LLM-without-RAG).
            reasoning_result = await reasoner.reason(q["question"], [])
        else:
            pipe = retriever.retrieve_pipeline(
                q["question"],
                top_k=top_k,
                mode=retrieval_mode,
                use_reranker=use_reranker,
            )
            reasoning_result = await reasoner.reason(q["question"], pipe["passages"])

        signals.append(
            {
                "question": q,
                "passages": pipe["passages"],
                "dense": pipe.get("dense", []),
                "sparse": pipe.get("sparse", []),
                "fused": pipe.get("fused", []),
                "retrieved_ids": pipe.get("retrieved_ids", []),
                "retrieval_mode": pipe.get("retrieval_mode", retrieval_mode),
                "reranked": bool(pipe.get("reranked", False)),
                "retriever_confidence": _retriever_confidence(pipe["passages"]),
                "reasoning_result": reasoning_result,
            }
        )
        if i % 10 == 0 or i == len(questions):
            print(
                f"[run_experiments] profile=({retrieval_mode},rerank={use_reranker}) "
                f"{i}/{len(questions)}"
            )
    return signals


def apply_systems_for_profile(
    signals: List[Dict],
    systems: Dict,
    system_names: List[str],
    seed: int,
    reasoning_mode: str,
    rag_threshold: float,
    embed_model: str,
    llm_model: str,
) -> List[Dict]:
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
            "course": q.get("course"),
            "gold_answerable": bool(q["gold_answerable"])
            if "gold_answerable" in q
            else bool(q.get("answerable", True)),
            "gold_answers": q.get("gold_answers", []),
            "evidence_ids": list(q.get("evidence_ids") or ([] if not q.get("evidence_id") else [q["evidence_id"]])),
            "category": question_category(q),
            "expected_action": expected_action(q),
            "retriever_confidence": round(retr_conf, 4),
            "reasoner_confidence": round(float(rr.get("confidence", 0.0)), 4),
            "answerability_confidence": round(
                float(rr.get("answerability_confidence", rr.get("confidence", 0.0))), 4
            ),
            # Continuous unanswerable-risk score for ROC/PR-AUC (documented).
            "unanswerable_score": round(
                1.0
                - float(rr.get("answerability_confidence", rr.get("confidence", 0.0))),
                4,
            ),
            "model_is_answerable": bool(rr.get("is_answerable", False)),
            "model_needs_clarification": bool(rr.get("needs_clarification", False)),
            "raw_answer": str(rr.get("answer", "") or ""),
            "n_passages": len(s["passages"]),
            "retrieved_ids": list(s.get("retrieved_ids") or []),
            "passage_texts": [str(p.get("text", "") or "") for p in (s.get("passages") or [])],
            "dense_ids": [str(p.get("id", "")) for p in (s.get("dense") or [])],
            "sparse_ids": [str(p.get("id", "")) for p in (s.get("sparse") or [])],
            "fused_ids": [str(p.get("id", "")) for p in (s.get("fused") or [])],
            "retrieval_mode": s.get("retrieval_mode"),
            "reranked": bool(s.get("reranked", False)),
            "rag_threshold": float(rag_threshold),
            "embed_model": embed_model,
            "llm_model": llm_model,
        }
        for name in system_names:
            decide = systems[name]
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
    from app.config import Config
    from eval.baselines import ALL_SYSTEMS, RETRIEVAL_PROFILES, build_systems

    cfg = load_config(args.config)
    if args.reasoning_mode:
        cfg["reasoning_mode"] = args.reasoning_mode
    if args.questions:
        cfg["questions"] = args.questions
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.seeds is not None:
        cfg["seeds"] = args.seeds

    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out_dir / "experiments_config_used.yml")

    questions = load_questions(Path(cfg["questions"]), int(cfg.get("limit", 0) or 0))
    print(f"[run_experiments] Loaded {len(questions)} questions from {cfg['questions']}")

    # Fail fast if evidence IDs are missing (retrieval metrics need them).
    n_with_ev = sum(1 for q in questions if q.get("evidence_passage"))
    n_linked = sum(1 for q in questions if q.get("evidence_ids") or q.get("evidence_id"))
    if n_with_ev and n_linked == 0:
        print(
            "[run_experiments] WARNING: evidence passages present but no evidence_ids. "
            "Run: python -m eval.link_evidence_ids  then rebuild the FAISS index."
        )

    os.environ["REASONING_MODE"] = cfg.get("reasoning_mode", "mock")
    reasoning_mode = cfg.get("reasoning_mode", "mock")
    if reasoning_mode == "ollama":
        preflight_ollama()

    from app.agents.hybrid_retriever import HybridRetrieverAgent
    from app.agents.reasoning_agent import ReasoningAgent
    from app.agents.reranker_agent import RerankerAgent

    rag_threshold = float(cfg.get("rag_threshold", 0.2))
    top_k = int(cfg.get("top_k", 5))
    embed_model = getattr(Config, "EMBED_MODEL", "all-MiniLM-L6-v2")
    llm_model = getattr(Config, "OLLAMA_MODEL", "qwen2.5:3b-instruct")

    print("[run_experiments] Initialising hybrid retriever + reranker ...")
    reranker = RerankerAgent(enabled=True)
    retriever = HybridRetrieverAgent(reranker=reranker)
    reasoner = ReasoningAgent(mode=reasoning_mode)
    systems = build_systems(cfg.get("governance") or {}, rag_threshold=rag_threshold)

    # Allow subset via config.
    selected = list(cfg.get("systems") or ALL_SYSTEMS)
    for name in selected:
        if name not in systems:
            raise SystemExit(f"Unknown system '{name}'. Known: {list(systems)}")
    print(f"[run_experiments] Systems: {selected}")

    profiles = unique_profiles(selected, RETRIEVAL_PROFILES)
    print(f"[run_experiments] Retrieval profiles: {profiles}")

    seeds = list(cfg.get("seeds") or [42])
    all_rows: List[Dict] = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        # Shuffle a copy of questions; same seed => same order.
        qs = list(questions)
        random.shuffle(qs)

        seed_rows: List[Dict] = []
        for mode, use_rerank in profiles:
            names = [
                n
                for n in selected
                if RETRIEVAL_PROFILES[n] == (mode, use_rerank)
            ]
            if not names:
                continue
            print(f"[run_experiments] seed={seed} profile=({mode}, rerank={use_rerank}) -> {names}")
            # For profiles that need reranker disabled, pass use_reranker=False
            # even though the agent exists (FULL - Reranker ablation).
            signals = await collect_signals_for_profile(
                qs, retriever, reasoner, top_k, mode, use_rerank
            )
            seed_rows.extend(
                apply_systems_for_profile(
                    signals,
                    systems,
                    names,
                    seed,
                    reasoning_mode,
                    rag_threshold,
                    embed_model,
                    llm_model,
                )
            )

        seed_path = out_dir / f"predictions_seed{seed}.jsonl"
        with open(seed_path, "w", encoding="utf-8") as f:
            for row in seed_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[run_experiments] Wrote {seed_path} ({len(seed_rows)} rows)")
        all_rows.extend(seed_rows)

    # Combine from every predictions_seed*.jsonl file present in out_dir, not
    # just the seeds processed in THIS invocation. This allows running each
    # seed in a separate invocation (e.g. on a machine with limited
    # continuous runtime) -- the combined file is rebuilt from whatever seed
    # files exist on disk, so the last invocation naturally produces a
    # complete predictions_all.jsonl once every seed has been run at least
    # once, regardless of the order or whether they ran in the same process.
    pred_name = cfg.get("predictions_file", "predictions_all.jsonl")
    all_path = out_dir / pred_name
    seed_files = sorted(out_dir.glob("predictions_seed*.jsonl"))
    combined_rows: List[Dict] = []
    for sf in seed_files:
        with open(sf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    combined_rows.append(json.loads(line))
    with open(all_path, "w", encoding="utf-8") as f:
        for row in combined_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"[run_experiments] Wrote {all_path} ({len(combined_rows)} rows, "
        f"combined from {len(seed_files)} seed file(s): "
        f"{[sf.name for sf in seed_files]})"
    )
    found_seeds = sorted({r.get("seed", 0) for r in combined_rows})
    expected_seeds = sorted(set(cfg.get("seeds") or [42]))
    missing = sorted(set(expected_seeds) - set(found_seeds))
    if missing:
        print(
            f"[run_experiments] NOTE: experiments.yml lists seeds {expected_seeds}, "
            f"but only {found_seeds} were found on disk. predictions_all.jsonl is "
            f"missing seed(s) {missing} -- run them before using this file for the "
            f"final report."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--reasoning-mode", default=None, choices=["mock", "ollama"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run only these seed(s) in this invocation, e.g. --seeds 42. "
        "Overrides the 'seeds' list in experiments.yml. predictions_all.jsonl "
        "is rebuilt from every predictions_seed*.jsonl file already present "
        "in out_dir, so running one seed per invocation (e.g. across several "
        "sessions) still produces a complete combined file once all seeds "
        "have been run at least once.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()