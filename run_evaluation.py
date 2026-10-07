#!/usr/bin/env python3
"""Single reproducible entry point for the complete evaluation.

    python run_evaluation.py --dataset educational_v2 --seed 42 123 2024 \
        --reasoning-mode ollama --evaluate-all --save-results

    # Re-derive every aggregate from signals of an earlier run (no LLM / GPU needed):
    python run_evaluation.py --from-signals results/runs/<run>/signals.jsonl --evaluate-all

Stages (each writes into ONE run directory, results/runs/<timestamp>_<hash>/):
  1. collect signals   retrieval -> rerank -> reasoner, per seed & retrieval profile   [needs LLM/index]
  2. fit on validation score model + abstain threshold (grouped split, validation only)
  3. decide            every baseline/ablation, from signals (pure, replayable)
  4. aggregate         all metrics/tests/CIs from the single predictions.jsonl

Provenance written to config.json: config, seeds, LLM, embedder, retriever, reranker,
thresholds, dataset path + sha256, timestamp, git commit (or "unavailable").
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DATASETS = {"educational_v2": "data/eval/educational_v2_questions.jsonl",
            "sciq": "data/eval/sciq_questions.jsonl"}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unavailable (not a git checkout)"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def signals_to_rows(signals: List[Dict], system_names: List[str], artifacts: Dict,
                    rag_threshold: float, meta: Dict) -> List[Dict]:
    from eval.pipeline_v2 import RETRIEVAL_PROFILES, build_systems
    from app.agents.answerability_agent import AnswerabilityAgent
    systems = build_systems(artifacts, rag_threshold=rag_threshold)
    plain = AnswerabilityAgent(thresholds=artifacts.get("assessment_thresholds"),
                               score_model=artifacts.get("score_model"))
    rows = []
    for s in signals:
        q, rr = s["question"], s["reasoning_result"]
        prof = (s["retrieval_mode"], bool(s["reranked"]))
        for name in system_names:
            if RETRIEVAL_PROFILES[name] != prof:
                continue
            d = systems[name](q["question"], s["passages"], rr, s["retriever_confidence"])
            # Assessment is logged for EVERY system so spec-compliance can be checked independently.
            a = plain.assess(q["question"], s["passages"], s["retriever_confidence"], rr)
            row = {
                "id": q["id"], "seed": s["seed"], "mode": name, "course": q.get("course"),
                "category": q.get("category"), "question": q["question"],
                "gold_answerable": bool(q.get("gold_answerable", q.get("answerable", True))),
                "gold_answers": q.get("gold_answers", []), "expected_action": q["expected_action"],
                "out_of_scope_part_question": q.get("out_of_scope_part_question"),
                "source_document": q.get("source_document"),
                "retriever_confidence": s["retriever_confidence"],
                "retrieval_confidence": s["retriever_confidence"],
                "legacy_answerability_confidence": rr.get("answerability_confidence"),
                "model_is_answerable": rr.get("is_answerable"),
                "model_needs_clarification": rr.get("needs_clarification"),
                "raw_answer": rr.get("answer", ""),
                "passage_texts": [p.get("text", "") for p in s["passages"]],
                "retrieved_ids": s["retrieved_ids"],
                "answerability_label": d.get("answerability_label", a.answerability_label),
                "answerability_score": d.get("answerability_score", a.answerability_score),
                "evidence_sufficient": d.get("evidence_sufficient", a.evidence_sufficient),
                "clarification_trigger": d.get("clarification_trigger", "none"),
                "clarification_reason": d.get("clarification_reason", ""),
                "clarification_question": d.get("clarification_question", ""),
                "missing_information": d.get("missing_information", ""),
                "missing_parts": d.get("missing_parts", []),
                **{k: d.get(k) for k in ("policy_checked", "policy_rule_ids", "policy_rule_triggered",
                                         "policy_decision", "policy_override", "original_action",
                                         "final_action", "policy_reason")},
                "predicted_action": d["action"], "action": d["action"],
                "final_answer": d["final_answer"], "reason": d.get("reason", ""),
                **meta,
            }
            rows.append(row)
    return rows


def retrieval_trace(signals: List[Dict]) -> List[Dict]:
    """One row per question from the (hybrid, rerank=True) profile; all four configs derive from it."""
    seen, out = set(), []
    for s in signals:
        if s.get("replayed_from_predictions"):
            return []          # candidate lists are depth-limited in replays: no retrieval metrics
        if (s["retrieval_mode"], s["reranked"]) != ("hybrid", True) or s["question"]["id"] in seen:
            continue
        seen.add(s["question"]["id"])
        ids = lambda xs: [str(p.get("id", "")) for p in xs]  # noqa: E731
        out.append({"id": s["question"]["id"], "gold_ids": s["question"].get("evidence_ids") or [],
                    "dense_ids": ids(s.get("dense", [])), "bm25_ids": ids(s.get("sparse", [])),
                    "sparse_ids": ids(s.get("sparse", [])), "hybrid_ids": ids(s.get("pre_rerank") or s.get("fused", [])),
                    "hybrid_rerank_ids": ids(s.get("ranked_deep", []))})
    return out


async def collect(args, cfg, questions, run_dir: Path) -> List[Dict]:
    import os
    os.environ["REASONING_MODE"] = args.reasoning_mode
    from app.agents.hybrid_retriever import HybridRetrieverAgent
    from app.agents.reasoning_agent import ReasoningAgent
    from app.agents.reranker_agent import RerankerAgent
    from eval.pipeline_v2 import ALL_SYSTEMS, RETRIEVAL_PROFILES
    from eval.run_experiments import collect_signals_for_profile, preflight_ollama
    import random
    if args.reasoning_mode == "ollama":
        preflight_ollama()
    retriever = HybridRetrieverAgent(reranker=RerankerAgent(enabled=True))
    reasoner = ReasoningAgent(mode=args.reasoning_mode)
    profiles = sorted({RETRIEVAL_PROFILES[n] for n in ALL_SYSTEMS})
    sig_path = run_dir / "signals.jsonl"
    out: List[Dict] = []
    with open(sig_path, "w", encoding="utf-8") as f:
        for seed in args.seed:
            random.seed(seed); reasoner.seed = seed
            qs = list(questions); random.shuffle(qs)
            for mode, rer in profiles:
                for s in await collect_signals_for_profile(qs, retriever, reasoner, args.top_k, mode, rer):
                    s["seed"] = seed
                    out.append(s)
                    f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="educational_v2", help="alias or path to a questions .jsonl")
    ap.add_argument("--seed", type=int, nargs="+", default=[42, 123, 2024])
    ap.add_argument("--reasoning-mode", choices=["ollama", "mock"], default="ollama")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rag-threshold", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.3)
    ap.add_argument("--max-false-accept", type=float, default=0.10,
                    help="cap on validation false-accept rate when choosing the abstain threshold")
    ap.add_argument("--from-signals", default=None, help="skip collection; replay from signals.jsonl")
    ap.add_argument("--evaluate-all", action="store_true")
    ap.add_argument("--save-results", action="store_true")
    ap.add_argument("--out-root", default="results/runs")
    args = ap.parse_args(argv)

    from eval.aggregate import aggregate_run
    from eval.pipeline_v2 import ALL_SYSTEMS, fit_artifacts
    from eval.run_experiments import expected_action, load_questions, question_category
    from app.config import Config

    ds_path = Path(DATASETS.get(args.dataset, args.dataset))
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.out_root) / f"{stamp}_{sha256_file(ds_path)[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.from_signals:
        signals = [json.loads(l) for l in open(args.from_signals, encoding="utf-8") if l.strip()]
        (run_dir / "signals.jsonl").write_text(Path(args.from_signals).read_text(encoding="utf-8"), encoding="utf-8")
    else:
        qs = load_questions(ds_path, args.limit)
        for q in qs:
            q["category"] = question_category(q); q["expected_action"] = expected_action(q)
        signals = asyncio.run(collect(args, {}, qs, run_dir))

    artifacts = fit_artifacts(signals, args.val_fraction, max_false_accept=args.max_false_accept)
    sm = artifacts["score_model"]
    (run_dir / "calibration_artifacts.json").write_text(json.dumps(
        {**{k: v for k, v in artifacts.items() if k != "score_model"}, "score_model": json.loads(sm.to_json())},
        indent=2))
    meta = {"reasoning_mode": args.reasoning_mode if not args.from_signals else "from_signals",
            "llm_model": getattr(Config, "OLLAMA_MODEL", None), "embed_model": getattr(Config, "EMBED_MODEL", None)}
    rows = signals_to_rows(signals, ALL_SYSTEMS, artifacts, args.rag_threshold, meta)
    with open(run_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    trace = retrieval_trace(signals)
    if trace:
        with open(run_dir / "retrieval_trace.jsonl", "w", encoding="utf-8") as f:
            for r in trace:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run_dir / "config.json").write_text(json.dumps({
        "dataset": args.dataset, "dataset_path": str(ds_path), "dataset_sha256": sha256_file(ds_path),
        "seeds": args.seed, "limit": args.limit, "top_k": args.top_k, "rag_threshold": args.rag_threshold,
        "val_fraction": args.val_fraction, "max_false_accept": args.max_false_accept,
        "reasoning_mode": meta["reasoning_mode"], "llm_model": meta["llm_model"],
        "embed_model": meta["embed_model"], "retrieval": getattr(Config, "RETRIEVAL", None),
        "reranker": getattr(Config, "RERANKER", None), "thresholds": artifacts["thresholds"],
        "threshold_selection": artifacts["threshold_selection"],
        "systems": ALL_SYSTEMS, "timestamp_utc": stamp, "git_commit": git_commit(),
        "argv": sys.argv, "replayed_from_signals": args.from_signals,
    }, indent=2, default=str))
    if args.evaluate_all:
        aggregate_run(run_dir, banned=list(getattr(Config, "BANNED_PHRASES", []) or []))
    print(f"[run_evaluation] wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
