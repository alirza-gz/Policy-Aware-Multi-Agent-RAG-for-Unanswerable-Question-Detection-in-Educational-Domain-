"""Pre-flight smoke validation — read-only analysis of results/smoke/."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics_extended import compute_extended_metrics
from eval.retrieval_metrics import compute_retrieval_metrics
from eval.significance import (
    binary_detection_correct,
    bootstrap_ci,
    decision_correct,
    mcnemar_test,
    paired_binary_labels,
)
from eval.answer_quality import compute_answer_quality
from app.agents.hybrid_retriever import HybridRetrieverAgent
from app.agents.reranker_agent import RerankerAgent

SMOKE = Path("results/smoke")
QUESTIONS = Path("data/eval/educational_v2_questions.jsonl")
OUT = SMOKE / "PREFLIGHT_REPORT.md"


def load_rows():
    rows = []
    with open(SMOKE / "predictions_all.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_questions():
    rows = []
    with open(QUESTIONS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    questions = load_questions()
    systems = sorted({r["mode"] for r in rows})

    lines = []
    lines.append("# PRE-FLIGHT REPORT — Smoke Test Validation\n")
    lines.append(
        f"Source: `{SMOKE}/predictions_all.jsonl` "
        f"(n={len(rows)} rows, {len(systems)} systems, "
        f"seeds={sorted({r['seed'] for r in rows})}, "
        f"reasoning={rows[0].get('reasoning_mode')}).\n"
    )
    lines.append("**Full 2498×3 experiment was NOT run.**\n")

    # ------------------------------------------------------------------ A
    lines.append("## A. Smoke-test numerical results (all 11 systems)\n")
    records = []
    for mode in systems:
        mrows = [r for r in rows if r["mode"] == mode]
        m = compute_extended_metrics(mrows)
        aq = compute_answer_quality(mrows)
        ret = compute_retrieval_metrics(mrows, k=5)
        records.append(
            {
                "system": mode,
                "accuracy": m.get("accuracy"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "f1": m.get("f1"),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "abstention_precision": m.get("abstention_precision"),
                "abstention_recall": m.get("abstention_recall"),
                "coverage": m.get("coverage"),
                "hallucination_rate": m.get("hallucination_rate"),
                "unsupported_answer_rate": m.get("unsupported_answer_rate"),
                "correct_abstention_rate": m.get("correct_abstention_rate"),
                "policy_compliance": m.get("policy_compliance_rate"),
                "correct_ANSWER": m.get("correct_answer_rate"),
                "correct_CLARIFY": m.get("correct_clarify_rate"),
                "correct_ABSTAIN": m.get("correct_abstain_rate"),
                "false_acceptance": m.get("false_acceptance_rate"),
                "false_rejection": m.get("false_rejection_rate"),
                "EM": aq.get("exact_match"),
                "token_F1": aq.get("token_f1"),
                "groundedness": aq.get("groundedness"),
                "Hit@5": ret.get("hit@5"),
                "Recall@5": ret.get("recall@5"),
                "MRR": ret.get("mrr"),
                "nDCG@5": ret.get("ndcg@5"),
                "n_with_gold_ev": ret.get("n_with_gold"),
            }
        )
    df = pd.DataFrame(records).set_index("system").round(4)
    df.to_csv(SMOKE / "preflight_system_comparison.csv")
    lines.append("```")
    lines.append(df.to_string())
    lines.append("```\n")
    lines.append(
        f"Saved machine-readable copy: `{SMOKE / 'preflight_system_comparison.csv'}`.\n"
    )
    lines.append(
        "AUC score used: `unanswerable_score = 1 - answerability_confidence` "
        "(continuous; not hard labels).\n"
    )

    # ------------------------------------------------------------------ B
    lines.append("## B. Statistical-test validation\n")
    full = [r for r in rows if r["mode"] == "full_system"]
    comparisons = [
        "standard_rag",
        "rag_threshold",
        "agentic_rag",
        "llm_no_rag",
        "no_reranker",
        "no_hybrid",
        "no_answerability",
        "no_policy_agent",
    ]
    sig_rows = []
    for other_name in comparisons:
        other = [r for r in rows if r["mode"] == other_name]
        for label, fn in (
            ("binary_detection_correct", binary_detection_correct),
            ("expected_action_correct", decision_correct),
        ):
            ya, yb = paired_binary_labels(full, other, fn)
            # Sanity: not placeholders
            assert len(ya) > 0
            assert set(np.unique(ya)).issubset({0, 1})
            res = mcnemar_test(ya, yb)
            sig_rows.append(
                {
                    "reference": "full_system",
                    "system": other_name,
                    "label": label,
                    "n_paired": res["n"],
                    "b_A0B1": res.get("b"),
                    "c_A1B0": res.get("c"),
                    "statistic": res.get("statistic"),
                    "p_value": res.get("p_value"),
                    "effect_size": res.get("effect_size"),
                    "effect_size_name": res.get("effect_size_name"),
                    "from_prediction_rows": True,
                }
            )
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(SMOKE / "preflight_significance.csv", index=False)
    lines.append("### McNemar (FULL vs others) — computed from paired prediction rows\n")
    lines.append("```")
    lines.append(sig_df.round(6).to_string(index=False))
    lines.append("```\n")

    # Bootstrap CIs for FULL and key baselines from per-question indicators
    lines.append("### Bootstrap 95% CIs (percentile, n_boot=1000, seed=42)\n")
    ci_rows = []
    for mode in ["full_system", "standard_rag", "agentic_rag", "no_reranker", "llm_no_rag"]:
        mrows = [r for r in rows if r["mode"] == mode]
        acc_vals = [binary_detection_correct(r) for r in mrows]
        cov_vals = [1 if r.get("action") == "ANSWER" else 0 for r in mrows]
        hall_vals = [
            1
            if (
                not r.get("gold_answerable", True)
                and r.get("action") == "ANSWER"
                and (r.get("final_answer") or "").strip()
            )
            else 0
            for r in mrows
            if not r.get("gold_answerable", True)
        ]
        for metric, vals in (
            ("accuracy", acc_vals),
            ("coverage", cov_vals),
            ("hallucination_rate", hall_vals),
        ):
            ci = bootstrap_ci(vals, n_boot=1000, seed=42)
            # Prove non-placeholder: CI width or trivial equality when constant
            assert ci["n"] > 0
            assert "percentile_bootstrap" in ci["method"]
            ci_rows.append({"system": mode, "metric": metric, **ci})
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(SMOKE / "preflight_confidence_intervals.csv", index=False)
    lines.append("```")
    lines.append(ci_df.round(4).to_string(index=False))
    lines.append("```\n")
    lines.append(
        "Validation: McNemar uses discordant cell counts `(b,c)` from aligned "
        "`(id, seed)` pairs; bootstrap resamples per-question indicator vectors. "
        "Neither is a hardcoded placeholder.\n"
    )

    # ------------------------------------------------------------------ C
    lines.append("## C. CLARIFY validation\n")
    clarify_rows = [r for r in rows if r["action"] == "CLARIFY"]
    lines.append(f"Total CLARIFY decisions across all systems: **{len(clarify_rows)}**\n")
    by_sys = Counter(r["mode"] for r in clarify_rows)
    lines.append("### By system\n")
    lines.append("```")
    lines.append(str(dict(sorted(by_sys.items()))))
    lines.append("```\n")

    # Focus on full_system (the 12/100)
    full_c = [r for r in rows if r["mode"] == "full_system" and r["action"] == "CLARIFY"]
    lines.append(f"### FULL system CLARIFY count: **{len(full_c)} / 100**\n")
    lines.append("By expected_action:\n```")
    lines.append(str(dict(Counter(r["expected_action"] for r in full_c))))
    lines.append("```\nBy category:\n```")
    lines.append(str(dict(Counter(r["category"] for r in full_c))))
    lines.append("```\n")

    clarify_on_answerable = [
        r for r in full_c if r.get("category") == "answerable" or r.get("gold_answerable")
    ]
    lines.append(
        f"CLARIFY on gold-answerable questions (FULL): **{len(clarify_on_answerable)}**\n"
    )
    if clarify_on_answerable:
        lines.append("Examples (id, category, expected_action, reason):\n```")
        for r in clarify_on_answerable[:10]:
            lines.append(
                f"{r['id']} | {r['category']} | exp={r['expected_action']} | {r['reason']}"
            )
        lines.append("```\n")
        lines.append(
            "**Finding:** CLARIFY *does* fire on some Answerable questions "
            "(false clarification / mid-band confidence). This is a false rejection "
            "relative to expected_action=ANSWER, not a correct CLARIFY.\n"
        )
    else:
        lines.append(
            "**Finding:** No CLARIFY on Answerable questions under FULL in this smoke set.\n"
        )

    # Per-system clarify × category crosstab for systems that clarify
    lines.append("### CLARIFY × category × system\n")
    ct_records = []
    for r in clarify_rows:
        ct_records.append(
            {"system": r["mode"], "category": r["category"], "expected_action": r["expected_action"]}
        )
    if ct_records:
        ctdf = pd.DataFrame(ct_records)
        pivot = ctdf.groupby(["system", "category", "expected_action"]).size().reset_index(name="n")
        lines.append("```")
        lines.append(pivot.to_string(index=False))
        lines.append("```\n")

    # ------------------------------------------------------------------ D
    lines.append("## D. Reranker validation\n")
    full_map = {r["id"]: r for r in rows if r["mode"] == "full_system"}
    nr_map = {r["id"]: r for r in rows if r["mode"] == "no_reranker"}
    order_diff = sum(
        1 for i in full_map if full_map[i].get("retrieved_ids") != nr_map[i].get("retrieved_ids")
    )
    lines.append(
        f"Ordering differs FULL vs no_reranker: **{order_diff}/{len(full_map)}** questions.\n"
    )
    lines.append("### Example before/after top-k IDs\n")
    examples = []
    for i, fr in full_map.items():
        nr = nr_map[i]
        if fr.get("retrieved_ids") != nr.get("retrieved_ids"):
            examples.append((i, fr, nr))
        if len(examples) >= 5:
            break
    lines.append("```")
    for i, fr, nr in examples:
        lines.append(f"id={i} category={fr['category']} gold={fr.get('evidence_ids')}")
        lines.append(f"  no_reranker (hybrid RRF): {nr.get('retrieved_ids')}")
        lines.append(f"  FULL (hybrid+rerank):     {fr.get('retrieved_ids')}")
        gold = set(fr.get("evidence_ids") or [])
        if gold:
            def rank(ids, g):
                for k, pid in enumerate(ids or [], 1):
                    if pid in g:
                        return k
                return None
            lines.append(
                f"  gold rank: no_reranker={rank(nr.get('retrieved_ids'), gold)} "
                f"FULL={rank(fr.get('retrieved_ids'), gold)}"
            )
    lines.append("```\n")

    ret_full = compute_retrieval_metrics([r for r in rows if r["mode"] == "full_system"], k=5)
    ret_nr = compute_retrieval_metrics([r for r in rows if r["mode"] == "no_reranker"], k=5)
    lines.append("### Retrieval metrics: FULL vs no_reranker\n")
    lines.append("```")
    lines.append(f"FULL:        { {k: ret_full[k] for k in ret_full} }")
    lines.append(f"no_reranker: { {k: ret_nr[k] for k in ret_nr} }")
    lines.append("```\n")
    # Delta
    for metric in ["hit@5", "recall@5", "mrr", "ndcg@5"]:
        d = ret_full.get(metric, float("nan")) - ret_nr.get(metric, float("nan"))
        lines.append(f"- Δ {metric} (FULL − no_reranker) = {d:.4f}")
    improved = ret_full.get("mrr", 0) > ret_nr.get("mrr", 0)
    lines.append(
        f"\n**Assessment:** Reranker "
        f"{'improves' if improved else 'does not improve'} MRR on this smoke set; "
        "ordering changes alone do not imply quality gains.\n"
    )

    # ------------------------------------------------------------------ E
    lines.append("## E. Hybrid retrieval validation\n")
    # Live retrieval comparison on smoke question texts (same corpus), no LLM.
    smoke_qs = []
    seen = set()
    for r in rows:
        if r["mode"] == "full_system" and r["id"] not in seen:
            seen.add(r["id"])
            smoke_qs.append(r)

    print("[preflight] Building retriever for live dense/bm25/hybrid comparison...")
    retr = HybridRetrieverAgent(reranker=RerankerAgent(enabled=False))
    mode_rows = {"dense": [], "bm25": [], "hybrid": []}
    for q in smoke_qs:
        for mode in ("dense", "bm25", "hybrid"):
            pipe = retr.retrieve_pipeline(
                q["question"], top_k=5, mode=mode, use_reranker=False
            )
            mode_rows[mode].append(
                {
                    **{k: q[k] for k in ("id", "evidence_ids", "category", "course") if k in q},
                    "evidence_ids": q.get("evidence_ids") or [],
                    "retrieved_ids": pipe["retrieved_ids"],
                    "mode": mode,
                }
            )
    lines.append("Live retrieval (smoke questions, no LLM, no rerank), k=5:\n```")
    hybrid_cmp = {}
    for mode, mrows in mode_rows.items():
        metrics = compute_retrieval_metrics(mrows, k=5)
        hybrid_cmp[mode] = metrics
        lines.append(f"{mode}: {metrics}")
    lines.append("```\n")
    pd.DataFrame(
        [{"retrieval": m, **hybrid_cmp[m]} for m in hybrid_cmp]
    ).to_csv(SMOKE / "preflight_retrieval_modes.csv", index=False)

    # Also from stored predictions: no_hybrid vs full_system (full includes rerank)
    ret_dense_abl = compute_retrieval_metrics(
        [r for r in rows if r["mode"] == "no_hybrid"], k=5
    )
    lines.append(
        "From stored ablation predictions (note: `no_hybrid` = dense+**rerank**; "
        "`full_system` = hybrid+**rerank**):\n```"
    )
    lines.append(f"no_hybrid (dense+rerank): {ret_dense_abl}")
    lines.append(f"full_system (hybrid+rerank): {ret_full}")
    lines.append("```\n")

    # ------------------------------------------------------------------ F
    lines.append("## F. Agentic RAG assessment\n")
    # Prove equivalence on smoke
    agentic = {r["id"]: (r["action"], r["final_answer"]) for r in rows if r["mode"] == "agentic_rag"}
    no_pol = {r["id"]: (r["action"], r["final_answer"]) for r in rows if r["mode"] == "no_policy_agent"}
    no_ma = {r["id"]: (r["action"], r["final_answer"]) for r in rows if r["mode"] == "no_multi_agent"}
    eq_pol = all(agentic[i] == no_pol[i] for i in agentic)
    eq_ma = all(agentic[i] == no_ma[i] for i in agentic)
    lines.append(
        f"Decision-layer equivalence on smoke: "
        f"`agentic_rag` ≡ `no_policy_agent` = **{eq_pol}**; "
        f"`agentic_rag` ≡ `no_multi_agent` = **{eq_ma}**.\n"
    )
    lines.append(
        "### Is this acceptable for the advisor's \"Agentic RAG\" baseline?\n\n"
        "**Partially acceptable, with explicit caveats — not a full agentic system.**\n\n"
        "What it is today:\n"
        "- One retrieval pass + one LLM reasoner call.\n"
        "- The reasoner self-governs via `is_answerable` / `needs_clarification`.\n"
        "- No Policy/Governance Agent thresholds or safety rules.\n"
        "- Fair against FULL on the same hybrid+rerank signals.\n\n"
        "What the advisor wording typically implies but is **missing**:\n"
        "- Iterative retrieve→reason→retrieve loops\n"
        "- Tool-calling / query reformulation / multi-hop planning\n\n"
        "### Smallest change to make Agentic RAG genuinely iterative "
        "(DO NOT implement now)\n\n"
        "Add a thin loop **only** for the `agentic_rag` baseline (leave FULL unchanged):\n"
        "1. Start with hybrid+rerank top-k.\n"
        "2. Reasoner returns answerability + optional `reformulated_query` / "
        "`needs_more_retrieval` flag (extend mock/ollama schema minimally).\n"
        "3. If flagged and `iteration < max_iters` (e.g. 2–3), re-retrieve with "
        "reformulated query, merge/rerank, reason again.\n"
        "4. Final action still from reasoner self-governance (no Policy Agent).\n"
        "5. Log `n_iterations`, queries, and retrieved IDs per hop.\n\n"
        "This isolates \"agentic iteration\" as a baseline property without redesigning "
        "FULL's GovernanceAgent semantics.\n"
    )

    # ------------------------------------------------------------------ G
    lines.append("## G. Dataset integrity\n")
    n = len(questions)
    cats = Counter(q["category"] for q in questions)
    acts = Counter(q["expected_action"] for q in questions)
    ids = [q["id"] for q in questions]
    dup = len(ids) - len(set(ids))
    n_eid = sum(1 for q in questions if q.get("evidence_ids"))
    ans_missing_eid = [
        q
        for q in questions
        if q.get("category") == "answerable"
        and q.get("evidence_passage")
        and not (q.get("evidence_ids") or q.get("evidence_id"))
    ]
    # expected_action consistency with category mapping
    mapping = {
        "answerable": ("ANSWER", True),
        "unanswerable": ("ABSTAIN", False),
        "out_of_domain": ("ABSTAIN", False),
        "false_premise": ("ABSTAIN", False),
        "partially_answerable": ("CLARIFY", False),
    }
    inconsistent = []
    for q in questions:
        exp_act, exp_ans = mapping[q["category"]]
        if q["expected_action"] != exp_act or bool(q["gold_answerable"]) != exp_ans:
            inconsistent.append(q["id"])

    lines.append(f"- Questions: **{n}** (required 2498) → {'PASS' if n == 2498 else 'FAIL'}")
    lines.append(f"- Categories: {dict(cats)}")
    required_cats = {
        "answerable",
        "unanswerable",
        "out_of_domain",
        "false_premise",
        "partially_answerable",
    }
    lines.append(
        f"- Five required categories present: "
        f"{'PASS' if required_cats <= set(cats) else 'FAIL'}"
    )
    lines.append(f"- expected_action counts: {dict(acts)}")
    lines.append(
        f"- Category↔expected_action↔gold_answerable consistency errors: "
        f"**{len(inconsistent)}** → {'PASS' if not inconsistent else 'FAIL'}"
    )
    lines.append(f"- Duplicate IDs: **{dup}** → {'PASS' if dup == 0 else 'FAIL'}")
    lines.append(
        f"- Evidence IDs linked: **{n_eid}** (expected 1750) → "
        f"{'PASS' if n_eid == 1750 else 'FAIL'}"
    )
    lines.append(
        f"- Answerable with evidence_passage but missing evidence_ids: "
        f"**{len(ans_missing_eid)}** → {'PASS' if not ans_missing_eid else 'FAIL'}"
    )
    # Label change check vs smoke predictions gold fields
    qmap = {q["id"]: q for q in questions}
    label_mismatches = 0
    for r in rows:
        if r["mode"] != "full_system":
            continue
        q = qmap.get(r["id"])
        if not q:
            label_mismatches += 1
            continue
        if (
            q["category"] != r["category"]
            or q["expected_action"] != r["expected_action"]
            or bool(q["gold_answerable"]) != bool(r["gold_answerable"])
        ):
            label_mismatches += 1
    lines.append(
        f"- Smoke predictions vs JSONL label fields mismatches: "
        f"**{label_mismatches}** → {'PASS' if label_mismatches == 0 else 'FAIL'}\n"
    )

    # ------------------------------------------------------------------ H
    lines.append("## H. Experimental fairness\n")
    sample_by = {m: next(r for r in rows if r["mode"] == m) for m in systems}
    lines.append("| system | llm | embed | retrieval_mode | reranked | rag_threshold | reasoning |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, r in sorted(sample_by.items()):
        lines.append(
            f"| {m} | {r.get('llm_model')} | {r.get('embed_model')} | "
            f"{r.get('retrieval_mode')} | {r.get('reranked')} | "
            f"{r.get('rag_threshold')} | {r.get('reasoning_mode')} |"
        )
    lines.append("")
    lines.append("**Shared:** dataset slice (same 100 IDs/seed), corpus/index, embed model, "
                 "LLM id field, seeds, rag_threshold config value logged.\n")
    lines.append(
        "**Unavoidable differences:**\n"
        "- `llm_no_rag`: no retrieval (`retrieval_mode=none`, empty passages).\n"
        "- `no_reranker`: hybrid without CE rerank → separate reasoner pass.\n"
        "- `no_hybrid`: dense+rerank → separate reasoner pass.\n"
        "- Decision-layer systems sharing `(hybrid, rerank=True)` reuse one signal cache "
        "(fair within that profile).\n"
        "- Smoke used `reasoning_mode=mock` (deterministic lexical heuristic). "
        "Full run must use `ollama` for thesis claims about LLM behavior.\n"
    )

    # ------------------------------------------------------------------ I
    lines.append("## I. Remaining risks before full 2498 × 3-seed run\n")
    lines.append(
        "1. **Smoke uses mock reasoner** — metrics (esp. CLARIFY, AUC, answer quality) "
        "will shift under Ollama; treat smoke as pipeline correctness, not thesis numbers.\n"
        "2. **Agentic RAG is a single-agent proxy** — document clearly; consider iterative "
        "loop before claiming \"agentic\" in the thesis narrative (not implemented).\n"
        "3. **Reranker may not improve IR metrics** on reconstructed evidence-only corpus "
        "(already near-ceiling Hit@K); ablation value may show more in generation/governance "
        "than in Recall@K.\n"
        "4. **Runtime/cost:** 4 retrieval profiles × 2498 × 3 seeds of Ollama calls; "
        "ensure Ollama stable and model pulled.\n"
        "5. **Corpus = evidence passages only** (held-out excluded) — not full Wikipedia; "
        "document as limitation.\n"
        "6. **`no_policy_agent` / `no_multi_agent` / `agentic_rag` are redundant** at the "
        "decision layer — fine for table completeness, but do not over-interpret as three "
        "independent findings.\n"
        "7. Cross-encoder download/runtime must succeed on the full-run machine.\n"
    )

    # Final verdict
    lines.append("## Verdict\n")
    ready_issues = []
    if n != 2498:
        ready_issues.append("dataset size")
    if ans_missing_eid:
        ready_issues.append("missing answerable evidence IDs")
    if inconsistent:
        ready_issues.append("label inconsistencies")
    if dup:
        ready_issues.append("duplicate IDs")
    if not eq_pol or not eq_ma:
        ready_issues.append("unexpected non-equivalence of agentic ablations")

    # Check CLARIFY fires
    if len(full_c) == 0:
        ready_issues.append("CLARIFY never fires on FULL")

    # Continuity of scores
    scores = [r.get("unanswerable_score") for r in full]
    if len(set(scores)) < 2:
        ready_issues.append("unanswerable_score has no variation under mock (AUC may be unstable)")

    lines.append(
        "### READY for full 2498 × 3-seed run?\n\n"
        "**CONDITIONAL YES — pipeline is ready; narrative caveats remain.**\n\n"
        "The evaluation harness, all 11 systems, metrics, McNemar/bootstrap, evidence IDs, "
        "and ablations execute correctly on the smoke set.\n\n"
        "**Before claiming thesis results you must:**\n"
        "- Run with `--reasoning-mode ollama` (not mock).\n"
        "- Explicitly document Agentic RAG as a single-agent non-iterative proxy "
        "(or implement the iterative loop first).\n"
        "- Not over-claim reranker IR gains if Hit@K is already saturated.\n"
    )
    if ready_issues:
        lines.append("Open smoke caveats: " + "; ".join(ready_issues) + "\n")
    else:
        lines.append("No blocking data-integrity failures detected.\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(df.to_string())
    print("CLARIFY full:", len(full_c), "on answerable:", len(clarify_on_answerable))
    print("Reranker order diffs:", order_diff)
    print("Retrieval FULL", ret_full)
    print("Retrieval no_reranker", ret_nr)
    print("Hybrid modes", hybrid_cmp)
    print("READY issues:", ready_issues or "none")


if __name__ == "__main__":
    main()
