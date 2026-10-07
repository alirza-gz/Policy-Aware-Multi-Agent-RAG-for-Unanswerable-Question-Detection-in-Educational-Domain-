"""Single source of truth: every aggregate file is derived from ONE predictions.jsonl.

Run directory layout (all produced by ``write_run`` / ``aggregate_run``):

    config.json                  provenance (model, seeds, git hash, thresholds, dataset ...)
    signals.jsonl                retrieval + reasoner outputs (replayable raw artifact)
    predictions.jsonl            one row per (system, question, seed): THE authoritative output
    retrieval_trace.jsonl        ranked ids per retrieval config (for retrieval_metrics.json)
    calibration_artifacts.json   score model + thresholds fit on VALIDATION only
    metrics.json  category_metrics.json  course_metrics.json  retrieval_metrics.json
    calibration.json  statistical_tests.json  confidence_intervals.json
    partial_answerable_eval.json  policy_instrumentation.json  ablation_equivalence.json
    consistency.json             sha256 of predictions.jsonl embedded in every aggregate

Reported metrics use TEST-split question ids only (thresholds were selected on validation).
No aggregate re-reads signals or any other metric script; metric definitions live in
eval.metrics_extended / eval.hallucination / eval.clarification_metrics /
eval.calibration / eval.retrieval_eval / eval.significance.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.calibration import calibration_report, compare_calibrators
from eval.clarification_metrics import compute_clarification_metrics, compute_partial_answerable_eval
from eval.hallucination import compute_hallucination_metrics
from eval.metrics_extended import compute_extended_metrics
from eval.retrieval_eval import evaluate_retrieval
from eval.significance import (compare_systems_by_question, decision_correct,
                               question_level_mean_std_ci)

BANNED = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _dump(path: Path, obj, meta: Dict) -> None:
    path.write_text(json.dumps({"_meta": meta, **obj} if isinstance(obj, dict) else obj,
                               indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def policy_compliance(rows: List[Dict], banned: List[str]) -> Dict:
    """Specification check independent of the decision code that produced the rows.

    Violation = delivered ANSWER while (evidence insufficient) or (label out_of_domain /
    false_premise) or (banned phrase present). Measures enforcement of the policy spec,
    NOT answer correctness; a high value for the full system is expected by construction
    of rules R4/R5/R2 and does not by itself show the policy is *useful*.
    """
    viol = 0
    for r in rows:
        if r.get("action") != "ANSWER":
            continue
        a = (r.get("final_answer") or "").lower()
        if (not r.get("evidence_sufficient", True)) or r.get("answerability_label") in ("out_of_domain", "false_premise") \
                or any(b.lower() in a for b in banned):
            viol += 1
    return {"policy_compliance": round(1 - viol / len(rows), 4) if rows else float("nan"),
            "policy_violations": viol}


def system_metrics(rows: List[Dict], banned: List[str]) -> Dict:
    m = compute_extended_metrics(rows)
    for k in ("roc_auc", "pr_auc", "auc_score_name", "policy_compliance_rate",
              "unsupported_answer_rate", "hallucination_rate"):
        m.pop(k, None)          # replaced below by definitions with explicit scope
    m.update(compute_hallucination_metrics(rows))
    m.update(compute_clarification_metrics(rows))
    m.update(policy_compliance(rows, banned))
    m["correct_abstention_rate"] = m.get("correct_abstention_rate")
    return m


def score_metrics(rows: List[Dict]) -> Dict:
    """AUC / PR-AUC for the standardised score (one row per question; shared by all systems
    on the same retrieval profile because the score is a pipeline signal, not a decision)."""
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"]); uniq.append(r)
    y_full = np.array([int(r["expected_action"] == "ANSWER") for r in uniq])
    s = np.array([float(r.get("answerability_score", np.nan)) for r in uniq])
    legacy = np.array([float(r.get("legacy_answerability_confidence", np.nan)) for r in uniq])
    out = {"n_questions": len(uniq), "score": "answerability_score = P(fully answerable)"}
    if len(set(y_full)) == 2 and not np.isnan(s).any():
        out["roc_auc_standardised"] = round(float(roc_auc_score(y_full, s)), 4)
        out["pr_auc_unanswerable_positive"] = round(float(average_precision_score(1 - y_full, 1 - s)), 4)
        out["pr_auc_answerable_positive"] = round(float(average_precision_score(y_full, s)), 4)
        out["calibration"] = calibration_report(y_full, s)
    if len(set(y_full)) == 2 and not np.isnan(legacy).any():
        out["roc_auc_legacy_llm_confidence"] = round(float(roc_auc_score(y_full, legacy)), 4)
        out["legacy_score_note"] = "LLM self-reported answerability_confidence (not a probability)"
    return out


def ablation_equivalence(rows: List[Dict]) -> List[Dict]:
    by = defaultdict(dict)
    for r in rows:
        by[r["mode"]][(r["id"], r["seed"])] = r["action"]
    out, names = [], sorted(by)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            keys = set(by[a]) & set(by[b])
            if keys:
                same = sum(by[a][k] == by[b][k] for k in keys) / len(keys)
                if same == 1.0:
                    out.append({"system_a": a, "system_b": b, "action_agreement": 1.0, "n": len(keys),
                                "note": "behaviourally identical on this run"})
    return out


def policy_instrumentation(rows: List[Dict]) -> Dict:
    out: Dict = {}
    for mode in sorted({r["mode"] for r in rows}):
        rs = [r for r in rows if r["mode"] == mode and r.get("policy_checked")]
        if not rs:
            continue
        trig = Counter(rid for r in rs for rid in r["policy_rule_triggered"])
        sole = Counter(r["policy_rule_triggered"][0] for r in rs
                       if r["policy_override"] and len(r["policy_rule_triggered"]) == 1)
        out[mode] = {"n_checked": len(rs), "override_rate": round(sum(r["policy_override"] for r in rs) / len(rs), 4),
                     "rule_trigger_counts": dict(trig), "rule_sole_override_counts": dict(sole),
                     "rules_never_triggered": sorted(set(rs[0]["policy_rule_ids"]) - set(trig))}
    return out


def aggregate_run(run_dir: Path, banned: Optional[List[str]] = None) -> Dict:
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.jsonl"
    rows_all = _read(pred_path)
    art = json.loads((run_dir / "calibration_artifacts.json").read_text()) \
        if (run_dir / "calibration_artifacts.json").exists() else {}
    test_ids = set(art.get("test_ids") or []) or {r["id"] for r in rows_all}
    rows = [r for r in rows_all if r["id"] in test_ids]
    banned = banned or []
    sha = _sha(pred_path)
    meta = {"predictions_sha256": sha, "n_prediction_rows_total": len(rows_all),
            "n_prediction_rows_test_split": len(rows), "split": "test" if art else "all (no split artifacts)"}
    modes = sorted({r["mode"] for r in rows})
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in modes}

    metrics = {m: system_metrics(rs, banned) for m, rs in by_mode.items()}
    _dump(run_dir / "metrics.json", {"systems": metrics}, meta)

    cats = {m: {c: system_metrics([r for r in rs if r["category"] == c], banned)
                for c in sorted({r["category"] for r in rs})} for m, rs in by_mode.items()}
    _dump(run_dir / "category_metrics.json", {"systems": cats}, meta)
    crs = {m: {c: {k: v for k, v in system_metrics([r for r in rs if str(r.get("course")) == c], banned).items()
                   if not isinstance(v, (list, dict))}
               for c in sorted({str(r.get("course")) for r in rs})} for m, rs in by_mode.items()}
    _dump(run_dir / "course_metrics.json", {"systems": crs}, meta)

    pa = {m: compute_partial_answerable_eval([r for r in rs]) for m, rs in by_mode.items()}
    _dump(run_dir / "partial_answerable_eval.json", {"systems": pa}, meta)

    full_prof = [r for r in rows if r["mode"] == "full_system"]
    sm = {"full_system_profile": score_metrics(full_prof) if full_prof else {}}
    if full_prof:
        y = [int(r["expected_action"] == "ANSWER") for r in full_prof]
        s = [r["answerability_score"] for r in full_prof]
        # Calibrator comparison: fit on validation rows from the *signals*-independent split is
        # done in fit_artifacts (score model). Here only report calibration of the delivered score.
        sm["note"] = "calibrators are compared in calibration_compare.json when validation scores exist"
    _dump(run_dir / "calibration.json", sm, meta)

    if (run_dir / "retrieval_trace.jsonl").exists():
        ret = evaluate_retrieval(_read(run_dir / "retrieval_trace.jsonl"))
        _dump(run_dir / "retrieval_metrics.json", ret, {**meta, "scope": "all questions with gold evidence (no fitted parameters involved)"})

    ref = "full_system"
    tests, cis = {}, {}
    for m in modes:
        for name, fn in (("expected_action_correct", decision_correct),):
            cis.setdefault(m, {})[name] = question_level_mean_std_ci(by_mode[m], fn)
            if m != ref and ref in by_mode:
                tests[f"{ref}_vs_{m}"] = compare_systems_by_question(by_mode[m], by_mode[ref], fn)
    _dump(run_dir / "statistical_tests.json",
          {"comparison": "A=other system, B=full_system; diff = B - A on expected-action accuracy",
           "unit": "question (seeds collapsed; seeds are not independent observations)", "tests": tests}, meta)
    _dump(run_dir / "confidence_intervals.json", {"systems": cis}, meta)
    _dump(run_dir / "policy_instrumentation.json", policy_instrumentation(rows), meta)
    _dump(run_dir / "ablation_equivalence.json", {"identical_pairs": ablation_equivalence(rows)}, meta)
    (run_dir / "consistency.json").write_text(json.dumps(
        {"predictions_sha256": sha, "generated_files": sorted(p.name for p in run_dir.glob("*.json"))}, indent=2))
    return {"metrics": metrics, "meta": meta}
