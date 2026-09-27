"""Validate pilot dataset labels and retrieval sanity against the edu DS corpus.

Usage (from repo root, after indexing data/corpus_edu_ds):
    python scripts/validate_edu_pilot.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.run_experiments import expected_action, question_category

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "eval" / "educational_questions_pilot.jsonl"
CORPUS = ROOT / "data" / "corpus_edu_ds"


def load_questions():
    rows = []
    with PILOT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_label_consistency(rows):
    errors = []
    for r in rows:
        cat = question_category(r)
        exp = expected_action(r)
        if r.get("expected_action") != exp:
            errors.append(f"{r['id']}: stored expected_action={r.get('expected_action')} != recomputed {exp}")
        if cat == "answerable":
            if not r.get("answerable"):
                errors.append(f"{r['id']}: answerable category but answerable=false")
            if not r.get("evidence_spans"):
                errors.append(f"{r['id']}: answerable missing evidence_spans")
            if not r.get("gold_answers"):
                errors.append(f"{r['id']}: answerable missing gold_answers")
        else:
            if r.get("answerable"):
                errors.append(f"{r['id']}: non-answerable category but answerable=true")
    return errors


def check_evidence_exists(rows):
    errors = []
    corpus_passages = {}
    for p in CORPUS.glob("*.md"):
        paras = [x.strip() for x in p.read_text(encoding="utf-8").split("\n\n") if x.strip()]
        for i, text in enumerate(paras):
            corpus_passages[f"{p.name}#p{i}"] = text
    for r in rows:
        if r.get("category") != "answerable":
            continue
        for span in r.get("evidence_spans") or []:
            pid = span.get("passage_id")
            if pid not in corpus_passages:
                errors.append(f"{r['id']}: missing passage {pid}")
            elif span.get("text", "").strip() and span["text"].strip() not in corpus_passages[pid]:
                # allow exact equality
                if span["text"].strip() != corpus_passages[pid]:
                    errors.append(f"{r['id']}: evidence text mismatch for {pid}")
    return errors


def retrieval_sanity(rows, top_k: int = 5):
    from app.agents.retriever_agent import RetrieverAgent

    retriever = RetrieverAgent()
    answerable = [r for r in rows if r.get("category") == "answerable"]
    hits = 0
    miss_ids = []
    for r in answerable:
        gold_ids = {e["passage_id"] for e in (r.get("evidence_spans") or [])}
        passages = retriever.retrieve(r["question"], top_k=top_k)
        retrieved_ids = {p.get("id") for p in passages}
        if gold_ids & retrieved_ids:
            hits += 1
        else:
            miss_ids.append(
                {
                    "id": r["id"],
                    "question": r["question"],
                    "gold": sorted(gold_ids),
                    "retrieved": [p.get("id") for p in passages],
                    "scores": [round(float(p.get("score", 0)), 4) for p in passages],
                }
            )
    return {
        "n_answerable": len(answerable),
        "hit_at_k": hits,
        "hit_rate": round(hits / len(answerable), 4) if answerable else 0.0,
        "misses": miss_ids,
    }


def main():
    rows = load_questions()
    print(f"Loaded {len(rows)} questions")
    print("Per course:", dict(Counter(r["course"] for r in rows)))
    print("Per category:", dict(Counter(r["category"] for r in rows)))
    print("Per expected_action:", dict(Counter(r["expected_action"] for r in rows)))

    by_course_cat = defaultdict(Counter)
    for r in rows:
        by_course_cat[r["course"]][r["category"]] += 1
    print("\nPer course x category:")
    for course in sorted(by_course_cat):
        print(f"  {course}: {dict(by_course_cat[course])}")

    errors = check_label_consistency(rows) + check_evidence_exists(rows)
    if errors:
        print(f"\nLABEL ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(" -", e)
    else:
        print("\nLabel/action consistency: OK")
        print("Evidence passage existence: OK")

    print("\nRunning retrieval sanity (hit@5 for answerable)...")
    sanity = retrieval_sanity(rows, top_k=5)
    print(f"Answerable hit@5: {sanity['hit_at_k']}/{sanity['n_answerable']} ({sanity['hit_rate']})")
    out = ROOT / "results" / "edu_pilot" / "retrieval_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")
    if sanity["misses"]:
        print(f"Miss examples ({min(5, len(sanity['misses']))}):")
        for m in sanity["misses"][:5]:
            print(f"  {m['id']}: gold={m['gold']} retrieved={m['retrieved']}")


if __name__ == "__main__":
    main()
