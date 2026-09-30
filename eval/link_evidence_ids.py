"""Add stable evidence passage IDs to educational_v2 without changing labels.

Rebuilds ``data/corpus/edu2_*.txt`` from the unique ``evidence_passage`` texts
already present in the JSONL (held-out passages are intentionally excluded).
Then writes ``evidence_ids`` onto each question that has supporting evidence.

Does NOT modify: category, gold_answerable, expected_action, gold_answers,
question text, or any governance label.

Usage:
    python -m eval.link_evidence_ids
    python -m eval.link_evidence_ids --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DEFAULT_QUESTIONS = Path("data/eval/educational_v2_questions.jsonl")
DEFAULT_CORPUS = Path("data/corpus")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _passage_key(text: str) -> str:
    return hashlib.sha1(_norm(text).encode("utf-8")).hexdigest()[:12]


def build_corpus_from_evidence(
    rows: List[Dict], corpus_dir: Path
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Write edu2_{course}.txt files; return (norm_text -> id, course -> ids)."""
    # Preserve first-seen order per course for stable #p{i} indices.
    per_course: Dict[str, List[str]] = defaultdict(list)
    seen_per_course: Dict[str, set] = defaultdict(set)

    for r in rows:
        ev = r.get("evidence_passage")
        if not ev:
            continue
        course = str(r.get("course", "unknown") or "unknown")
        key = _norm(ev)
        if key in seen_per_course[course]:
            continue
        seen_per_course[course].add(key)
        per_course[course].append(ev.strip())

    corpus_dir.mkdir(parents=True, exist_ok=True)
    text_to_id: Dict[str, str] = {}
    course_ids: Dict[str, List[str]] = {}

    for course, passages in sorted(per_course.items()):
        out = corpus_dir / f"edu2_{course}.txt"
        out.write_text("\n\n".join(passages), encoding="utf-8")
        ids = []
        for i, text in enumerate(passages):
            pid = f"edu2_{course}.txt#p{i}"
            text_to_id[_norm(text)] = pid
            ids.append(pid)
        course_ids[course] = ids
        print(f"[link_evidence_ids] Wrote {out} ({len(passages)} passages)")

    return text_to_id, course_ids


def annotate_rows(rows: List[Dict], text_to_id: Dict[str, str]) -> Dict[str, int]:
    """Attach evidence_ids in-place. Returns match stats."""
    stats = {"with_evidence": 0, "linked": 0, "unlinked": 0, "no_evidence": 0}
    for r in rows:
        ev = r.get("evidence_passage")
        if not ev:
            stats["no_evidence"] += 1
            # Keep field explicit for schema stability.
            r["evidence_ids"] = []
            continue
        stats["with_evidence"] += 1
        pid = text_to_id.get(_norm(ev))
        if pid:
            r["evidence_ids"] = [pid]
            r["evidence_id"] = pid  # convenience singular
            stats["linked"] += 1
        else:
            r["evidence_ids"] = []
            stats["unlinked"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows: List[Dict] = []
    with open(args.questions, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Snapshot labels to prove they are unchanged.
    label_snapshot = [
        (
            r.get("id"),
            r.get("category"),
            r.get("gold_answerable"),
            r.get("expected_action"),
            r.get("question"),
            tuple(r.get("gold_answers") or []),
        )
        for r in rows
    ]

    text_to_id, _ = build_corpus_from_evidence(rows, args.corpus_dir)
    stats = annotate_rows(rows, text_to_id)

    for i, r in enumerate(rows):
        snap = label_snapshot[i]
        assert (r.get("id"), r.get("category"), r.get("gold_answerable"),
                r.get("expected_action"), r.get("question"),
                tuple(r.get("gold_answers") or [])) == snap, "Label mutation detected"

    print(f"[link_evidence_ids] Stats: {stats}")
    print(f"[link_evidence_ids] Unique corpus passages: {len(text_to_id)}")

    if args.dry_run:
        print("[link_evidence_ids] Dry-run: JSONL not written")
        return

    with open(args.questions, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[link_evidence_ids] Updated {args.questions}")


if __name__ == "__main__":
    main()
