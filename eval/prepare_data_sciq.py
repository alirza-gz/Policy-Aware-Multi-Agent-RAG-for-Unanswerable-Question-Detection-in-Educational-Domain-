"""Prepare a REAL educational-domain evaluation set from SciQ.

SciQ (Welbl et al., 2017; https://huggingface.co/datasets/allenai/sciq) is a
real dataset of 13,679 crowdsourced science exam questions (physics,
chemistry, biology, ...) with a supporting passage for most questions.
License: CC-BY-NC-3.0 -- fine for academic/thesis use, not for commercial
redistribution.

SciQ itself has NO unanswerable questions -- every question ships with its
own correct supporting passage. To get genuine answerable/unanswerable pairs
for this research, we use the same negative-sampling technique SQuAD 2.0's
authors cite (Clark & Gardner, 2017): pair a question with a *different*
question's passage instead of its own. Because that passage was written to
answer a different, unrelated question, the original question has no answer
in it -- a real "no evidence in the corpus" case, not a synthetic label.

A light lexical safety check skips pairings where the correct answer string
happens to already appear in the borrowed passage (to avoid accidentally
creating an easy / still-answerable "unanswerable" example).

Usage:
    python -m eval.prepare_data_sciq --limit 200 --balance --clean-corpus
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def _load_sciq() -> List[Dict]:
    """Load SciQ (train split) through the HuggingFace datasets library."""
    from datasets import load_dataset

    ds = load_dataset("allenai/sciq", split="train")
    examples = []
    for i, row in enumerate(ds):
        support = (row.get("support") or "").strip()
        question = (row.get("question") or "").strip()
        correct = (row.get("correct_answer") or "").strip()
        if not support or not question or not correct:
            # Skip rows without a usable supporting passage.
            continue
        examples.append(
            {
                "id": f"sciq-{i}",
                "question": question,
                "support": support,
                "correct_answer": correct,
            }
        )
    return examples


def build_examples(raw: List[Dict], limit: int, seed: int) -> List[Dict]:
    """Build a balanced list of real-answerable + cross-paired-unanswerable examples."""
    rng = random.Random(seed)
    pool = list(raw)
    rng.shuffle(pool)

    half = (limit // 2) if limit else len(pool) // 2
    half = min(half, len(pool) // 2)

    answerable_src = pool[:half]
    unanswerable_src = pool[half : half * 2]

    examples: List[Dict] = []

    # --- Answerable: each question paired with its own real support. ---
    for row in answerable_src:
        examples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "context": row["support"],
                "answerable": True,
                "gold_answers": [row["correct_answer"]],
                "title": "sciq",
                "source": "sciq",
            }
        )

    # --- Unanswerable: each question paired with a DIFFERENT row's support. ---
    donors = [r for r in pool if r not in unanswerable_src]
    if not donors:
        donors = answerable_src  # fallback if the pool is too small
    for row in unanswerable_src:
        # Find a donor passage that doesn't already contain this question's
        # correct answer (avoids accidentally-still-answerable pairs).
        donor = None
        for _ in range(10):
            candidate = rng.choice(donors)
            if row["correct_answer"].lower() not in candidate["support"].lower():
                donor = candidate
                break
        if donor is None:
            continue  # give up on this one rather than risk a bad label
        examples.append(
            {
                "id": row["id"],
                "question": row["question"],
                "context": donor["support"],
                "answerable": False,
                "gold_answers": [],
                "title": "sciq",
                "source": "sciq_cross_paired",
            }
        )

    rng.shuffle(examples)
    return examples


def write_corpus(examples: List[Dict], corpus_dir: Path, clean: bool) -> int:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        for existing in corpus_dir.glob("*.txt"):
            existing.unlink()
        for existing in corpus_dir.glob("*.md"):
            existing.unlink()
        print(f"[prepare_data_sciq] Cleaned existing corpus files in {corpus_dir}")

    seen = set()
    unique_contexts = []
    for e in examples:
        ctx = e["context"]
        if ctx and ctx not in seen:
            seen.add(ctx)
            unique_contexts.append(ctx)

    out_file = corpus_dir / "sciq_contexts.txt"
    out_file.write_text("\n\n".join(unique_contexts), encoding="utf-8")
    print(f"[prepare_data_sciq] Wrote {len(unique_contexts)} unique contexts to {out_file}")
    return len(unique_contexts)


def write_questions(examples: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in examples:
            record = {
                "id": e["id"],
                "question": e["question"],
                "answerable": e["answerable"],
                "gold_answers": e["gold_answers"],
                "title": e.get("title", ""),
                "source": e.get("source", "sciq"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    n_ans = sum(1 for e in examples if e["answerable"])
    print(f"[prepare_data_sciq] Wrote {len(examples)} questions to {out_path} "
          f"({n_ans} answerable / {len(examples) - n_ans} unanswerable)")


def main():
    parser = argparse.ArgumentParser(description="Prepare a real SciQ-based eval set.")
    parser.add_argument("--limit", type=int, default=200, help="Total questions (0 = all usable).")
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--questions-out", default="data/eval/sciq_questions.jsonl")
    parser.add_argument("--clean-corpus", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("[prepare_data_sciq] Loading SciQ via HuggingFace datasets ...")
    raw = _load_sciq()
    print(f"[prepare_data_sciq] Loaded {len(raw)} usable SciQ rows (with support + answer).")

    examples = build_examples(raw, limit=args.limit, seed=args.seed)
    write_corpus(examples, Path(args.corpus_dir), clean=args.clean_corpus)
    write_questions(examples, Path(args.questions_out))


if __name__ == "__main__":
    main()