"""Prepare the evaluation dataset for unanswerable-question detection.

Primary dataset: SQuAD 2.0 (Rajpurkar et al.), which pairs every question with a
context paragraph and an ``is_impossible`` flag - exactly the answerable /
unanswerable distinction this research targets.

The script:
  1. Loads SQuAD 2.0 (via the HuggingFace ``datasets`` library, or by downloading
     the official JSON as a fallback).
  2. Samples a balanced subset of answerable and unanswerable questions.
  3. Writes the supporting context paragraphs to the RAG corpus directory so the
     retriever can index them.
  4. Writes a labelled question file (JSONL) consumed by ``run_eval.py``.

Usage:
    python -m eval.prepare_data --limit 200 --balance
    python -m eval.prepare_data --limit 200 --corpus-dir data/corpus --clean-corpus
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

SQUAD_V2_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_V2_TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"


def _load_via_hf(split: str) -> List[Dict]:
    """Load SQuAD 2.0 through the HuggingFace datasets library."""
    from datasets import load_dataset

    hf_split = "validation" if split == "dev" else "train"
    ds = load_dataset("rajpurkar/squad_v2", split=hf_split)
    examples = []
    for row in ds:
        answers = row.get("answers", {}) or {}
        gold = list(answers.get("text", []) or [])
        examples.append(
            {
                "id": row["id"],
                "question": row["question"].strip(),
                "context": row["context"].strip(),
                "title": row.get("title", ""),
                "answerable": len(gold) > 0,
                "gold_answers": gold,
            }
        )
    return examples


def _load_via_url(split: str) -> List[Dict]:
    """Download and parse the official SQuAD 2.0 JSON as a fallback."""
    import httpx

    url = SQUAD_V2_DEV_URL if split == "dev" else SQUAD_V2_TRAIN_URL
    print(f"[prepare_data] Downloading SQuAD 2.0 ({split}) from {url} ...")
    resp = httpx.get(url, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()

    examples = []
    for article in data.get("data", []):
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "").strip()
            for qa in para.get("qas", []):
                gold = [a["text"] for a in qa.get("answers", [])]
                is_impossible = qa.get("is_impossible", len(gold) == 0)
                examples.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"].strip(),
                        "context": context,
                        "title": title,
                        "answerable": not is_impossible,
                        "gold_answers": gold,
                    }
                )
    return examples


def load_squad_v2(split: str, source: str) -> List[Dict]:
    if source in ("hf", "auto"):
        try:
            return _load_via_hf(split)
        except Exception as e:  # noqa: BLE001
            if source == "hf":
                raise
            print(f"[prepare_data] HuggingFace load failed ({e}); falling back to URL download.")
    return _load_via_url(split)


def sample_examples(examples: List[Dict], limit: int, balance: bool, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    if not balance:
        rng.shuffle(examples)
        return examples[:limit] if limit else examples

    answerable = [e for e in examples if e["answerable"]]
    unanswerable = [e for e in examples if not e["answerable"]]
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)

    half = (limit // 2) if limit else min(len(answerable), len(unanswerable))
    selected = answerable[:half] + unanswerable[:half]
    rng.shuffle(selected)
    return selected


def write_corpus(examples: List[Dict], corpus_dir: Path, clean: bool) -> int:
    """Write the unique supporting contexts as a corpus file for the retriever."""
    corpus_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        for existing in corpus_dir.glob("*.txt"):
            existing.unlink()
        for existing in corpus_dir.glob("*.md"):
            existing.unlink()
        print(f"[prepare_data] Cleaned existing corpus files in {corpus_dir}")

    seen = set()
    unique_contexts = []
    for e in examples:
        ctx = e["context"]
        if ctx and ctx not in seen:
            seen.add(ctx)
            unique_contexts.append(ctx)

    # The retriever chunks by blank-line-separated paragraphs, so join with "\n\n".
    out_file = corpus_dir / "squad_v2_contexts.txt"
    out_file.write_text("\n\n".join(unique_contexts), encoding="utf-8")
    print(f"[prepare_data] Wrote {len(unique_contexts)} unique contexts to {out_file}")
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
                "source": "squad_v2",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    n_ans = sum(1 for e in examples if e["answerable"])
    n_unans = len(examples) - n_ans
    print(
        f"[prepare_data] Wrote {len(examples)} questions to {out_path} "
        f"(answerable={n_ans}, unanswerable={n_unans})"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare SQuAD 2.0 evaluation data.")
    parser.add_argument("--split", choices=["dev", "train"], default="dev")
    parser.add_argument("--source", choices=["auto", "hf", "url"], default="auto")
    parser.add_argument("--limit", type=int, default=200, help="Total questions to sample (0 = all).")
    parser.add_argument("--balance", action="store_true", help="Balance answerable/unanswerable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corpus-dir", default="data/corpus")
    parser.add_argument("--clean-corpus", action="store_true", help="Remove existing corpus files first.")
    parser.add_argument("--out", default="data/eval/questions.jsonl")
    args = parser.parse_args()

    print(f"[prepare_data] Loading SQuAD 2.0 ({args.split}) via source={args.source} ...")
    examples = load_squad_v2(args.split, args.source)
    print(f"[prepare_data] Loaded {len(examples)} raw examples.")

    selected = sample_examples(examples, args.limit, args.balance, args.seed)
    print(f"[prepare_data] Selected {len(selected)} examples.")

    write_corpus(selected, Path(args.corpus_dir), args.clean_corpus)
    write_questions(selected, Path(args.out))

    # Remove any stale FAISS index so the retriever rebuilds from the new corpus.
    for stale in [Path("app/index.faiss"), Path("app/index_meta.pkl")]:
        if stale.exists():
            stale.unlink()
            print(f"[prepare_data] Removed stale index {stale} (will rebuild on next run).")

    print("[prepare_data] Done.")


if __name__ == "__main__":
    main()
