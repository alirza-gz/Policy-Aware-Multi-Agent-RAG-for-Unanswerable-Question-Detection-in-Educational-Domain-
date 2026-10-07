"""Offline BM25-only retrieval evaluation (no embedding model / GPU needed).

Rebuilds the edu2 corpus from the dataset's ``evidence_passage`` texts exactly as
eval.link_evidence_ids does (into a scratch dir; the dataset file is NOT modified),
verifies that the dataset's stored ``evidence_ids`` agree with the rebuilt ids, then
evaluates BM25 with eval.retrieval_eval. Dense / Hybrid / Hybrid+Reranker require the
embedding + cross-encoder models and are produced by run_evaluation.py, not here.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from app.agents.bm25_index import BM25Index
from eval.link_evidence_ids import _norm, build_corpus_from_evidence
from eval.retrieval_eval import evaluate_retrieval


def main(dataset="data/eval/educational_v2_questions.jsonl", out=None) -> dict:
    rows = [json.loads(l) for l in open(dataset, encoding="utf-8") if l.strip()]
    with tempfile.TemporaryDirectory() as td:
        text_to_id, course_ids = build_corpus_from_evidence(rows, Path(td))
        # corpus files are paragraph-chunked exactly like RetrieverAgent._load_corpus
        meta = []
        for p in sorted(Path(td).glob("*.txt")):
            for i, para in enumerate(x.strip() for x in p.read_text(encoding="utf-8").split("\n\n") if x.strip()):
                meta.append({"id": f"{p.name}#p{i}", "text": para, "source": p.name})
    stored = [r for r in rows if r.get("evidence_ids")]
    agree = sum(1 for r in stored if text_to_id.get(_norm(r["evidence_passage"])) == r["evidence_ids"][0])
    bm = BM25Index(meta)
    trace = []
    for r in rows:
        ids = [h["id"] for h in bm.search(r["question"], top_k=20)]
        trace.append({"id": r["id"], "gold_ids": r.get("evidence_ids") or [], "bm25_ids": ids})
    res = evaluate_retrieval(trace)
    res["corpus_size_passages"] = len(meta)
    res["evidence_id_agreement"] = {"stored_with_ids": len(stored), "agree_with_rebuilt_corpus": agree}
    res["scope_note"] = ("BM25 only. Corpus = unique evidence passages of the dataset (held-out passages excluded), "
                         "so this corpus is smaller and easier than a real course corpus.")
    if out:
        Path(out).write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    print(json.dumps(main(out=sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
