"""Rebuild replayable *signals* from an older predictions file (decision-layer replay only).

Used when the LLM/index are unavailable. The stored retrieval confidence, passages and
reasoner outputs are real outputs of the earlier run; only the decision layer is re-executed.
Limitations (flagged in the signals so downstream code cannot over-interpret them):
  * retrieval candidate lists are depth-limited -> retrieval metrics are NOT produced;
  * reasoner outputs lack the new answerability_label/missing_information fields;
  * the original run's reasoner (e.g. mock) is whatever produced those rows.
"""
from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path


def main(pred_path: str, dataset_path: str, out_path: str) -> int:
    ds = {}
    for l in open(dataset_path, encoding="utf-8"):
        if l.strip():
            q = json.loads(l); ds[q["id"]] = q
    seen = OrderedDict()
    for l in open(pred_path, encoding="utf-8"):
        r = json.loads(l)
        key = (r["id"], r["seed"], r["retrieval_mode"], bool(r["reranked"]))
        if key in seen:
            continue
        q = dict(ds[r["id"]])
        q["category"], q["expected_action"] = r["category"], r["expected_action"]
        seen[key] = {
            "seed": r["seed"], "retrieval_mode": r["retrieval_mode"], "reranked": bool(r["reranked"]),
            "retriever_confidence": r["retriever_confidence"], "retrieved_ids": r["retrieved_ids"],
            "passages": [{"id": i, "text": t, "score": 0.0} for i, t in zip(r["retrieved_ids"], r["passage_texts"])],
            "dense": [], "sparse": [], "fused": [], "pre_rerank": [], "ranked_deep": [],
            "replayed_from_predictions": True,
            "reasoning_result": {"answer": r["raw_answer"], "is_answerable": r["model_is_answerable"],
                                 "needs_clarification": r["model_needs_clarification"],
                                 "answerability_confidence": r["answerability_confidence"],
                                 "confidence": r["reasoner_confidence"], "clarification_question": ""},
            "question": q,
        }
    with open(out_path, "w", encoding="utf-8") as f:
        for s in seen.values():
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(seen)} signals -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
