# Educational RAG Pilot — JSONL Schema

Version: `edu_rag_pilot_v1`  
Compatible with: `eval/run_experiments.py` (reads core fields; extra fields are ignored safely)

## Required fields (evaluation harness)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique question id (primary key for eval) |
| `question` | string | Question text |
| `answerable` | bool | `true` only for fully answerable items |
| `category` | string | One of the primary taxonomy labels below |
| `gold_answers` | string[] | Acceptable short answers; empty if not answerable |
| `title` | string | Course id (legacy field used by existing reports) |

## Taxonomy → governance mapping (frozen for this thesis)

| Advisor label | `category` | `answerable` | `expected_action` |
|---------------|------------|--------------|-------------------|
| Answerable | `answerable` | true | `ANSWER` |
| Unanswerable | `out_of_knowledge` | false | `ABSTAIN` |
| Out-of-Domain | `out_of_scope` | false | `ABSTAIN` |
| False Premise | `false_presupposition` | false | `ABSTAIN` |
| Partially Answerable / Underspecified | `underspecified` | false | `CLARIFY` |

Optional secondary stress category (not in pilot quotas): `low_confidence` → `ABSTAIN`.

Note: `eval/run_experiments.expected_action()` recomputes gold actions from `answerable` + `category` (`underspecified` → CLARIFY). The stored `expected_action` field must match that rule.

## Extended metadata (dataset quality / traceability)

| Field | Type | Required when |
|-------|------|----------------|
| `question_id` | string | Always (mirrors `id`) |
| `course` | string | Always (`data_science`, `statistics`, `machine_learning`, `data_mining`, `probability`) |
| `expected_action` | string | Always (`ANSWER` / `CLARIFY` / `ABSTAIN`) |
| `source` | string | Always (e.g. `edu_rag_pilot_v1`) |
| `source_documents` | string[] | Answerable (filenames in `data/corpus_edu_ds/`) |
| `source_type` | string | Answerable when known (`lecture_notes`, `textbook`, `slides`, `article`, `oer`) |
| `evidence_spans` | object[] | Answerable: `{doc_id, passage_id, text}` |
| `reference_answer` | string | Answerable (canonical long/short answer) |
| `difficulty` | string | Preferred: `easy` / `medium` / `hard` |
| `question_type` | string | Preferred: `factual` / `reasoning` / `definition` / `comparison` |
| `construction_method` | string | Preferred: `corpus_grounded` / `cross_doc_gap` / `ood` / `premise_flip` / `underspecify` |
| `false_premise` | string | `false_presupposition` items |
| `missing_aspects` | string | `underspecified` items |
| `split` | string | `pilot` / `dev` / `test` |

### `evidence_spans` object

```json
{
  "doc_id": "statistics_03_hypothesis_slides.md",
  "passage_id": "statistics_03_hypothesis_slides.md#p2",
  "text": "A p-value is the probability, under the null hypothesis..."
}
```

`passage_id` must match indexer IDs: `{filename}#p{i}` after blank-line paragraph splits.

## Example (answerable)

```json
{
  "id": "pilot-stat-a14",
  "question_id": "pilot-stat-a14",
  "question": "What is a p-value?",
  "answerable": true,
  "category": "answerable",
  "expected_action": "ANSWER",
  "gold_answers": ["the probability, under the null hypothesis, of obtaining a test statistic at least as extreme as the one observed"],
  "title": "statistics",
  "course": "statistics",
  "source": "edu_rag_pilot_v1",
  "source_documents": ["statistics_03_hypothesis_slides.md"],
  "source_type": "slides",
  "evidence_spans": [{"doc_id": "statistics_03_hypothesis_slides.md", "passage_id": "statistics_03_hypothesis_slides.md#p2", "text": "..."}],
  "reference_answer": "the probability, under the null hypothesis, of obtaining a test statistic at least as extreme as the one observed",
  "difficulty": "hard",
  "question_type": "definition",
  "construction_method": "corpus_grounded",
  "split": "pilot"
}
```

## Files

- Pilot questions: `data/eval/educational_questions_pilot.jsonl`
- Smoke/regression (unchanged): `data/eval/educational_questions.jsonl`
- Corpus: `data/corpus_edu_ds/`
- Manifest: `data/corpus_manifest.json`
