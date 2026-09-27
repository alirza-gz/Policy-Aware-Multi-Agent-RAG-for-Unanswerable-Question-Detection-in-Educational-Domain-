# Labeling / Annotation Guidelines — Educational RAG Dataset

## Scope

These guidelines define how to label questions relative to the **indexed educational corpus** in `data/corpus_edu_ds/` only.

Governance actions remain exactly three: **ANSWER / CLARIFY / ABSTAIN**.

## Primary categories

### 1. `answerable` → ANSWER

Use when the corpus contains enough information to answer the question **correctly and completely**.

Requirements:

- At least one `evidence_spans` entry with a stable `passage_id`.
- `gold_answers` must be supportable by those spans alone (no outside knowledge).
- Prefer paraphrases of the question; avoid copying the passage verbatim as the question text when possible.

Do **not** mark answerable if only a partial fragment is present → use `underspecified`.

### 2. `out_of_knowledge` → ABSTAIN (advisor: Unanswerable)

On-topic for one of the five courses, but the required fact is **not** in the corpus.

Examples: course logistics, unpublished dataset details, thresholds never stated, instructor-specific facts.

Check: a diligent reader of the full course documents still cannot answer.

### 3. `out_of_scope` → ABSTAIN (advisor: Out-of-Domain)

Clearly outside the five Data Science–related courses / educational DS domain.

Examples: geography trivia, sports, literature, cooking, unrelated history.

Check: even perfect retrieval should not find a sufficient educational answer in this corpus. Avoid OOD items that accidentally match corpus keywords tightly enough to become quasi-answerable.

### 4. `false_presupposition` → ABSTAIN (advisor: False Premise)

The question embeds a **false assumption** that is contradicted or clearly unsupported by the corpus, while remaining plausible.

Construction: take a true corpus fact → invert it → ask why/how based on the false claim.

Fill `false_premise` with a short note naming the false assumption.

The system should **ABSTAIN**, not “correct the premise and answer” as the gold action for this thesis.

### 5. `underspecified` → CLARIFY (advisor: Partially Answerable / Underspecified)

Operational merge for this thesis: questions that are ambiguous, underspecified, or only partially supported such that a responsible tutor should **ask for clarification** rather than give a full answer.

Fill `missing_aspects` describing what is missing (referent, metric, dataset, hypothesis, etc.).

Do **not** invent a `PARTIAL_ANSWER` action.

## Optional secondary category

### `low_confidence` → ABSTAIN

Thin-evidence / high-uncertainty items for stress tests only. **Not** part of the 2,500 primary quota. Not included in the 250 pilot.

## Decision tree (annotator)

1. Is the question outside the DS educational domain of this corpus? → `out_of_scope`
2. Else, does it rest on a false assumption relative to the corpus? → `false_presupposition`
3. Else, is the question too incomplete / only partially supportable to answer fully? → `underspecified`
4. Else, is the full answer grounded in cited passages? → `answerable`
5. Else → `out_of_knowledge`

## Consistency rules

| If category is… | then |
|-----------------|------|
| `answerable` | `answerable=true`, `expected_action=ANSWER`, non-empty evidence |
| `underspecified` | `answerable=false`, `expected_action=CLARIFY` |
| any other primary | `answerable=false`, `expected_action=ABSTAIN` |

## Quality checks before acceptance

- No near-duplicate questions (paraphrase duplicates across the set).
- Answerable items: evidence passage actually entails the gold answer.
- Unanswerable/OOD/false-premise: confirm no other corpus passage secretly answers them.
- Balanced difficulty and mix of factual vs reasoning items among answerable questions.
- Dual annotation on a stratified sample for the full dataset; adjudicate disagreements.

## Pilot quotas (per course / 50 items)

| Category | Count |
|----------|------:|
| answerable | 20 |
| out_of_knowledge | 8 |
| out_of_scope | 8 |
| false_presupposition | 7 |
| underspecified | 7 |
| **Total** | **50** |

Full-scale quotas (per course / 500): 200 / 75 / 75 / 75 / 75.
