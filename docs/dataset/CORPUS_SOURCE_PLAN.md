# Educational DS Corpus — Source Plan (Pilot → Full)

## Purpose

Dedicated educational corpus for Policy-Aware Multi-Agent RAG evaluation.
Primary experimental corpus for the thesis educational track (not SciQ/SQuAD).

## Courses

| course id         | Display name              |
|-------------------|---------------------------|
| data_science      | Introduction to Data Science |
| statistics        | Statistics                |
| machine_learning  | Machine Learning          |
| data_mining       | Data Mining               |
| probability       | Probability               |

## Layout

Indexer loads **top-level** `.txt` / `.md` only (no recursion). Files live flat in:

```
data/corpus_edu_ds/
  {course}_{nn}_{topic}_{source_tag}.md
```

Passage IDs (stable): `{filename}#p{i}` where `i` is the 0-based paragraph index
after splitting on blank lines (same rule as `indexer.py` / `RetrieverAgent._load_corpus`).

Legacy smoke corpus remains at `data/corpus/edu_*.txt` (unchanged).
Config for the educational track: `CORPUS_DIR: data/corpus_edu_ds`.

## Source-type targets (per course)

Each course has ≥3 source types among:

- `lecture_notes`
- `textbook`
- `slides`
- `article`
- `oer`

Pilot corpus: **4 documents × 5 courses = 20 files** (synthetic educational notes
written for this thesis pilot; licenses recorded as `CC0-1.0` / original).

Full-scale plan: replace/extend with openly licensed materials (OpenIntro,
LibreTexts, open university notes, etc.) and update `data/corpus_manifest.json`.

## Manifest fields

See `data/corpus_manifest.json`:

`doc_id`, `course`, `source_type`, `title`, `license`, `url`, `version_date`, `filename`

## Scaling notes

- Pilot: enough paragraphs to support 20 answerable questions/course with distinct evidence.
- Full (~200 answerable/course): expand to ~8–15 docs/course and longer chapters.
- Never mix SciQ/SQuAD context dumps into `corpus_edu_ds` for official runs.
