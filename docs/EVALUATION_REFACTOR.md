# Evaluation refactor: findings, changes, and what is (not) established

**Status.** Code, tests and a reproducible entry point are done. A new *end-to-end* run with the real
LLM (Ollama qwen2.5:3b) + embedder + reranker has **not** been executed: the sandbox this work was done in
has no Ollama, no HuggingFace access, no FAISS index and no corpus. Every number below was produced by
actually executing code, and each says exactly what was executed. Nothing is estimated.

## 1. Findings in the original system (Part 1, root causes)

| # | Question | Answer (file) |
|---|----------|---------------|
| 1 | Where are predictions generated? | `eval/run_experiments.py::apply_systems_for_profile` (experiments) and legacy `eval/run_eval.py` (SciQ). Reasoner output: `ReasoningAgent.reason` |
| 2 | Final action | `GovernanceAgent.decide` (full system), `eval/baselines.py::decide_*` (others) |
| 3 | Answerability score | The LLM's self-reported JSON float `answerability_confidence` (mock mode: lexical coverage, and `1-coverage` when it flags unanswerable, so its meaning flips by branch) |
| 4 | Policy decision | Fixed precedence in `GovernanceAgent.decide`; the only "rules" are 9 medical/classified banned phrases matched on the *generated answer* |
| 5 | Clarification trigger | Only if the 3B model sets `needs_clarification`, or confidence falls in [0.3, 0.5) *after* passing the answerable gates |
| 6 | Hallucination | `ANSWER` with non-empty text on `gold_answerable=False`. `unsupported_answer_rate` was a hard alias of it (same number, two names). Includes partially-answerable, false-premise, OOD |
| 7 | Coverage | `answer_rate` = share of all questions with action ANSWER |
| 8 | ROC-AUC score | `1 - answerability_confidence` (LLM self-report, not a probability) |
| 9 | Retrieval relevance | Binary: `evidence_ids` = the passage the question was generated from; rows without it are excluded |
| 10 | Statistical unit | `(id, seed)` pairs pooled, n = 7,494 = 2,498 questions x 3 seeds |

Additional root causes found:

* **Pseudoreplication.** The Ollama sampling seed was hard-coded to 42; "seeds" only shuffled question order. The
  same question evaluated three times was treated as three independent observations, so McNemar p-values and CIs were overconfident.
* **Ablations identical by construction.** `agentic_rag`, `no_policy_agent`, `no_multi_agent` are the same function.
  `no_policy_rules` only blanked the banned-phrase list; none of the 9 phrases occurs in the dataset's gold answers/evidence or
  in the smoke reasoner outputs (0 hits), so it could not change anything.
* **Policy compliance was tautological** (`policy_expected_action` re-applied the governor's own logic to stored fields).
* **BM25 bug.** `BM25Index._idf` returned `(N-df+0.5)/(df+0.5)` without the logarithm (see section 4).
* **Retrieval depth.** Candidate lists were truncated to `top_k=5` before logging, so Recall@10 was undefinable.
* **No `answerability_label`** existed, so Partially Answerable could not be handled as a class.

### Issues 1-4 (Part 13), verified against the files

1. `metrics_comparison.csv` (`eval/analyze.py`) pools all rows; `metrics_summary.csv` (`eval/report.py`) averages per-seed values already rounded to 4 d.p. Different aggregation, same predictions; small differences are expected. Now one aggregator.
2. `trace_report.md` counts "correct abstention" as `ABSTAIN` on `gold_answerable=False`; the confusion matrix uses 3-class `expected_action`. 2,318 = 2,131 (expected ABSTAIN) + 187 (expected CLARIFY, abstained). Definitional, not a bug. Encoded as a regression test.
3. `EVALUATION_REPORT.md` says 82,434 rows (= 7,494 x 11 systems, from `predictions_all.jsonl`, which is **not in the repo**); `results/predictions.jsonl` is the 400-row legacy SciQ run. Different experiments.
4. `sciq_metrics.json` matches `results/predictions.jsonl` exactly (confusion [[96,4],[63,37]]). `metrics.json` ([[91,9],[59,41]]) comes from an earlier run whose predictions are absent; neither file carries seed/model/threshold metadata. Not recoverable; superseded by `config.json` per run.

`results/` also holds files from at least three code versions (`policy_aware`, `vanilla_rag`, `single_agent_rag`, `no_clarification` ...). I did not edit or delete any old result file.

## 2. Changes

| Problem | Files | Implementation |
|---|---|---|
| No answerability classes; CLARIFY fires only on LLM flag | `app/agents/answerability_agent.py` (new) | 5-way label from sub-question coverage, retrieval confidence, ambiguity, reasoner signals; optional NLI for false premise. Never reads gold fields (test enforces) |
| Generic clarification | same | Names supported and unsupported sub-questions and what to provide; `is_generic_clarification` detector |
| Score not a probability | `answerability_agent.ScoreModel`, `eval/calibration.py` | Logistic P(fully answerable) fit on a **validation split grouped by source_document**; Platt/isotonic/temperature compared; ECE, Brier, reliability bins |
| Hard-coded thresholds, leakage risk | `eval/calibration.py::select_thresholds`, `grouped_split` | Threshold chosen on validation only (max F1 subject to a false-accept cap); all reported metrics use the test split |
| Policy effect unobservable | `app/agents/policy_agent.py` (new) | Rules R1-R7 with IDs; per-row `policy_checked/rule_ids/rule_triggered/decision/override/original_action/final_action/reason` |
| Ablations not distinct/measured | `eval/pipeline_v2.py` | `no_policy_agent`, `no_policy_rules`, `no_safety_rules`, `no_guard_rules`, `no_answerability`, `no_governance`; `aggregate.ablation_equivalence` reports measured identical pairs |
| Conflated hallucination metrics | `eval/hallucination.py` | Separate unanswerable-answer, wrong-answer, unsupported, contradiction (NLI only), any; `lexical_groundedness` named as such; `semantic_groundedness` (NLI) implemented |
| Clarification metrics missing | `eval/clarification_metrics.py` | Rate, correct rate, precision, recall, F1, rule-based quality; dedicated Partially-Answerable eval |
| Seeds pooled | `eval/significance.py` | Question-level unit, exact McNemar, Cohen's h, paired bootstrap CI, explicit "statistically significant but practically limited" wording |
| Retrieval not independently evaluated | `eval/retrieval_eval.py`, `hybrid_retriever.py` | Dense/BM25/Hybrid/Hybrid+Rerank, Recall@1/5/10, P@5, Hit@5, MRR, nDCG@10 (binary); reranker top-1/top-5 change counts; full-depth lists logged |
| BM25 IDF | `app/agents/bm25_index.py` | `log(1 + (N-df+0.5)/(df+0.5))` |
| Many scripts computing the same metric | `eval/aggregate.py`, `run_evaluation.py` | One run dir; every aggregate carries the SHA-256 of `predictions.jsonl` |
| LLM seed fixed | `reasoning_agent.py` | `self.seed` set per run; prompt/schema add `answerability_label`, `missing_information` |

Old scripts (`analyze.py`, `report.py`, `run_eval.py`, `run_experiments.py`, `metrics*.py`) are left in place for traceability. **Do not use them for thesis numbers.**

### Rule redundancy (found by tests and the replay)

In the full system the answerability stage already proposes ABSTAIN for out_of_domain/false_premise and CLARIFY for partial,
so **R5 and R7 are redundant there** (they matter only when the answerability stage is bypassed). In the replay below only **R6** (validated
confidence threshold) ever triggered; R1-R4, R5, R7 never did. The Policy Agent's measured effect is therefore the threshold, not the rule set.

## 3. Tests

`python -m pytest -q` -> **116 passed** (50 pre-existing + 66 new): answerability, each of R1-R7 (positive/negative/edge),
instrumentation contract, thresholds, grouped split leakage, calibration, hallucination, groundedness, clarification, RRF fusion,
BM25 IDF, retrieval metrics, statistics; integration (ANSWER/CLARIFY/ABSTAIN, OOD, false premise, partial, service error, full synthetic run
with single-hash check); regression (Issues 2 and 4).

## 4. Executed experiments and results

### A. Offline BM25 retrieval evaluation (real data, real execution)
`python -m eval.run_bm25_retrieval_eval` on educational_v2: 2,498 questions, 1,750 with gold evidence, rebuilt corpus of 1,352 passages
(stored `evidence_ids` agree 1,750/1,750). Corpus = the dataset's evidence passages only, so it is smaller/easier than a real course corpus.

| BM25 | Recall@1 | Recall@5 | Recall@10 | P@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| before IDF fix | 0.7434 | 0.9069 | 0.9446 | 0.1814 | 0.8130 | 0.8434 |
| after IDF fix  | 0.8440 | 0.9451 | 0.9686 | 0.1890 | 0.8899 | 0.9082 |

Consequence: every earlier hybrid-retrieval result used a defective BM25 and must be re-run before it is compared with dense-only.
Dense, Hybrid and Hybrid+Reranker retrieval numbers were **not** produced (need the embedding/cross-encoder models).

### B. Decision-layer replay (NOT an end-to-end run)
Stored signals from `results/smoke/predictions_all.jsonl`: **mock reasoner**, first 100 questions, seed 42, one run. Test split 62 questions (868 prediction rows), validation 38.
Threshold selected on validation: `abstain_below = 0.6333` (val false-accept 0.0357, cap 0.10 met). The score model and threshold were fit on the same 38 validation
questions, so validation numbers are optimistic; the test split is clean. n = 62 is far too small for any claim about the real system.

| System | Binary acc | False accept | False reject | Coverage | Clarify rate | Clarify P / R / F1 |
|---|---|---|---|---|---|---|
| standard_rag | 0.436 | 1.000 | 0.000 | 1.000 | 0 | - |
| agentic_rag | 0.581 | 0.743 | 0.000 | 0.855 | 0 | 0 / 0 / 0 |
| **full_system** | 0.774 | **0.086** | **0.407** | 0.306 | 0.113 | 1.00 / 0.50 / 0.667 |
| no_policy_agent | 0.645 | 0.629 | 0.000 | 0.790 | 0.113 | 1.00 / 0.50 / 0.667 |
| no_answerability | 0.774 | 0.086 | 0.407 | 0.306 | 0 | 0 |

* The full system refuses 41% of answerable questions in this replay. Lower false acceptance is bought with false rejection; it is a trade-off, not a free gain.
* Partially Answerable (n = 14): full system CLARIFY 7 / ABSTAIN 7 (action accuracy 0.50); agentic_rag ANSWER 10 / ABSTAIN 4 (0.00); clarification quality on the 7 = 1.0 (rule-based heuristic). Half of the partials were not detected as partial by the lexical sub-question heuristic and fell to ABSTAIN via R6.
* `no_answerability` has the same ANSWER/non-ANSWER accuracy as the full system: the evidence floor/threshold does the filtering; the answerability stage contributes the CLARIFY-vs-ABSTAIN split.
* Standardised score (n = 62): ROC-AUC 0.8825, PR-AUC (unanswerable positive) 0.9317, ECE 0.1317, Brier 0.1451, vs the legacy LLM-confidence score 0.8101 on the same rows. This is the **mock** reasoner's score; the original Ollama run had ROC-AUC 0.5136. Do not read the replay as evidence that the real system's AUC improved.
* `unsupported_answer_rate` is 0.0 everywhere: the mock reasoner copies passages, so lexical groundedness is ~1 by construction. The metric is uninformative in mock mode.
* Statistics (question level, exact McNemar; B = full_system): vs standard_rag diff +0.226, CI [0.032, 0.403], p = 0.029, Cohen's h = 0.46 (small); vs agentic_rag p = 0.150; vs no_reranker p = 0.453; vs no_hybrid p = 0.500; vs no_policy_agent p = 1.0. With n = 62 almost nothing is distinguishable.
* Measured identical pairs: agentic_rag = no_multi_agent; no_policy_agent = no_policy_rules = no_guard_rules; full_system = no_safety_rules; no_governance = standard_rag = rag_threshold = llm_no_rag (in this replay).

### C. Reproducibility check
`run_evaluation.py --from-signals ... --evaluate-all` run twice: identical `predictions.jsonl` (SHA-256 `8d3f0312...`), and every aggregate JSON carries that hash.
Git commit is recorded as "unavailable" because the uploaded zip has no `.git`.
Outputs: `results/runs/replay_smoke_*` (with `NOTE.md`) and `results/retrieval_offline/`.

## 5. To produce the real thesis results (on your machine)

```bash
python -m eval.link_evidence_ids --dry-run        # check ids; then python -m eval.rebuild_index
ollama serve &                                    # qwen2.5:3b-instruct pulled
python run_evaluation.py --dataset educational_v2 --seed 42 123 2024 \
    --reasoning-mode ollama --evaluate-all --save-results
```
This writes `results/runs/<timestamp>_<datasethash>/` with config.json (model, seeds, retriever, reranker, thresholds, dataset sha256, git commit), predictions, and all aggregates.

## 6. Not Fixable by Implementation Alone

* **False-premise detection.** Lexical evidence cannot detect a false presupposition. It needs an NLI model or the 3B LLM's own label; the NLI path is implemented but not run, and `contradiction_rate` is "not computed" until it is.
* **Real Hallucination/Groundedness.** Faithfulness needs NLI or human annotation; lexical groundedness is an overlap proxy and rewards extractive answers. Wrong-answer rate uses string containment/token F1 against a gold span.
* **Retrieval ground truth.** Binary and source-passage-only: other passages that also answer count as irrelevant (recall is a lower bound), and graded relevance does not exist, so nDCG is binary-gain.
* **Partial-answerable detection generalisation.** The sub-question splitter is a surface heuristic that matches how the `partially_answerable` category was built ("A? And B?"). Whether it generalises to naturally phrased partial questions is untested; this is a threat to validity.
* **Model limits.** A 3B model's self-reported confidence is weakly informative (original AUC 0.51); the calibrated feature score helps only to the extent the retrieval/coverage features carry signal.
* **Multi-agent claim.** The code is a sequential pipeline: Retriever (dense+BM25+RRF+cross-encoder), one LLM call (answer + answerability reasoning), a deterministic Answerability Agent, a rule-based Policy Agent, and a Decision Controller; clarification text is templated, not LLM-generated; there is no iterative loop. "Multi-agent" is defensible only as modular components with defined inputs/outputs, not as autonomous agents.
* **Not done:** the FastAPI app (`app/services/*`) still uses the old `GovernanceAgent`; answers are still generated for every question before the decision (needed to measure unsupported answers), so evidence sufficiency does not yet gate generation; NLI/semantic groundedness not executed; Dense/Hybrid/Reranker retrieval and end-to-end metrics not executed.
