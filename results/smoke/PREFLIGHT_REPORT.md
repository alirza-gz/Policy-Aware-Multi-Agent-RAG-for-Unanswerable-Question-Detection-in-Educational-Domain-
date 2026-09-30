# PRE-FLIGHT REPORT — Smoke Test Validation

Source: `results\smoke/predictions_all.jsonl` (n=1100 rows, 11 systems, seeds=[42], reasoning=mock).

**Full 2498×3 experiment was NOT run.**

## A. Smoke-test numerical results (all 11 systems)

```
                  accuracy  precision  recall      f1  roc_auc  pr_auc  abstention_precision  abstention_recall  coverage  hallucination_rate  unsupported_answer_rate  correct_abstention_rate  policy_compliance  correct_ANSWER  correct_CLARIFY  correct_ABSTAIN  false_acceptance  false_rejection   EM  token_F1  groundedness   Hit@5  Recall@5     MRR  nDCG@5  n_with_gold_ev
system                                                                                                                                                                                                                                                                                                                                                                                
agentic_rag           0.52        1.0  0.2381  0.3846   0.8402  0.8888                   1.0             0.2381      0.85              0.7619                   0.7619                   0.2381               0.88             1.0           0.0000           0.2000            0.7619              0.0  0.0    0.1030        0.9943  0.9855    0.9855  0.9353  0.9480              69
full_system           0.64        1.0  0.4286  0.6000   0.8402  0.8888                   1.0             0.2381      0.73              0.5714                   0.5714                   0.2381               1.00             1.0           0.3333           0.2000            0.5714              0.0  0.0    0.1030        0.9844  0.9855    0.9855  0.9353  0.9480              69
llm_no_rag            0.37        0.0  0.0000  0.0000   0.5000  0.6300                   0.0             0.0000      1.00              0.0000                   0.0000                   0.0000               0.00             1.0           0.0000           0.0000            1.0000              0.0  0.0    0.0000           NaN  0.0000    0.0000  0.0000  0.0000              69
no_answerability      0.37        0.0  0.0000  0.0000   0.8402  0.8888                   0.0             0.0000      1.00              0.7619                   0.7619                   0.0000               0.73             1.0           0.0000           0.0000            1.0000              0.0  0.0    0.1030        0.9844  0.9855    0.9855  0.9353  0.9480              69
no_hybrid             0.64        1.0  0.4286  0.6000   0.8550  0.8990                   1.0             0.2381      0.73              0.5714                   0.5714                   0.2381               1.00             1.0           0.3333           0.2000            0.5714              0.0  0.0    0.1032        0.9839  0.9710    0.9710  0.9280  0.9388              69
no_multi_agent        0.52        1.0  0.2381  0.3846   0.8402  0.8888                   1.0             0.2381      0.85              0.7619                   0.7619                   0.2381               0.88             1.0           0.0000           0.2000            0.7619              0.0  0.0    0.1030        0.9943  0.9855    0.9855  0.9353  0.9480              69
no_policy_agent       0.52        1.0  0.2381  0.3846   0.8402  0.8888                   1.0             0.2381      0.85              0.7619                   0.7619                   0.2381               0.88             1.0           0.0000           0.2000            0.7619              0.0  0.0    0.1030        0.9943  0.9855    0.9855  0.9353  0.9480              69
no_policy_rules       0.64        1.0  0.4286  0.6000   0.8402  0.8888                   1.0             0.2381      0.73              0.5714                   0.5714                   0.2381               1.00             1.0           0.3333           0.2000            0.5714              0.0  0.0    0.1030        0.9844  0.9855    0.9855  0.9353  0.9480              69
no_reranker           0.66        1.0  0.4603  0.6304   0.8642  0.9086                   1.0             0.2063      0.71              0.5397                   0.5397                   0.2063               1.00             1.0           0.4444           0.1778            0.5397              0.0  0.0    0.1078        0.9833  0.9710    0.9710  0.8435  0.8753              69
rag_threshold         0.40        1.0  0.0476  0.0909   0.8402  0.8888                   1.0             0.0476      0.97              0.7619                   0.7619                   0.0476               0.76             1.0           0.0000           0.0667            0.9524              0.0  0.0    0.1030        0.9943  0.9855    0.9855  0.9353  0.9480              69
standard_rag          0.37        0.0  0.0000  0.0000   0.8402  0.8888                   0.0             0.0000      1.00              0.7619                   0.7619                   0.0000               0.73             1.0           0.0000           0.0000            1.0000              0.0  0.0    0.1030        0.9943  0.9855    0.9855  0.9353  0.9480              69
```

Saved machine-readable copy: `results\smoke\preflight_system_comparison.csv`.

AUC score used: `unanswerable_score = 1 - answerability_confidence` (continuous; not hard labels).

## B. Statistical-test validation

### McNemar (FULL vs others) — computed from paired prediction rows

```
  reference           system                    label  n_paired  b_A0B1  c_A1B0  statistic  p_value  effect_size      effect_size_name  from_prediction_rows
full_system     standard_rag binary_detection_correct       100       0      27    25.0370 0.000001        -0.27 proportion_difference                  True
full_system     standard_rag  expected_action_correct       100       0      15    13.0667 0.000301        -0.15 proportion_difference                  True
full_system    rag_threshold binary_detection_correct       100       0      24    22.0417 0.000003        -0.24 proportion_difference                  True
full_system    rag_threshold  expected_action_correct       100       0      12    10.0833 0.001496        -0.12 proportion_difference                  True
full_system      agentic_rag binary_detection_correct       100       0      12    10.0833 0.001496        -0.12 proportion_difference                  True
full_system      agentic_rag  expected_action_correct       100       0       6     4.1667 0.041227        -0.06 proportion_difference                  True
full_system       llm_no_rag binary_detection_correct       100       0      27    25.0370 0.000001        -0.27 proportion_difference                  True
full_system       llm_no_rag  expected_action_correct       100       0      15    13.0667 0.000301        -0.15 proportion_difference                  True
full_system      no_reranker binary_detection_correct       100       3       1     0.2500 0.617075         0.02 proportion_difference                  True
full_system      no_reranker  expected_action_correct       100       3       2     0.0000 1.000000         0.01 proportion_difference                  True
full_system        no_hybrid binary_detection_correct       100       0       0     0.0000 1.000000         0.00 proportion_difference                  True
full_system        no_hybrid  expected_action_correct       100       1       1     0.5000 0.479500         0.00 proportion_difference                  True
full_system no_answerability binary_detection_correct       100       0      27    25.0370 0.000001        -0.27 proportion_difference                  True
full_system no_answerability  expected_action_correct       100       0      15    13.0667 0.000301        -0.15 proportion_difference                  True
full_system  no_policy_agent binary_detection_correct       100       0      12    10.0833 0.001496        -0.12 proportion_difference                  True
full_system  no_policy_agent  expected_action_correct       100       0       6     4.1667 0.041227        -0.06 proportion_difference                  True
```

### Bootstrap 95% CIs (percentile, n_boot=1000, seed=42)

```
      system             metric   mean  ci_low  ci_high   n  n_boot  alpha               method
 full_system           accuracy 0.6400  0.5400   0.7400 100    1000   0.05 percentile_bootstrap
 full_system           coverage 0.7300  0.6400   0.8100 100    1000   0.05 percentile_bootstrap
 full_system hallucination_rate 0.5714  0.4444   0.6984  63    1000   0.05 percentile_bootstrap
standard_rag           accuracy 0.3700  0.2700   0.4700 100    1000   0.05 percentile_bootstrap
standard_rag           coverage 1.0000  1.0000   1.0000 100    1000   0.05 percentile_bootstrap
standard_rag hallucination_rate 0.7619  0.6508   0.8730  63    1000   0.05 percentile_bootstrap
 agentic_rag           accuracy 0.5200  0.4200   0.6200 100    1000   0.05 percentile_bootstrap
 agentic_rag           coverage 0.8500  0.7800   0.9102 100    1000   0.05 percentile_bootstrap
 agentic_rag hallucination_rate 0.7619  0.6508   0.8730  63    1000   0.05 percentile_bootstrap
 no_reranker           accuracy 0.6600  0.5700   0.7600 100    1000   0.05 percentile_bootstrap
 no_reranker           coverage 0.7100  0.6200   0.8000 100    1000   0.05 percentile_bootstrap
 no_reranker hallucination_rate 0.5397  0.4123   0.6667  63    1000   0.05 percentile_bootstrap
  llm_no_rag           accuracy 0.3700  0.2700   0.4700 100    1000   0.05 percentile_bootstrap
  llm_no_rag           coverage 1.0000  1.0000   1.0000 100    1000   0.05 percentile_bootstrap
  llm_no_rag hallucination_rate 0.0000  0.0000   0.0000  63    1000   0.05 percentile_bootstrap
```

Validation: McNemar uses discordant cell counts `(b,c)` from aligned `(id, seed)` pairs; bootstrap resamples per-question indicator vectors. Neither is a hardcoded placeholder.

## C. CLARIFY validation

Total CLARIFY decisions across all systems: **52**

### By system

```
{'full_system': 12, 'no_hybrid': 12, 'no_policy_rules': 12, 'no_reranker': 16}
```

### FULL system CLARIFY count: **12 / 100**

By expected_action:
```
{'CLARIFY': 6, 'ABSTAIN': 6}
```
By category:
```
{'partially_answerable': 6, 'unanswerable': 3, 'out_of_domain': 1, 'false_premise': 2}
```

CLARIFY on gold-answerable questions (FULL): **0**

**Finding:** No CLARIFY on Answerable questions under FULL in this smoke set.

### CLARIFY × category × system

```
         system             category expected_action  n
    full_system        false_premise         ABSTAIN  2
    full_system        out_of_domain         ABSTAIN  1
    full_system partially_answerable         CLARIFY  6
    full_system         unanswerable         ABSTAIN  3
      no_hybrid        false_premise         ABSTAIN  2
      no_hybrid        out_of_domain         ABSTAIN  1
      no_hybrid partially_answerable         CLARIFY  6
      no_hybrid         unanswerable         ABSTAIN  3
no_policy_rules        false_premise         ABSTAIN  2
no_policy_rules        out_of_domain         ABSTAIN  1
no_policy_rules partially_answerable         CLARIFY  6
no_policy_rules         unanswerable         ABSTAIN  3
    no_reranker        false_premise         ABSTAIN  3
    no_reranker        out_of_domain         ABSTAIN  2
    no_reranker partially_answerable         CLARIFY  8
    no_reranker         unanswerable         ABSTAIN  3
```

## D. Reranker validation

Ordering differs FULL vs no_reranker: **99/100** questions.

### Example before/after top-k IDs

```
id=data_science-unans-0085 category=unanswerable gold=[]
  no_reranker (hybrid RRF): ['edu2_data_science.txt#p3', 'edu2_data_science.txt#p144', 'edu2_machine_learning.txt#p219', 'edu2_data_science.txt#p248', 'edu2_statistics.txt#p108']
  FULL (hybrid+rerank):     ['edu2_machine_learning.txt#p219', 'edu2_data_science.txt#p248', 'edu2_data_science.txt#p3', 'edu2_data_science.txt#p144', 'edu2_statistics.txt#p3']
id=probability-ans-0081 category=answerable gold=['edu2_probability.txt#p6']
  no_reranker (hybrid RRF): ['edu2_probability.txt#p6', 'edu2_probability.txt#p238', 'edu2_data_mining.txt#p258', 'edu2_probability.txt#p196', 'edu2_probability.txt#p89']
  FULL (hybrid+rerank):     ['edu2_probability.txt#p6', 'edu2_probability.txt#p196', 'edu2_data_mining.txt#p258', 'edu2_probability.txt#p238', 'edu2_probability.txt#p204']
  gold rank: no_reranker=1 FULL=1
id=data_mining-unans-0099 category=unanswerable gold=[]
  no_reranker (hybrid RRF): ['edu2_machine_learning.txt#p19', 'edu2_data_mining.txt#p245', 'edu2_data_mining.txt#p230', 'edu2_machine_learning.txt#p40', 'edu2_data_mining.txt#p20']
  FULL (hybrid+rerank):     ['edu2_machine_learning.txt#p19', 'edu2_machine_learning.txt#p40', 'edu2_data_mining.txt#p230', 'edu2_data_mining.txt#p245', 'edu2_machine_learning.txt#p109']
id=data_science-part-0010 category=partially_answerable gold=['edu2_data_science.txt#p1']
  no_reranker (hybrid RRF): ['edu2_data_science.txt#p1', 'edu2_data_science.txt#p37', 'edu2_data_science.txt#p82', 'edu2_data_science.txt#p84', 'edu2_data_science.txt#p168']
  FULL (hybrid+rerank):     ['edu2_data_science.txt#p1', 'edu2_data_science.txt#p175', 'edu2_data_science.txt#p48', 'edu2_data_science.txt#p37', 'edu2_data_science.txt#p199']
  gold rank: no_reranker=1 FULL=1
id=machine_learning-unans-0000 category=unanswerable gold=[]
  no_reranker (hybrid RRF): ['edu2_machine_learning.txt#p268', 'edu2_data_mining.txt#p107', 'edu2_machine_learning.txt#p112', 'edu2_machine_learning.txt#p177', 'edu2_machine_learning.txt#p113']
  FULL (hybrid+rerank):     ['edu2_data_mining.txt#p107', 'edu2_machine_learning.txt#p268', 'edu2_machine_learning.txt#p281', 'edu2_machine_learning.txt#p113', 'edu2_machine_learning.txt#p177']
```

### Retrieval metrics: FULL vs no_reranker

```
FULL:        {'n_with_gold': 69, 'k': 5, 'recall@5': 0.9855, 'precision@5': 0.1971, 'hit@5': 0.9855, 'mrr': 0.9353, 'ndcg@5': 0.948}
no_reranker: {'n_with_gold': 69, 'k': 5, 'recall@5': 0.971, 'precision@5': 0.1942, 'hit@5': 0.971, 'mrr': 0.8435, 'ndcg@5': 0.8753}
```

- Δ hit@5 (FULL − no_reranker) = 0.0145
- Δ recall@5 (FULL − no_reranker) = 0.0145
- Δ mrr (FULL − no_reranker) = 0.0918
- Δ ndcg@5 (FULL − no_reranker) = 0.0727

**Assessment:** Reranker improves MRR on this smoke set; ordering changes alone do not imply quality gains.

## E. Hybrid retrieval validation

Live retrieval (smoke questions, no LLM, no rerank), k=5:
```
dense: {'n_with_gold': 69, 'k': 5, 'recall@5': 0.8986, 'precision@5': 0.1797, 'hit@5': 0.8986, 'mrr': 0.7836, 'ndcg@5': 0.8123}
bm25: {'n_with_gold': 69, 'k': 5, 'recall@5': 0.8406, 'precision@5': 0.1681, 'hit@5': 0.8406, 'mrr': 0.7734, 'ndcg@5': 0.7905}
hybrid: {'n_with_gold': 69, 'k': 5, 'recall@5': 0.971, 'precision@5': 0.1942, 'hit@5': 0.971, 'mrr': 0.8435, 'ndcg@5': 0.8753}
```

From stored ablation predictions (note: `no_hybrid` = dense+**rerank**; `full_system` = hybrid+**rerank**):
```
no_hybrid (dense+rerank): {'n_with_gold': 69, 'k': 5, 'recall@5': 0.971, 'precision@5': 0.1942, 'hit@5': 0.971, 'mrr': 0.928, 'ndcg@5': 0.9388}
full_system (hybrid+rerank): {'n_with_gold': 69, 'k': 5, 'recall@5': 0.9855, 'precision@5': 0.1971, 'hit@5': 0.9855, 'mrr': 0.9353, 'ndcg@5': 0.948}
```

## F. Agentic RAG assessment

Decision-layer equivalence on smoke: `agentic_rag` ≡ `no_policy_agent` = **True**; `agentic_rag` ≡ `no_multi_agent` = **True**.

### Is this acceptable for the advisor's "Agentic RAG" baseline?

**Partially acceptable, with explicit caveats — not a full agentic system.**

What it is today:
- One retrieval pass + one LLM reasoner call.
- The reasoner self-governs via `is_answerable` / `needs_clarification`.
- No Policy/Governance Agent thresholds or safety rules.
- Fair against FULL on the same hybrid+rerank signals.

What the advisor wording typically implies but is **missing**:
- Iterative retrieve→reason→retrieve loops
- Tool-calling / query reformulation / multi-hop planning

### Smallest change to make Agentic RAG genuinely iterative (DO NOT implement now)

Add a thin loop **only** for the `agentic_rag` baseline (leave FULL unchanged):
1. Start with hybrid+rerank top-k.
2. Reasoner returns answerability + optional `reformulated_query` / `needs_more_retrieval` flag (extend mock/ollama schema minimally).
3. If flagged and `iteration < max_iters` (e.g. 2–3), re-retrieve with reformulated query, merge/rerank, reason again.
4. Final action still from reasoner self-governance (no Policy Agent).
5. Log `n_iterations`, queries, and retrieved IDs per hop.

This isolates "agentic iteration" as a baseline property without redesigning FULL's GovernanceAgent semantics.

## G. Dataset integrity

- Questions: **2498** (required 2498) → PASS
- Categories: {'partially_answerable': 375, 'false_premise': 375, 'unanswerable': 500, 'answerable': 1000, 'out_of_domain': 248}
- Five required categories present: PASS
- expected_action counts: {'CLARIFY': 375, 'ABSTAIN': 1123, 'ANSWER': 1000}
- Category↔expected_action↔gold_answerable consistency errors: **0** → PASS
- Duplicate IDs: **0** → PASS
- Evidence IDs linked: **1750** (expected 1750) → PASS
- Answerable with evidence_passage but missing evidence_ids: **0** → PASS
- Smoke predictions vs JSONL label fields mismatches: **0** → PASS

## H. Experimental fairness

| system | llm | embed | retrieval_mode | reranked | rag_threshold | reasoning |
|---|---|---|---|---|---|---|
| agentic_rag | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| full_system | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| llm_no_rag | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | none | False | 0.2 | mock |
| no_answerability | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| no_hybrid | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | dense | True | 0.2 | mock |
| no_multi_agent | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| no_policy_agent | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| no_policy_rules | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| no_reranker | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | False | 0.2 | mock |
| rag_threshold | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |
| standard_rag | qwen2.5:3b-instruct | all-MiniLM-L6-v2 | hybrid | True | 0.2 | mock |

**Shared:** dataset slice (same 100 IDs/seed), corpus/index, embed model, LLM id field, seeds, rag_threshold config value logged.

**Unavoidable differences:**
- `llm_no_rag`: no retrieval (`retrieval_mode=none`, empty passages).
- `no_reranker`: hybrid without CE rerank → separate reasoner pass.
- `no_hybrid`: dense+rerank → separate reasoner pass.
- Decision-layer systems sharing `(hybrid, rerank=True)` reuse one signal cache (fair within that profile).
- Smoke used `reasoning_mode=mock` (deterministic lexical heuristic). Full run must use `ollama` for thesis claims about LLM behavior.

## I. Remaining risks before full 2498 × 3-seed run

1. **Smoke uses mock reasoner** — metrics (esp. CLARIFY, AUC, answer quality) will shift under Ollama; treat smoke as pipeline correctness, not thesis numbers.
2. **Agentic RAG is a single-agent proxy** — document clearly; consider iterative loop before claiming "agentic" in the thesis narrative (not implemented).
3. **Reranker may not improve IR metrics** on reconstructed evidence-only corpus (already near-ceiling Hit@K); ablation value may show more in generation/governance than in Recall@K.
4. **Runtime/cost:** 4 retrieval profiles × 2498 × 3 seeds of Ollama calls; ensure Ollama stable and model pulled.
5. **Corpus = evidence passages only** (held-out excluded) — not full Wikipedia; document as limitation.
6. **`no_policy_agent` / `no_multi_agent` / `agentic_rag` are redundant** at the decision layer — fine for table completeness, but do not over-interpret as three independent findings.
7. Cross-encoder download/runtime must succeed on the full-run machine.

## Verdict

### READY for full 2498 × 3-seed run?

**CONDITIONAL YES — pipeline is ready; narrative caveats remain.**

The evaluation harness, all 11 systems, metrics, McNemar/bootstrap, evidence IDs, and ablations execute correctly on the smoke set.

**Before claiming thesis results you must:**
- Run with `--reasoning-mode ollama` (not mock).
- Explicitly document Agentic RAG as a single-agent non-iterative proxy (or implement the iterative loop first).
- Not over-claim reranker IR gains if Hit@K is already saturated.

No blocking data-integrity failures detected.
