# NOTE — decision-layer REPLAY, not an end-to-end run
Produced by `eval/replay_from_predictions.py` + `run_evaluation.py --from-signals` from
`results/smoke/predictions_all.jsonl` (mock reasoner, 100 questions, seed 42, one run).
Retrieval + reasoner outputs are the stored ones; only the NEW decision layer was executed.
Test split = 62 questions; thresholds/score model fit on the 38 validation questions (grouped by source_document).
Do not cite these numbers as results of the Ollama/qwen2.5 system. Retrieval metrics are absent by design (depth-limited lists).
